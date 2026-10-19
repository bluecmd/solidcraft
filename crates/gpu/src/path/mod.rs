//! Native progressive product rendering. Software BVH traversal needs no ray-query
//! features. Every worker quantum submits at most 256K paths, then checks cancellation.
use solidcraft_geom::Vec3;
use solidcraft_render::{
    Availability, ExecutorDescriptor, ExecutorId, ProductScene, RenderFeature, RenderFormat, RenderLimits, RenderMode, RenderOperation,
    RenderProfile, ScratchLayout,
    job::{
        Admission, CancelToken, ErrorCode, ImageExecutor, ImageHost, ImageRequest, ImageWork, JobError, JobOptions, Phase, PngTask, Progress, Result,
        TaskPoll, ToneMap, admit_geometry,
    },
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

mod denoise;
mod scene;
#[cfg(test)]
mod tests;
mod textures;
const SHADER: &str = concat!(include_str!("path.wgsl"), "\n", include_str!("shadows.wgsl"), "\n", include_str!("textures.wgsl"));
// LOCAL HWRT PROTOTYPE: SC_HWRT=1 swaps the f32 BVH traversal for wgpu ray queries (opaque scenes only).
fn hwrt() -> bool {
    std::env::var("SC_HWRT").is_ok_and(|v| v == "1")
}
const HW_TRACE: &str = r#"
@group(1) @binding(0) var scene_as: acceleration_structure;
fn trace_ordered_within(o: vec3<f32>, d: vec3<f32>, any_hit: bool, excluded: u32, minimum: f32, inside: bool, limit: f32, blocker: f32) -> Hit {
    var flags = 0u;
    if any_hit || blocker >= limit { flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT; }
    var q: ray_query;
    rayQueryInitialize(&q, scene_as, RayDesc(flags, 0xffu, minimum, limit, o, d));
    while rayQueryProceed(&q) {}
    let c = rayQueryGetCommittedIntersection(&q);
    if c.kind == RAY_QUERY_INTERSECTION_NONE { return Hit(limit, 0xffffffffu, vec3<f32>(0.0)); }
    return Hit(c.t, c.primitive_index, vec3<f32>(1.0-c.barycentrics.x-c.barycentrics.y, c.barycentrics.x, c.barycentrics.y));
}
"#;
fn shader_source() -> String {
    if hwrt() {
        format!("enable wgpu_ray_query;\n{}\n{HW_TRACE}", SHADER.replace("fn trace_ordered_within(", "fn trace_ordered_within_software("))
    } else {
        SHADER.to_string()
    }
}
const MAX_TRIANGLES: u64 = 1_048_576;
const PARAMETER_BYTES: usize = 448;
// Eight u32 words: count low, pixel, bounce, sample, code, count high, first-writer flag, spare.
const FAILURE_BYTES: u64 = 32;
// A path that cannot classify its glass medium ends there. Isolated paths at the
// limits of f32 geometry are tolerated and logged; more than this fraction of all
// paths (any systematic error such as a camera inside glass) fails the job.
const FAILURE_FRACTION: f64 = 1e-6;
const TILE: u32 = 262_144;
const TIMEOUT: Duration = Duration::from_secs(30);
// Software Vulkan drivers can compile a pipeline at its first dispatch, which on a cold
// shader cache exceeds the per-dispatch hang timeout. Setup dispatches once with no work
// and allows this longer wait, so jobs keep the short one.
const SETUP_TIMEOUT: Duration = Duration::from_secs(600);

fn gpu(e: impl std::fmt::Display) -> JobError {
    JobError::new(ErrorCode::HostUnavailable, e.to_string())
}

fn scopes(device: &wgpu::Device) -> [wgpu::ErrorScopeGuard; 3] {
    [wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal].map(|f| device.push_error_scope(f))
}
fn checked(scopes: [wgpu::ErrorScopeGuard; 3]) -> Result<()> {
    let mut error = None;
    for scope in scopes.into_iter().rev() {
        if let Some(e) = pollster::block_on(scope.pop()) {
            error = Some(e);
        }
    }
    error.map_or(Ok(()), |e| Err(gpu(e)))
}

/// Open and validate a native device before advertising the executor. Failure leaves the
/// caller's CPU host usable. Respects SOLIDCRAFT_GPU, including the software adapter policy.
pub fn host() -> Result<ImageHost> {
    let (adapter, device, queue) = open_headless()?;
    log::info!("gpu-path adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().device_type);
    ImageHost::with_executor(Arc::new(Executor::new(device, queue)?))
}

/// Open an independent job device on the viewport's selected adapter. In particular,
/// software fallback and safe mode must not rediscover a failing hardware driver.
/// An explicit incompatible SOLIDCRAFT_GPU policy leaves this executor unavailable.
pub fn host_on_adapter(adapter: &wgpu::Adapter) -> Result<ImageHost> {
    let policy = crate::select::Policy::from_env().map_err(gpu)?;
    let info = adapter.get_info();
    if !policy.admits(crate::select::tier_of(info.device_type)) {
        return Err(gpu(format!("viewport adapter does not satisfy SOLIDCRAFT_GPU={}", policy.name())));
    }
    let (device, queue) = open_device(adapter)?;
    log::info!("gpu-path viewport adapter: {} ({:?})", info.name, info.device_type);
    ImageHost::with_executor(Arc::new(Executor::new(device, queue)?))
}

// Path resources need six storage buffers (geometry, BVH, accumulation, HDRI,
// textures, medium-failure counts). Request that limit only on this independent path device: raster and
// tessellation keep their existing four-buffer downlevel requirement.
const PATH_STORAGE_BUFFERS: u32 = 6;
fn device_limits(supported: wgpu::Limits) -> Result<wgpu::Limits> {
    if supported.max_storage_buffers_per_shader_stage < PATH_STORAGE_BUFFERS {
        return Err(gpu("gpu-path requires six storage buffers per shader stage"));
    }
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(supported);
    limits.max_storage_buffers_per_shader_stage = PATH_STORAGE_BUFFERS;
    Ok(limits)
}
fn open_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let mut required_limits = device_limits(adapter.limits())?;
    if hwrt() {
        required_limits = required_limits.using_minimum_supported_acceleration_structure_values();
    }
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("solidcraft progressive path"),
        required_features: if hwrt() { crate::depth_features(adapter) | wgpu::Features::EXPERIMENTAL_RAY_QUERY } else { crate::depth_features(adapter) },
        required_limits,
        experimental_features: if hwrt() { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
        ..Default::default()
    }))
    .map_err(gpu)
}
fn open_headless() -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let policy = crate::select::Policy::from_env().map_err(gpu)?;
    let (adapter, _) = crate::select::pick(policy).ok_or_else(|| gpu(format!("no gpu-path adapter satisfies {}", policy.name())))?;
    let (device, queue) = open_device(&adapter)?;
    Ok((adapter, device, queue))
}

struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    // Built and warmed by the first denoised job, then shared by later ones.
    denoise: std::sync::OnceLock<denoise::Pipelines>,
    layout: wgpu::BindGroupLayout,
    as_layout: wgpu::BindGroupLayout,
}
struct Executor {
    context: Arc<Context>,
}
impl Executor {
    fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        device_limits(device.limits())?;
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| log::error!("gpu-path: {e}")));
        let scopes = scopes(&device);
        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gpu-path"), source: wgpu::ShaderSource::Wgsl(shader_source().into()) });
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding != 3 && binding != 6 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gpu-path"), entries: &entries });
        let as_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-path hwrt"),
            entries: if hwrt() {
                &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::AccelerationStructure { vertex_return: false }, count: None }]
            } else {
                &[]
            },
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-path"),
            bind_group_layouts: &if hwrt() { vec![Some(&layout), Some(&as_layout)] } else { vec![Some(&layout)] },
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-path"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        checked(scopes)?;
        if !hwrt() {
            warm_up(&device, &queue, &pipeline, &placeholder_bind(&device, &layout, 7)?)?;
        }
        Ok(Self { context: Arc::new(Context { device, queue, pipeline, layout, as_layout, denoise: std::sync::OnceLock::new() }) })
    }
}

// Small zeroed bindings: binding 0 is a uniform whose zero work count makes every
// invocation return at once; the others are storage buffers.
fn placeholder_bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, count: u32) -> Result<wgpu::BindGroup> {
    let scopes = scopes(device);
    let buffers: Vec<_> = (0..count)
        .map(|binding| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-path warm-up"),
                size: if binding == 0 { PARAMETER_BYTES as u64 } else { 256 },
                usage: if binding == 0 { wgpu::BufferUsages::UNIFORM } else { wgpu::BufferUsages::STORAGE },
                mapped_at_creation: false,
            })
        })
        .collect();
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: binding as u32, resource: buffer.as_entire_binding() })
        .collect();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gpu-path warm-up"), layout, entries: &entries });
    checked(scopes)?;
    Ok(bind)
}

fn warm_up(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup) -> Result<()> {
    let scopes = scopes(device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu-path warm-up") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gpu-path warm-up"), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let submission = queue.submit([encoder.finish()]);
    checked(scopes)?;
    device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(SETUP_TIMEOUT) }).map_err(gpu)?;
    Ok(())
}

// LOCAL HWRT PROTOTYPE: triangles exactly as path.wgsl's triangle() decodes them, in one BLAS.
fn build_hw(context: &Context, geometry: &[u8]) -> Result<(wgpu::BindGroup, wgpu::Tlas, wgpu::Blas, wgpu::Buffer)> {
    let start = Instant::now();
    let word = |i: usize| u32::from_le_bytes([geometry[i * 4], geometry[i * 4 + 1], geometry[i * 4 + 2], geometry[i * 4 + 3]]);
    let (count, vertex_offset) = (word(0) as usize, word(1) as usize);
    let mut verts = Vec::with_capacity(count * 36);
    for t in 0..count {
        for k in 0..3 {
            let v = vertex_offset + word(4 + t * 4 + k) as usize * 8;
            for c in 0..3 {
                verts.extend_from_slice(&word(v + c).to_le_bytes());
            }
        }
    }
    let device = &context.device;
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("hwrt vertices"), contents: &verts, usage: wgpu::BufferUsages::BLAS_INPUT });
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: (count * 3) as u32,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor { label: Some("hwrt"), flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor { label: Some("hwrt"), max_instances: 1, flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build });
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0, 0xff));
    let mut e = device.create_command_encoder(&Default::default());
    e.build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vb,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        std::iter::once(&tlas),
    );
    let submission = context.queue.submit([e.finish()]);
    device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(TIMEOUT) }).map_err(gpu)?;
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hwrt"),
        layout: &context.as_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: tlas.as_binding() }],
    });
    log::info!("hwrt: {count} triangles, BLAS+TLAS {:.1} ms", start.elapsed().as_secs_f64() * 1000.0);
    Ok((bind, tlas, blas, vb))
}

fn medium_failures_tolerated(failed: u64, paths: u64) -> bool {
    failed as f64 <= paths as f64 * FAILURE_FRACTION
}

fn descriptor() -> Result<ExecutorDescriptor> {
    Ok(ExecutorDescriptor {
        id: ExecutorId::new("gpu-path").map_err(gpu)?,
        availability: Availability::Available,
        profiles: vec![RenderProfile {
            mode: RenderMode::Preview,
            operation: RenderOperation::Image,
            formats: vec![RenderFormat::Png],
            features: vec![RenderFeature::Pbr, RenderFeature::Progressive, RenderFeature::Refraction, RenderFeature::Hdri],
        }],
        limits: RenderLimits::default(),
        scratch: ScratchLayout { sample_scale: 1, sample_bytes_per_pixel: 16, output_bytes_per_pixel: 16, encoding_bytes_per_pixel: 16 },
    })
}

fn admission(scene: &ProductScene, request: &ImageRequest, options: JobOptions) -> Result<Admission> {
    options.validate()?;
    if let Some(environment) = &request.environment {
        environment.validate()?;
    }
    let camera = request.camera;
    if options.lens.is_some_and(|lens| lens.aperture_radius_mm > 0.0)
        && (camera.distance < 1e-6 || !(1e-6..=1e6).contains(&(camera.half_height() / camera.distance)) || camera.fov == 0.0)
    {
        return Err(JobError::new(
            ErrorCode::InvalidRequest,
            "thin-lens depth of field requires a bounded perspective camera (distance >= 0.000001 mm and tan(fov/2) in 0.000001..1000000)",
        ));
    }
    if !camera.fov.is_finite()
        || !(0.0..std::f64::consts::PI).contains(&camera.fov)
        || !camera.half_height().is_finite()
        || camera.half_height() > 1e12
        || camera.distance > 1e9
    {
        return Err(JobError::new(ErrorCode::InvalidRequest, "gpu-path requires a bounded finite camera"));
    }
    let mut admission = admit_geometry(scene, request, MAX_TRIANGLES)?;
    if scene
        .meshes
        .iter()
        .any(|m| (m.opacity != 1.0 && m.transmission.is_none()) || m.faces.iter().any(|f| f.opacity != 1.0 && f.transmission.is_none()))
    {
        return Err(JobError::new(ErrorCode::InvalidScene, "gpu-path requires explicit transmission for nonopaque preview appearances"));
    }
    // Peak preparation: f64 triangles + nodes + serialized upload. After preparation:
    // serialized + device geometry + accumulation/readback/RGBA/PNG. These phases don't overlap.
    let pixels = u64::from(request.width) * u64::from(request.height);
    let denoise_bytes = pixels * if options.denoise { denoise::BYTES_PER_PIXEL } else { 0 };
    admission.pixel_scratch_bytes = pixels * 64 + denoise_bytes;
    let vertex_bytes = scene.meshes.iter().map(|m| m.mesh.positions.len() as u64 * 32).sum::<u64>();
    let triangle_preparation =
        (admission.triangles * 400_u64.max(scene::planar_preparation_bytes_per_triangle() as u64)).max(scene::medium_preparation_bytes(scene));
    admission.preparation_bytes = (triangle_preparation + vertex_bytes * 2 + admission.face_appearances * 8)
        .max(admission.triangles * 232 + vertex_bytes * 2 + pixels * 32 + denoise_bytes)
        .max(admission.triangles * 116 + vertex_bytes + admission.pixel_scratch_bytes)
        + 1_048_576;
    admission.preparation_bytes += request.environment.as_ref().map_or(0, |source| source.admitted_bytes());
    if let Some(textures) = &request.textures {
        textures.validate_scene(scene)?;
        admission.preparation_bytes += textures.working_bytes();
    }
    admission.estimated_total_bytes = admission.geometry_bytes + admission.preparation_bytes;
    if admission.estimated_total_bytes > 512 * 1024 * 1024 {
        return Err(JobError::new(ErrorCode::LimitExceeded, "gpu-path job exceeds its 512 MiB admission estimate"));
    }
    Ok(admission)
}

impl ImageExecutor for Executor {
    fn descriptor(&self) -> Result<ExecutorDescriptor> {
        descriptor()
    }
    fn admit(&self, scene: &ProductScene, request: &ImageRequest, options: JobOptions) -> Result<Admission> {
        let a = admission(scene, request, options)?;
        let limit = self.context.device.limits().max_storage_buffer_binding_size;
        let geometry_bytes = 16 + a.triangles.max(1) * 84 + scene.meshes.iter().map(|m| m.mesh.positions.len() as u64 * 32).sum::<u64>();
        let texture_bytes = request
            .textures
            .as_ref()
            .map_or(16, |t| t.summary().linear_pixel_bytes + 4 * (5 + scene.meshes.len() as u64 + t.bindings().len() as u64 * 16));
        let environment_bytes =
            request.environment.as_ref().map_or(48, |source| 16 + u64::from(source.info().width) * u64::from(source.info().height) * 32);
        if texture_bytes > limit
            || environment_bytes > limit
            || geometry_bytes > limit
            || u64::from(request.width) * u64::from(request.height) * 16 > limit
            || (options.denoise && u64::from(request.width) * u64::from(request.height) * 48 > limit)
        {
            return Err(JobError::new(ErrorCode::LimitExceeded, "gpu-path storage buffer exceeds this device's limit"));
        }
        Ok(a)
    }
    fn begin(&self, scene: ProductScene, request: ImageRequest, options: JobOptions, cancel: CancelToken) -> Result<Box<dyn ImageWork>> {
        self.admit(&scene, &request, options)?;
        Ok(Box::new(Task::new(self.context.clone(), scene, request, options, cancel)?))
    }
}

struct Task {
    context: Arc<Context>,
    request: ImageRequest,
    options: JobOptions,
    cancel: CancelToken,
    uniform: wgpu::Buffer,
    accumulation: wgpu::Buffer,
    failures: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind: wgpu::BindGroup,
    hw: Option<(wgpu::BindGroup, wgpu::Tlas, wgpu::Blas, wgpu::Buffer)>,
    denoiser: Option<denoise::Resources>,
    sample: u32,
    offset: u32,
    epsilon: f32,
    png: Option<PngTask>,
    started: Instant,
    // Uniform environment is used only by the numerical furnace test, never advertised as HDRI.
    environment: f32,
    environment_bottom_only: bool,
    studio: f32,
    #[cfg(test)]
    metrics: (f64, u64, f64),
}
impl Task {
    fn new(context: Arc<Context>, scene: ProductScene, mut request: ImageRequest, options: JobOptions, cancel: CancelToken) -> Result<Self> {
        cancel.check()?;
        let start = Instant::now();
        if let Some(resources) = &request.textures {
            resources.validate_scene(&scene)?;
        }
        let texture_upload = textures::flatten(request.textures.as_deref(), scene.meshes.len(), request.camera.target, &cancel)?;
        request.textures = None; // GPU upload owns decoded pixels; release captured source bytes.
        let upload = scene::flatten(&scene, request.camera.target, &cancel)?;
        let bvh_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut upload_bytes = (upload.geometry.len() + upload.nodes.len() + texture_upload.len()) as u64;
        let uploading = Instant::now();
        drop(scene);
        let device = &context.device;
        let scopes = [wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal].map(|f| device.push_error_scope(f));
        let make =
            |label, bytes: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage });
        let mut environment_bytes = Vec::new();
        if let Some(source) = &request.environment {
            let prepared = source.prepare(&cancel)?;
            environment_bytes = scene::reserved(16 + prepared.texels().len() * 32)?;
            let [width, height] = prepared.dimensions();
            for x in [width, height, width * height, 0] {
                environment_bytes.extend_from_slice(&x.to_le_bytes());
            }
            for (index, texel) in prepared.texels().iter().enumerate() {
                if index % 1024 == 0 {
                    cancel.check()?;
                }
                for x in texel.rgb.into_iter().chain([texel.pdf, texel.threshold]) {
                    environment_bytes.extend_from_slice(&x.to_le_bytes());
                }
                environment_bytes.extend_from_slice(&texel.alias.to_le_bytes());
                environment_bytes.extend_from_slice(&[0; 8]);
            }
        } else {
            environment_bytes.resize(48, 0);
        }
        upload_bytes += environment_bytes.len() as u64;
        let environment = make("path environment", &environment_bytes, wgpu::BufferUsages::STORAGE);
        drop(environment_bytes);
        // LOCAL BENCHMARK DUMP ONLY - never commit.
        if let Ok(dir) = std::env::var("SC_DUMP") { let _ = std::fs::write(format!("{dir}/geometry.bin"), &upload.geometry); let _ = std::fs::write(format!("{dir}/nodes.bin"), &upload.nodes); }
        let hw = if hwrt() { Some(build_hw(&context, &upload.geometry)?) } else { None };
        let triangles = make("path triangles", &upload.geometry, wgpu::BufferUsages::STORAGE);
        let nodes = make("path BVH", &upload.nodes, wgpu::BufferUsages::STORAGE);
        let texture_buffer = make("path opaque textures", &texture_upload, wgpu::BufferUsages::STORAGE);
        let uniform = make("path parameters", &[0; PARAMETER_BYTES], wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let size = u64::from(request.width) * u64::from(request.height) * 16;
        let accumulation = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("path accumulation"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Inconsistent-medium path count and the first such path's location (see read_linear).
        let failures = make(
            "path medium failures",
            &[0; FAILURE_BYTES as usize],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("path readback"),
            size: size + FAILURE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<_> = [&uniform, &triangles, &nodes, &accumulation, &environment]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: binding as u32, resource: buffer.as_entire_binding() })
            .collect();
        entries.push(wgpu::BindGroupEntry { binding: 5, resource: texture_buffer.as_entire_binding() });
        entries.push(wgpu::BindGroupEntry { binding: 6, resource: failures.as_entire_binding() });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gpu-path"), layout: &context.layout, entries: &entries });
        let denoiser = if options.denoise {
            Some(denoise::Resources::new(&context, &uniform, [&triangles, &nodes, &environment, &texture_buffer, &failures], &accumulation, size)?)
        } else {
            None
        };
        let mut error = None;
        for scope in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                error = Some(e);
            }
        }
        if let Some(e) = error {
            return Err(gpu(e));
        }
        let submission = context.queue.submit([]);
        device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(TIMEOUT) }).map_err(gpu)?;
        let upload_ms = uploading.elapsed().as_secs_f64() * 1000.0;
        log::info!("gpu-path BVH {bvh_ms:.3} ms, upload {upload_bytes} bytes in {upload_ms:.3} ms");
        cancel.check()?;
        Ok(Self {
            context,
            request,
            options,
            cancel,
            uniform,
            accumulation,
            failures,
            readback,
            bind,
            hw,
            denoiser,
            sample: 0,
            offset: 0,
            epsilon: upload.epsilon,
            png: None,
            started: Instant::now(),
            environment: 0.0,
            environment_bottom_only: false,
            studio: 1.0,
            #[cfg(test)]
            metrics: (bvh_ms, upload_bytes, upload_ms),
        })
    }
    fn parameters(&self, count: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PARAMETER_BYTES);
        for x in [
            self.request.width,
            self.request.height,
            self.sample,
            self.offset,
            self.options.seed,
            self.options.max_depth,
            count,
            u32::from(self.environment_bottom_only),
        ] {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let camera = self.request.camera;
        let (right, up, back) = camera.basis();
        let perspective = camera.fov > 0.0;
        let eye = back * (camera.distance * if perspective { 1.0 } else { 4.0 });
        scene::vector(&mut bytes, eye, if perspective { 1.0 } else { 0.0 });
        scene::vector(
            &mut bytes,
            right * camera.half_height() * (f64::from(self.request.width) / f64::from(self.request.height)),
            self.options.lens.map_or(0.0, |l| l.aperture_radius_mm as f32),
        );
        scene::vector(&mut bytes, up * camera.half_height(), self.options.lens.map_or(0.0, |l| l.focus_distance_mm as f32));
        scene::vector(&mut bytes, back, camera.distance as f32);
        scene::vector(&mut bytes, Vec3::new(f64::from(self.epsilon), f64::from(self.environment), f64::from(self.studio)), 0.0);
        let lighting = self.options.lighting.unwrap_or_default();
        scene::vector(
            &mut bytes,
            Vec3::new(lighting.environment[0], lighting.environment[1], lighting.environment[2]),
            if self.options.lighting.is_some() || self.request.environment.is_some() { 1.0 } else { 0.0 },
        );
        scene::vector(&mut bytes, Vec3::ZERO, lighting.area_lights.iter().count() as f32);
        for light in lighting.area_lights.iter() {
            let vector = |[x, y, z]: [f64; 3]| Vec3::new(x, y, z);
            scene::vector(&mut bytes, vector(light.center) - camera.target, light.area() as f32);
            scene::vector(&mut bytes, vector(light.u), 0.0);
            scene::vector(&mut bytes, vector(light.v), 0.0);
            scene::vector(&mut bytes, vector(light.radiance), 0.0);
        }
        bytes.resize(400, 0);
        if let Some(source) = &self.request.environment {
            let info = source.info();
            scene::vector(&mut bytes, Vec3::new(info.rotation_rad.cos(), info.rotation_rad.sin(), info.intensity), 1.0);
        } else {
            scene::vector(&mut bytes, Vec3::ZERO, 0.0);
        }
        if let Some(sun) = lighting.sun {
            scene::vector(&mut bytes, sun.direction(), sun.angular_radius_rad.cos() as f32);
            let [r, g, b] = sun.radiance;
            scene::vector(&mut bytes, Vec3::new(r, g, b), (1.0 / sun.solid_angle()) as f32);
        } else {
            bytes.resize(PARAMETER_BYTES, 0);
        }

        bytes
    }
    fn dispatch(&mut self) -> Result<()> {
        self.cancel.check()?;
        let pixels = self.request.width * self.request.height;
        let count = (pixels - self.offset).min(TILE);
        let context = &self.context;
        let scopes = scopes(&context.device);
        if let Ok(dir) = std::env::var("SC_DUMP") { if self.sample == 0 && self.offset == 0 { let _ = std::fs::write(format!("{dir}/params.bin"), self.parameters(count)); } }
        context.queue.write_buffer(&self.uniform, 0, &self.parameters(count));
        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("path tile") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("path tile"), timestamp_writes: None });
            pass.set_pipeline(&context.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            if let Some((b, ..)) = &self.hw {
                pass.set_bind_group(1, b, &[]);
            }
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let submission = context.queue.submit([encoder.finish()]);
        checked(scopes)?;
        context.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(TIMEOUT) }).map_err(gpu)?;
        self.cancel.check()?;
        self.offset += count;
        if self.offset == pixels {
            self.sample += 1;
            self.offset = 0;
        }
        Ok(())
    }
    fn read_linear(&self) -> Result<Vec<[f32; 4]>> {
        self.read_linear_buffer(&self.accumulation)
    }
    fn read_linear_buffer(&self, source: &wgpu::Buffer) -> Result<Vec<[f32; 4]>> {
        self.cancel.check()?;
        let context = &self.context;
        let scopes = scopes(&context.device);
        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("path readback") });
        encoder.copy_buffer_to_buffer(source, 0, &self.readback, 0, self.accumulation.size());
        encoder.copy_buffer_to_buffer(&self.failures, 0, &self.readback, self.accumulation.size(), FAILURE_BYTES);
        let submission = context.queue.submit([encoder.finish()]);
        checked(scopes)?;
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        context.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(TIMEOUT) }).map_err(gpu)?;
        rx.try_recv().map_err(gpu)?.map_err(gpu)?;
        let data = slice.get_mapped_range().map_err(gpu)?;
        let (pixels, failures) = data.split_at(data.len().saturating_sub(FAILURE_BYTES as usize));
        let word = |i: usize| failures.as_chunks::<4>().0.get(i).map_or(0, |w| u32::from_le_bytes(*w));
        let failed = u64::from(word(0)) | (u64::from(word(5)) << 32);
        if failed > 0 {
            let paths = u64::from(self.request.width) * u64::from(self.request.height) * u64::from(self.sample.max(1));
            let detail = format!("{failed} of {paths} paths, first at pixel {}, bounce {}, sample {}, code {}", word(1), word(2), word(3), word(4));
            if !medium_failures_tolerated(failed, paths) {
                return Err(JobError::new(
                    ErrorCode::InvalidScene,
                    format!(
                        "inconsistent glass medium paths ({detail}): camera must start outside volumes; overlapping solids or more than 16 nested media are unsupported"
                    ),
                ));
            }
            log::warn!("gpu-path ended {detail} at an unclassifiable glass medium");
        }
        let mut values = scene::reserved((self.request.width * self.request.height) as usize)?;
        for pixel in pixels.as_chunks::<16>().0 {
            let mut value = [0.0; 4];
            for (v, bytes) in value.iter_mut().zip(pixel.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*bytes) / self.sample as f32;
                if !v.is_finite() || *v < 0.0 {
                    return Err(JobError::new(ErrorCode::Internal, "nonfinite or negative path radiance"));
                }
            }
            values.push(value);
        }
        drop(data);
        self.readback.unmap();
        self.cancel.check()?;
        Ok(values)
    }
}
#[derive(Clone, Copy)]
struct Display {
    exposure: f64,
    tone_map: ToneMap,
    background: [f64; 3],
    transparent: bool,
}
impl Display {
    fn encode(self, input: &[[f32; 4]], output: &mut [u8], cancel: &CancelToken) -> Result<()> {
        for (index, (pixel, rgba)) in input.iter().zip(output.as_chunks_mut::<4>().0).enumerate() {
            if index % 4096 == 0 {
                cancel.check()?;
            }
            let alpha = f64::from(pixel[3]).clamp(0.0, 1.0);
            for ((radiance, background), out) in pixel.iter().take(3).zip(self.background).zip(rgba.iter_mut()) {
                let c = if alpha > 0.0 { f64::from(*radiance) * self.exposure / alpha } else { 0.0 };
                let mapped = self.tone_map.map(c);
                let value = if self.transparent { mapped } else { mapped * alpha + background * (1.0 - alpha) };
                let encoded = if value <= 0.0031308 { value * 12.92 } else { 1.055 * value.powf(1.0 / 2.4) - 0.055 };
                *out = (encoded.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            rgba[3] = if self.transparent { (alpha * 255.0).round() as u8 } else { 255 };
        }
        Ok(())
    }
}
impl ImageWork for Task {
    fn step(&mut self, units: usize) -> Result<TaskPoll> {
        self.cancel.check()?;
        if let Some(png) = &mut self.png {
            return png.step(units);
        }
        if self.sample < self.options.samples {
            self.dispatch()?;
            return Ok(TaskPoll::Pending(Progress { phase: Phase::Running, done: u64::from(self.sample), total: u64::from(self.options.samples) }));
        }
        log::info!("gpu-path {} spp, {:.3} ms/spp", self.sample, self.started.elapsed().as_secs_f64() * 1000.0 / f64::from(self.sample));
        let mut linear = self.read_linear()?;
        if let Some(denoiser) = &self.denoiser {
            // Diagnose the raw path buffer before filtering: never hide invalid media
            // or nonfinite radiance by replacing a pixel with neighbouring values.
            drop(linear);
            denoiser.run(self)?;
            linear = self.read_linear_buffer(denoiser.output())?;
        }
        let mut rgba = scene::reserved(linear.len() * 4)?;
        rgba.resize(linear.len() * 4, 0);
        let display = Display {
            exposure: 2.0f64.powf(self.options.exposure_ev),
            tone_map: self.options.tone_map,
            background: self.request.background.map(solidcraft_render::surface::srgb8_to_linear),
            transparent: self.request.transparent_background,
        };
        // Pixels are independent: encode contiguous chunks on every core.
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16);
        let chunk = linear.len().div_ceil(workers).max(1);
        let cancel = &self.cancel;
        std::thread::scope(|scope| {
            let jobs: Vec<_> = linear
                .chunks(chunk)
                .zip(rgba.chunks_mut(chunk * 4))
                .map(|(input, output)| scope.spawn(move || display.encode(input, output, cancel)))
                .collect();
            jobs.into_iter()
                .try_for_each(|job| job.join().unwrap_or_else(|_| Err(JobError::new(ErrorCode::Internal, "display encoding worker failed"))))
        })?;
        self.png = Some(PngTask::new(&self.request, rgba, self.cancel.clone())?);
        Ok(TaskPoll::Pending(Progress { phase: Phase::Encoding, done: 0, total: u64::from(self.request.height) }))
    }
}
