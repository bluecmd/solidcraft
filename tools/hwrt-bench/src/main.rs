//! Private benchmark: SolidCraft's software BVH traversal vs wgpu hardware ray queries, on the
//! same rays against the same triangles (dumped from a real gpu-path job with SC_DUMP).
//! Usage: hwrt-bench DUMP_DIR FULL_PATH_WGSL [ITERATIONS]
use std::time::Instant;
use wgpu::util::DeviceExt;

const SW: &str = include_str!("bench.wgsl");
const HW: &str = include_str!("hw.wgsl");
const PATHS: &str = include_str!("paths.wgsl");
const TRACE_SW: &str = include_str!("trace_sw.wgsl");
const TRACE_HW: &str = include_str!("trace_hw.wgsl");
const NONE: u32 = 0xffff_ffff;
const RAY_BYTES: u64 = 64;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    timestamps: bool,
    period: f32,
}

impl Gpu {
    fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
    fn read(&self, src: &wgpu::Buffer) -> Vec<u8> {
        let size = src.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut e = self.device.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([e.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.wait();
        let v = staging.slice(..).get_mapped_range().expect("range").to_vec();
        staging.unmap();
        v
    }
    /// One dispatch; returns (GPU ms from timestamps if available, wall ms).
    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, binds: &[&wgpu::BindGroup], groups: u32) -> (f64, f64) {
        let query = self.timestamps.then(|| self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 }));
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let mut e = self.device.create_command_encoder(&Default::default());
        {
            let tw = query.as_ref().map(|q| wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
            p.set_pipeline(pipeline);
            for (i, b) in binds.iter().enumerate() {
                p.set_bind_group(i as u32, *b, &[]);
            }
            assert!(groups <= 65535, "kernels index by gid.x only");
            p.dispatch_workgroups(groups, 1, 1);
        }
        if let Some(q) = &query {
            e.resolve_query_set(q, 0..2, &resolve, 0);
        }
        let start = Instant::now();
        self.queue.submit([e.finish()]);
        self.wait();
        let wall = start.elapsed().as_secs_f64() * 1000.0;
        let gpu = if query.is_some() {
            let b = self.read(&resolve);
            let t0 = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let t1 = u64::from_le_bytes(b[8..16].try_into().unwrap());
            t1.saturating_sub(t0) as f64 * f64::from(self.period) / 1e6
        } else {
            f64::NAN
        };
        (gpu, wall)
    }
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn entries<'a>(bs: &[&'a wgpu::Buffer]) -> Vec<wgpu::BindGroupEntry<'a>> {
    bs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect()
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}
fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

struct Run {
    label: &'static str,
    gpu_ms: f64,
    wall_ms: f64,
    hits: Vec<u32>,
}
impl Run {
    fn ms(&self) -> f64 {
        if self.gpu_ms.is_nan() { self.wall_ms } else { self.gpu_ms }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dump = std::path::Path::new(&args[1]);
    let path_wgsl = std::fs::read_to_string(&args[2]).expect("path wgsl");
    let iterations: usize = args.get(3).map_or(10, |s| s.parse().expect("iterations"));
    let vertices: u32 = args.get(4).map_or(5, |s| s.parse().expect("vertices"));
    let geometry = std::fs::read(dump.join("geometry.bin")).expect("geometry.bin");
    let nodes = std::fs::read(dump.join("nodes.bin")).expect("nodes.bin");
    let mut params = std::fs::read(dump.join("params.bin")).expect("params.bin");
    params[8..16].fill(0); // sample index, tile offset
    let p = words(&params);
    let (width, height) = (p[0], p[1]);
    let rays_n = width * height;
    let tris = words(&geometry[0..4])[0];

    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("adapter");
    let info = adapter.get_info();
    assert!(adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY), "adapter lacks EXPERIMENTAL_RAY_QUERY");
    let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut features = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    if timestamps {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: adapter.limits(),
        // SAFETY: the single opt-in this benchmark exists to measure; see the SolidCraft decision issue.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        memory_hints: Default::default(),
        trace: Default::default(),
    }))
    .expect("device");
    let period = queue.get_timestamp_period();
    let gpu = Gpu { device, queue, timestamps, period };
    let d = &gpu.device;
    eprintln!("adapter: {} ({:?}, {:?}, driver {} {})", info.name, info.device_type, info.backend, info.driver, info.driver_info);
    eprintln!("{width}x{height} = {rays_n} rays per set, {tris} triangles, timestamps {timestamps}");

    let init = |label, contents: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
    let sto = wgpu::BufferUsages::STORAGE;
    let geometry_b = init("geometry", &geometry, sto);
    let nodes_b = init("nodes", &nodes, sto);
    let params_b = init("params", &params, wgpu::BufferUsages::UNIFORM);
    let empty = |label, size: u64, usage| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
    let rays_b = empty("rays", u64::from(rays_n) * RAY_BYTES, sto | wgpu::BufferUsages::COPY_SRC);
    let hits_b = empty("hits", u64::from(rays_n) * 16, sto | wgpu::BufferUsages::COPY_SRC);
    let verts_b = empty("verts", u64::from(tris) * 36, sto | wgpu::BufferUsages::BLAS_INPUT);
    let bench_b = empty("bench", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    let set_bench = |count: u32, seed: u32, kind: u32| {
        let b: Vec<u8> = [count, seed, kind, 0].iter().flat_map(|x| x.to_le_bytes()).collect();
        gpu.queue.write_buffer(&bench_b, 0, &b);
    };

    let g0 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[uniform(0), storage(1, true), storage(2, true)] });
    let g1 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[storage(0, false), storage(1, false), storage(2, false), uniform(3)] });
    let gh = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::AccelerationStructure { vertex_return: false }, count: None },
            storage(1, false),
            storage(2, false),
            uniform(3),
        ],
    });
    let b0 = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &g0, entries: &entries(&[&params_b, &geometry_b, &nodes_b]) });
    let b1 = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &g1, entries: &entries(&[&rays_b, &hits_b, &verts_b, &bench_b]) });

    let compile = Instant::now();
    let sw_module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sw"), source: wgpu::ShaderSource::Wgsl(format!("{path_wgsl}\n{SW}\n{TRACE_SW}\n{PATHS}").into()) });
    let sw_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&g0), Some(&g1)], immediate_size: 0 });
    let pipe = |module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, entry: &str| {
        d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(layout), module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
    };
    let export = pipe(&sw_module, &sw_layout, "export_vertices");
    let primary = pipe(&sw_module, &sw_layout, "primary_rays");
    let secondary = pipe(&sw_module, &sw_layout, "secondary_rays");
    let fast = pipe(&sw_module, &sw_layout, "sw_trace");
    let precise = pipe(&sw_module, &sw_layout, "sw_precise");
    let io = pipe(&sw_module, &sw_layout, "io_only");
    let hw_module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("hw"), source: wgpu::ShaderSource::Wgsl(HW.into()) });
    let hw_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&gh)], immediate_size: 0 });
    let hw = pipe(&hw_module, &hw_layout, "hw_trace");
    let ga = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::AccelerationStructure { vertex_return: false }, count: None }],
    });
    let hw_paths_module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hw paths"),
        source: wgpu::ShaderSource::Wgsl(format!("enable wgpu_ray_query;\n{path_wgsl}\n{SW}\n{TRACE_HW}\n{PATHS}").into()),
    });
    let hw_paths_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&g0), Some(&g1), Some(&ga)], immediate_size: 0 });
    let sw_paths = pipe(&sw_module, &sw_layout, "paths");
    let hw_paths = pipe(&hw_paths_module, &hw_paths_layout, "paths");
    eprintln!("pipelines created in {:.1} s", compile.elapsed().as_secs_f64());

    // Triangles exactly as the renderer reads them (triangle() in path.wgsl), then the BLAS/TLAS.
    set_bench(tris, 0, 0);
    gpu.dispatch(&export, &[&b0, &b1], tris.div_ceil(64));
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: tris * 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = d.create_blas(
        &wgpu::CreateBlasDescriptor { label: Some("scene"), flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
    );
    let mut tlas = d.create_tlas(&wgpu::CreateTlasDescriptor { label: Some("scene"), max_instances: 1, flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build });
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0, 0xff));
    let build = Instant::now();
    let mut e = d.create_command_encoder(&Default::default());
    e.build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &verts_b,
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
    gpu.queue.submit([e.finish()]);
    gpu.wait();
    let build_ms = build.elapsed().as_secs_f64() * 1000.0;
    eprintln!("BLAS+TLAS build {build_ms:.1} ms (wall, includes submit)");
    let bh = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &gh,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tlas.as_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: rays_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: hits_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bench_b.as_entire_binding() },
        ],
    });

    let ba = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &ga, entries: &[wgpu::BindGroupEntry { binding: 0, resource: tlas.as_binding() }] });
    let groups = rays_n.div_ceil(64);
    println!(
        "{{\"adapter\":\"{}\",\"driver\":\"{} {}\",\"width\":{width},\"height\":{height},\"triangles\":{tris},\"bvh_nodes\":{},\"hw_build_ms\":{build_ms:.2},\"iterations\":{iterations}}}",
        info.name,
        info.driver,
        info.driver_info,
        nodes.len() / 32
    );
    for (name, kind) in [("primary", None), ("diffuse_bounce", Some(0u32)), ("shadow", Some(1u32))] {
        // Rays: primary from the camera; secondary from the renderer's own (precise) primary hits.
        set_bench(rays_n, 1948, 0);
        gpu.dispatch(&primary, &[&b0, &b1], groups);
        if let Some(k) = kind {
            gpu.dispatch(&precise, &[&b0, &b1], groups);
            set_bench(rays_n, 1948, k);
            gpu.dispatch(&secondary, &[&b0, &b1], groups);
        }
        let rays = words(&gpu.read(&rays_b));
        let active: Vec<bool> = rays.chunks_exact((RAY_BYTES / 4) as usize).map(|r| f32::from_bits(r[7]) >= 0.0).collect();
        let n_active = active.iter().filter(|a| **a).count();
        let mut runs = Vec::new();
        for (label, pipeline, binds) in
            [("software_precise", &precise, vec![&b0, &b1]), ("software_fast", &fast, vec![&b0, &b1]), ("hardware", &hw, vec![&bh]), ("io_floor", &io, vec![&b0, &b1])]
        {
            gpu.dispatch(pipeline, &binds, groups); // warm-up
            let (mut g, mut w) = (Vec::new(), Vec::new());
            for _ in 0..iterations {
                let (a, b) = gpu.dispatch(pipeline, &binds, groups);
                g.push(a);
                w.push(b);
            }
            let hits = words(&gpu.read(&hits_b));
            runs.push(Run { label, gpu_ms: median(g), wall_ms: median(w), hits });
        }
        let (sw_hits, hw_hits) = (&runs[0].hits, &runs[2].hits);
        let (mut same_hit, mut same_tri, mut sw_found, mut hw_found) = (0usize, 0usize, 0usize, 0usize);
        for (i, a) in active.iter().enumerate() {
            if !*a {
                continue;
            }
            let (st, ht) = (sw_hits[i * 4 + 1], hw_hits[i * 4 + 1]);
            sw_found += usize::from(st != NONE);
            hw_found += usize::from(ht != NONE);
            same_hit += usize::from((st == NONE) == (ht == NONE));
            same_tri += usize::from(st == ht);
        }
        for r in &runs {
            println!(
                "{{\"set\":\"{name}\",\"tracer\":\"{}\",\"active_rays\":{n_active},\"gpu_ms\":{:.3},\"wall_ms\":{:.3},\"mrays_per_s\":{:.1}}}",
                r.label,
                r.gpu_ms,
                r.wall_ms,
                n_active as f64 / r.ms() / 1000.0
            );
        }
        let floor = runs[3].ms();
        println!(
            "{{\"set\":\"{name}\",\"agreement_precise_vs_hardware\":{{\"hit_or_miss_same\":{:.6},\"same_triangle\":{:.6},\"software_hits\":{sw_found},\"hardware_hits\":{hw_found}}},\"speedup_vs_precise\":{:.1},\"speedup_vs_fast\":{:.1},\"speedup_vs_precise_above_io_floor\":{:.1}}}",
            same_hit as f64 / n_active as f64,
            same_tri as f64 / n_active as f64,
            runs[0].ms() / runs[2].ms(),
            runs[1].ms() / runs[2].ms(),
            (runs[0].ms() - floor) / (runs[2].ms() - floor).max(1e-3)
        );
    }
    // Whole paths, traversal only: no ray buffers, so this is the renderer's access pattern.
    set_bench(rays_n, 1948, vertices);
    let mut runs = Vec::new();
    for (label, pipeline, binds) in [("software_precise", &sw_paths, vec![&b0, &b1]), ("hardware", &hw_paths, vec![&b0, &b1, &ba])] {
        gpu.dispatch(pipeline, &binds, groups);
        let (mut g, mut w) = (Vec::new(), Vec::new());
        for _ in 0..iterations {
            let (a, b) = gpu.dispatch(pipeline, &binds, groups);
            g.push(a);
            w.push(b);
        }
        let hits = words(&gpu.read(&hits_b));
        runs.push(Run { label, gpu_ms: median(g), wall_ms: median(w), hits });
    }
    for r in &runs {
        let sum = |k: usize| r.hits.chunks_exact(4).map(|h| u64::from(h[k])).sum::<u64>();
        let traced = sum(0);
        println!(
            "{{\"set\":\"paths_{vertices}_vertices\",\"tracer\":\"{}\",\"rays_traced\":{traced},\"surface_vertices\":{},\"shadowed\":{},\"gpu_ms\":{:.3},\"wall_ms\":{:.3},\"mrays_per_s\":{:.1}}}",
            r.label,
            sum(1),
            sum(2),
            r.gpu_ms,
            r.wall_ms,
            traced as f64 / r.ms() / 1000.0
        );
    }
    println!("{{\"set\":\"paths_{vertices}_vertices\",\"speedup_per_path_sample\":{:.1}}}", runs[0].ms() / runs[1].ms());
}
