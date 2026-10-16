//! Simulation on the GPU. [`GpuAccelerator`] runs the lattice-Boltzmann airflow step
//! (`lbm.wgsl`, in-place AA-pattern streaming) and the temperature solve (`solve.wgsl`, a
//! 7-point stencil with host-driven conjugate gradients or BiCGSTAB) as wgpu compute shaders,
//! and implements [`solidcraft_sim::Accelerator`] so `sim.solve` can use it. Populations and
//! vectors are f32 on the device; sums are combined in f64 on the host.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

use std::sync::Mutex;

use solidcraft_sim::{Accelerator, FlowFields, LatticeSetup, Progress, SimError, ThermalSolution, ThermalSystem};
use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests;

const LBM_SHADER: &str = include_str!("lbm.wgsl");
const REDUCE_SHADER: &str = include_str!("reduce.wgsl");
const SOLVE_SHADER: &str = include_str!("solve.wgsl");

/// Lattice step kernel workgroup size (matches `lbm.wgsl`).
const LBM_WG: u32 = 128;
/// Reduction and solver kernels' workgroup size (matches `reduce.wgsl` and `solve.wgsl`).
const VEC_WG: u32 = 256;
/// Workgroups along x per dispatch; the rest go along y.
const MAX_GROUPS_X: u32 = 32768;
/// Lattice steps encoded per submission; also the ring of parameter buffers.
const BATCH: usize = 50;

#[derive(Debug, thiserror::Error)]
pub enum GpuSimError {
    #[error("no GPU adapter: {0}")]
    NoAdapter(String),
    #[error("could not open the GPU device: {0}")]
    Device(String),
    #[error("the GPU could not build the simulation pipelines: {0}")]
    Pipeline(String),
}

type Result<T> = std::result::Result<T, SimError>;

fn internal(what: impl std::fmt::Display) -> SimError {
    SimError::Internal(what.to_string())
}

/// Workgroup counts (x, y) covering `n` items with workgroups of `wg`.
fn groups(n: usize, wg: u32) -> Result<(u32, u32)> {
    let total = u32::try_from(n.div_ceil(wg as usize).max(1)).map_err(|_| SimError::TooLarge("too many cells for one dispatch".into()))?;
    let gx = total.min(MAX_GROUPS_X);
    let gy = total.div_ceil(gx);
    if gy > 65535 {
        return Err(SimError::TooLarge("too many cells for one dispatch".into()));
    }
    Ok((gx, gy))
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The stencil solver's kernels in one precision.
struct SolvePipes {
    layout: wgpu::BindGroupLayout,
    matvec: wgpu::ComputePipeline,
    combine: wgpu::ComputePipeline,
    precond: wgpu::ComputePipeline,
    dot: wgpu::ComputePipeline,
    /// Bytes per element: 4 (f32) or 8 (f64).
    elem: u64,
}

struct Pipelines {
    lbm_layout: wgpu::BindGroupLayout,
    lbm_step: wgpu::ComputePipeline,
    lbm_init: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    solve32: SolvePipes,
    /// The same kernels in f64, on devices with `SHADER_F64`.
    solve64: Option<SolvePipes>,
}

/// The simulation backend on one GPU device.
pub struct GpuAccelerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    software: bool,
    max_buffer: u64,
    max_binding: u64,
    pipelines: Pipelines,
    /// Solves run one at a time on a device.
    lock: Mutex<()>,
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
const UNIFORM: wgpu::BufferBindingType = wgpu::BufferBindingType::Uniform;
const READ: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };

impl GpuAccelerator {
    /// The backend on an existing device (`adapter`: its name for reports; `software`: the
    /// adapter is a GPU driver running on the CPU).
    pub fn new(adapter: impl Into<String>, software: bool, device: wgpu::Device, queue: wgpu::Queue) -> std::result::Result<Self, GpuSimError> {
        let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation].map(|f| device.push_error_scope(f));
        let f64_ok = device.features().contains(wgpu::Features::SHADER_F64);
        let pipelines = Self::build(&device, f64_ok);
        let mut failed = None;
        for scope in scopes.into_iter().rev() {
            if let Some(e) = popped(scope.pop()) {
                failed.get_or_insert(e.to_string());
            }
        }
        if let Some(e) = failed {
            return Err(GpuSimError::Pipeline(e));
        }
        let limits = device.limits();
        Ok(GpuAccelerator {
            device,
            queue,
            adapter: adapter.into(),
            software,
            max_buffer: limits.max_buffer_size,
            max_binding: limits.max_storage_buffer_binding_size,
            pipelines,
            lock: Mutex::new(()),
        })
    }

    /// A device of its own, without a window (native only): the adapter the `SOLIDCRAFT_GPU`
    /// policy picks, opened with the largest buffers it allows.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn headless() -> std::result::Result<Self, GpuSimError> {
        use solidcraft_gpu::select::{Policy, Tier, pick};
        let policy = Policy::from_env().map_err(GpuSimError::NoAdapter)?;
        if policy == Policy::Cpu {
            return Err(GpuSimError::NoAdapter("SOLIDCRAFT_GPU=cpu: simulation requires a GPU".into()));
        }
        let (adapter, report) = pick(policy).ok_or_else(|| GpuSimError::NoAdapter(format!("no adapter under the {} policy", policy.name())))?;
        let have = adapter.limits();
        // Doubles for the temperature solve where the GPU has them (not Apple's).
        let required_features = adapter.features() & wgpu::Features::SHADER_F64;
        let mut limits = wgpu::Limits::default().using_resolution(have.clone());
        limits.max_buffer_size = have.max_buffer_size;
        limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage = have.max_storage_buffers_per_shader_stage.max(limits.max_storage_buffers_per_shader_stage);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("solidcraft simulation"),
            required_features,
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| GpuSimError::Device(e.to_string()))?;
        let software = report.tier == Tier::Software;
        Self::new(report.name, software, device, queue)
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter
    }

    /// The adapter is a GPU driver running on the CPU (lavapipe, llvmpipe, WARP): correct, slow.
    pub fn is_software(&self) -> bool {
        self.software
    }

    /// The temperature solve runs in f64 on this device.
    pub fn has_f64(&self) -> bool {
        self.pipelines.solve64.is_some()
    }

    /// Forget the f64 kernels (tests of the f32 path).
    #[doc(hidden)]
    pub fn without_f64(mut self) -> Self {
        self.pipelines.solve64 = None;
        self
    }

    fn build(device: &wgpu::Device, f64_ok: bool) -> Pipelines {
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
        };
        let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        let pipeline = |label: &str, layout: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, entry_point: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let lbm = module("solidcraft sim lbm", LBM_SHADER);
        let lbm_layout = layout(
            "solidcraft sim lbm",
            &[entry(0, UNIFORM), entry(1, READ), entry(2, WRITE), entry(3, WRITE), entry(4, READ), entry(5, READ), entry(6, READ)],
        );
        let reduce_mod = module("solidcraft sim reduce", REDUCE_SHADER);
        let reduce_layout = layout("solidcraft sim reduce", &[entry(0, UNIFORM), entry(1, READ), entry(2, READ), entry(3, READ), entry(4, WRITE)]);
        let solve_pipes = |source: &str, elem: u64| {
            let solve = module("solidcraft sim solve", source);
            let layout = layout("solidcraft sim solve", &[entry(0, UNIFORM), entry(1, READ), entry(2, READ), entry(3, READ), entry(4, WRITE)]);
            SolvePipes {
                matvec: pipeline("solidcraft sim matvec", &layout, &solve, "matvec"),
                combine: pipeline("solidcraft sim combine", &layout, &solve, "combine"),
                precond: pipeline("solidcraft sim precond", &layout, &solve, "precond"),
                dot: pipeline("solidcraft sim dot", &layout, &solve, "dot_partial"),
                layout,
                elem,
            }
        };
        let solve32 = solve_pipes(SOLVE_SHADER, 4);
        let solve64 = f64_ok.then(|| solve_pipes(&SOLVE_SHADER.replace("f32", "f64"), 8));
        Pipelines {
            lbm_step: pipeline("solidcraft sim lbm step", &lbm_layout, &lbm, "step"),
            lbm_init: pipeline("solidcraft sim lbm init", &lbm_layout, &lbm, "init"),
            lbm_layout,
            reduce: pipeline("solidcraft sim reduce", &reduce_layout, &reduce_mod, "reduce"),
            reduce_layout,
            solve32,
            solve64,
        }
    }

    fn check_size(&self, bytes: u64, what: &str) -> Result<()> {
        if bytes > self.max_buffer || bytes > self.max_binding {
            return Err(SimError::TooLarge(format!(
                "{what} needs {} MB; this GPU allows {} MB per buffer: use a coarser cell size or a smaller domain",
                bytes / (1 << 20),
                self.max_buffer.min(self.max_binding) / (1 << 20)
            )));
        }
        Ok(())
    }

    fn storage(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let contents = if bytes.is_empty() { &[0u8; 16][..] } else { bytes };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn empty(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Copy `size` bytes of `src` back to the host as f32 (blocks; native only).
    fn read_f32(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<f32>> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (src, size);
            Err(SimError::Internal("GPU simulation is not supported in the browser yet".into()))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let d = &self.device;
            let read = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solidcraft sim read"),
                size: size.max(16),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("solidcraft sim read") });
            enc.copy_buffer_to_buffer(src, 0, &read, 0, size);
            self.queue.submit([enc.finish()]);
            let slice = read.slice(..size.max(16));
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            d.poll(wgpu::PollType::wait_indefinitely()).map_err(internal)?;
            rx.recv().map_err(internal)?.map_err(internal)?;
            let bytes = slice.get_mapped_range().map_err(internal)?;
            let values = bytes.as_chunks::<4>().0.iter().take((size / 4) as usize).map(|b| f32::from_le_bytes(*b)).collect();
            drop(bytes);
            read.unmap();
            Ok(values)
        }
    }

    /// Copy `size` bytes of `src` back as f64 (blocks; native only).
    fn read_f64(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<f64>> {
        let words = self.read_f32(src, size)?;
        Ok(words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| f64::from_le_bytes([w[0].to_le_bytes(), w[1].to_le_bytes()].concat().try_into().unwrap_or([0; 8])))
            .collect())
    }

    /// Copy `size` bytes of `src` back as either precision, widened to f64.
    fn read_elems(&self, src: &wgpu::Buffer, pipes: &SolvePipes, count: usize) -> Result<Vec<f64>> {
        if pipes.elem == 8 {
            self.read_f64(src, 8 * count as u64)
        } else {
            Ok(self.read_f32(src, 4 * count as u64)?.into_iter().map(f64::from).collect())
        }
    }

    /// Everything the device raised while `f` ran, as an error.
    fn guarded<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let scopes =
            [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation].map(|k| self.device.push_error_scope(k));
        let r = f();
        let mut failed = None;
        for scope in scopes.into_iter().rev() {
            if let Some(e) = popped(scope.pop()) {
                failed.get_or_insert(e.to_string());
            }
        }
        match failed {
            Some(e) => Err(SimError::Internal(format!("GPU error: {e}"))),
            None => r,
        }
    }
}

/// The error a popped error scope caught (native devices answer at once).
fn popped(f: impl std::future::Future<Output = Option<wgpu::Error>>) -> Option<wgpu::Error> {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(e) => e,
        std::task::Poll::Pending => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Airflow

#[repr(C)]
#[derive(Clone, Copy)]
struct LbmParams {
    nx: u32,
    ny: u32,
    nz: u32,
    n: u32,
    tau0: f32,
    cs2: f32,
    lambda: f32,
    g_lat: f32,
    step: u32,
    open_walls: u32,
    ramp_steps: u32,
    groups_x: u32,
}

impl LbmParams {
    fn bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(48);
        for x in [self.nx, self.ny, self.nz, self.n] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in [self.tau0, self.cs2, self.lambda, self.g_lat] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in [self.step, self.open_walls, self.ramp_steps, self.groups_x] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }
}

impl GpuAccelerator {
    fn flow_inner(&self, setup: &LatticeSetup, progress: Progress) -> Result<FlowFields> {
        setup.validate()?;
        let n = setup.len();
        let n32 = u32::try_from(n).map_err(|_| SimError::TooLarge("too many cells for the GPU".into()))?;
        // The kernels index populations as a * n + cell in 32 bits.
        if (19 * n as u64) > u64::from(u32::MAX) {
            return Err(SimError::TooLarge(format!(
                "{n} cells: the GPU lattice indexes 19 populations per cell in 32 bits (at most {} cells); use a coarser cell size",
                u32::MAX / 19
            )));
        }
        self.check_size(19 * 4 * n as u64, "the lattice")?;
        let (gx, gy) = groups(n, LBM_WG)?;
        let (rgx, rgy) = groups(n, VEC_WG)?;
        let reduce_groups = (rgx as usize) * (rgy as usize);
        let flags = self.storage("sim flags", &bytes_u32(&setup.flags));
        let fans: Vec<f32> = if setup.fans.is_empty() { vec![0.0; 4] } else { setup.fans.iter().flat_map(|f| [f[0], f[1], f[2], 0.0]).collect() };
        let fans = self.storage("sim fans", &bytes_f32(&fans));
        let dtemp = match &setup.dtemp {
            Some(t) => self.storage("sim dtemp", &bytes_f32(t)),
            None => self.empty("sim dtemp", 4 * n as u64),
        };
        let sponge = match &setup.sponge {
            Some(t) => self.storage("sim sponge", &bytes_f32(t)),
            None => self.empty("sim sponge", 4 * n as u64),
        };
        let f = self.empty("sim populations", 19 * 4 * n as u64);
        let u = self.empty("sim velocity", 16 * n as u64);
        let u_prev = self.empty("sim velocity before", 16 * n as u64);
        let partials = self.empty("sim partials", 16 * reduce_groups as u64);
        let base = LbmParams {
            nx: setup.nx as u32,
            ny: setup.ny as u32,
            nz: setup.nz as u32,
            n: n32,
            tau0: setup.tau0,
            cs2: setup.smagorinsky2,
            lambda: setup.lambda,
            g_lat: if setup.dtemp.is_some() { setup.g_lat } else { 0.0 },
            step: 0,
            open_walls: setup.open_walls,
            ramp_steps: setup.ramp_steps,
            groups_x: gx,
        };
        // A ring of parameter buffers and bind groups: a batch of steps goes in one submission.
        let ring: Vec<(wgpu::Buffer, wgpu::BindGroup)> = (0..BATCH)
            .map(|_| {
                let params = self.uniform("sim lbm params", &base.bytes());
                let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sim lbm"),
                    layout: &self.pipelines.lbm_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: flags.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: f.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: fans.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: dtemp.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: sponge.as_entire_binding() },
                    ],
                });
                (params, bind)
            })
            .collect();
        let reduce_params = self.uniform("sim reduce params", &bytes_u32(&[n32, rgx, 0, 0]));
        let reduce_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim reduce"),
            layout: &self.pipelines.reduce_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: reduce_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_prev.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: partials.as_entire_binding() },
            ],
        });
        // init
        {
            let Some((_, bind)) = ring.first() else { return Err(internal("no parameter ring")) };
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim lbm init") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim lbm init"), timestamp_writes: None });
                pass.set_pipeline(&self.pipelines.lbm_init);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            enc.copy_buffer_to_buffer(&u, 0, &u_prev, 0, 16 * n as u64);
            self.queue.submit([enc.finish()]);
        }
        let mut steps = 0usize;
        let mut converged = false;
        let mut change = f64::INFINITY;
        let mut max_speed = 0.0f64;
        let mut step = 0usize;
        while step < setup.max_steps {
            let batch = BATCH.min(setup.max_steps - step);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim lbm batch") });
            for k in 0..batch {
                let Some((params, bind)) = ring.get(k) else { return Err(internal("parameter ring too short")) };
                let p = LbmParams { step: (step + k) as u32, ..base };
                self.queue.write_buffer(params, 0, &p.bytes());
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim lbm step"), timestamp_writes: None });
                pass.set_pipeline(&self.pipelines.lbm_step);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim reduce"), timestamp_writes: None });
                pass.set_pipeline(&self.pipelines.reduce);
                pass.set_bind_group(0, &reduce_bind, &[]);
                pass.dispatch_workgroups(rgx, rgy, 1);
            }
            enc.copy_buffer_to_buffer(&u, 0, &u_prev, 0, 16 * n as u64);
            self.queue.submit([enc.finish()]);
            step += batch;
            steps = step;
            let sums = self.read_f32(&partials, 16 * reduce_groups as u64)?;
            let (mut num, mut den, mut fastest) = (0.0f64, 0.0f64, 0.0f64);
            for p in sums.as_chunks::<4>().0 {
                num += f64::from(p[0]);
                den += f64::from(p[1]);
                fastest = fastest.max(f64::from(p[2]));
            }
            max_speed = fastest.sqrt();
            if !max_speed.is_finite() || !num.is_finite() || !den.is_finite() || max_speed > 0.5 {
                return Err(SimError::NotConverged(
                    "the airflow solve blew up; try a coarser grid, a lower fan speed or a smaller lattice velocity".into(),
                ));
            }
            change = if den > 0.0 {
                (num / den).sqrt()
            } else if num > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            if !progress("flow", step, setup.max_steps) {
                return Err(SimError::Cancelled);
            }
            if step >= 400 && change < setup.tolerance {
                converged = true;
                break;
            }
        }
        let out = self.read_f32(&u, 16 * n as u64)?;
        let mut vel = Vec::with_capacity(n);
        let mut rho = Vec::with_capacity(n);
        for v in out.as_chunks::<4>().0 {
            vel.push([v[0], v[1], v[2]]);
            rho.push(v[3]);
        }
        if vel.len() != n {
            return Err(internal("velocity readback is short"));
        }
        Ok(FlowFields { u: vel, rho, steps, converged, change, max_lattice_speed: max_speed })
    }
}

// ---------------------------------------------------------------------------------------------
// Temperature

/// The vectors of a solve on the device and the kernels over them, in one precision.
struct Solver<'a> {
    gpu: &'a GpuAccelerator,
    pipes: &'a SolvePipes,
    n: usize,
    gx: u32,
    gy: u32,
    groups: usize,
    diag: wgpu::Buffer,
    off: wgpu::Buffer,
    invdiag: wgpu::Buffer,
    partials: wgpu::Buffer,
    params: wgpu::Buffer,
}

/// Vector bytes in the solver's precision.
fn elems(values: &[f64], elem: u64) -> Vec<u8> {
    if elem == 8 { values.iter().flat_map(|v| v.to_le_bytes()).collect() } else { values.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect() }
}

impl Solver<'_> {
    fn params(&self, a: f64, b: f64, c: f64) -> Vec<u8> {
        let mut v = Vec::with_capacity(48);
        for x in [0u32, 0, self.n as u32, self.gx] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&elems(&[a, b, c, 0.0], self.pipes.elem));
        v
    }

    fn with_dims(&self, nx: u32, ny: u32, a: f64, b: f64, c: f64) -> Vec<u8> {
        let mut v = self.params(a, b, c);
        v.splice(0..8, nx.to_le_bytes().into_iter().chain(ny.to_le_bytes()));
        v
    }

    fn run(&self, pipeline: &wgpu::ComputePipeline, params: &[u8], diag: &wgpu::Buffer, off: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer) {
        let gpu = self.gpu;
        gpu.queue.write_buffer(&self.params, 0, params);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim solve"),
            layout: &self.pipes.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: diag.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: off.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: y.as_entire_binding() },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim solve") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim solve"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(self.gx, self.gy, 1);
        }
        gpu.queue.submit([enc.finish()]);
    }

    /// y = A x
    fn matvec(&self, nx: u32, ny: u32, x: &wgpu::Buffer, y: &wgpu::Buffer) {
        self.run(&self.pipes.matvec, &self.with_dims(nx, ny, 0.0, 0.0, 0.0), &self.diag, &self.off, x, y);
    }
    /// y = a*y + b*x + c*z
    fn combine(&self, y: &wgpu::Buffer, a: f64, x: &wgpu::Buffer, b: f64, z: &wgpu::Buffer, c: f64) {
        self.run(&self.pipes.combine, &self.params(a, b, c), z, &self.off, x, y);
    }
    /// y = x / diag
    fn precond(&self, x: &wgpu::Buffer, y: &wgpu::Buffer) {
        self.run(&self.pipes.precond, &self.params(0.0, 0.0, 0.0), &self.invdiag, &self.off, x, y);
    }
    /// x · y
    fn dot(&self, x: &wgpu::Buffer, y: &wgpu::Buffer) -> Result<f64> {
        self.run(&self.pipes.dot, &self.params(0.0, 0.0, 0.0), y, &self.off, x, &self.partials);
        Ok(self.gpu.read_elems(&self.partials, self.pipes, self.groups)?.iter().sum())
    }
}

impl GpuAccelerator {
    fn thermal_inner(&self, system: &ThermalSystem, progress: Progress) -> Result<ThermalSolution> {
        let st = &system.stencil;
        let n = st.len();
        if n == 0 || system.rhs.len() != n || st.off.len() != n {
            return Err(SimError::Invalid("thermal system does not match its stencil".into()));
        }
        let (gx, gy) = groups(n, VEC_WG)?;
        let groups = (gx as usize) * (gy as usize);
        let bnorm_true = system.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (nx, ny) = (st.nx as u32, st.ny as u32);
        let method = if system.symmetric { "cg" } else { "bicgstab" };
        let mut report =
            ThermalSolution { x: vec![0.0; n], iterations: 0, residual: f64::INFINITY, converged: false, method: format!("gpu-{method}") };
        if bnorm_true == 0.0 {
            report.residual = 0.0;
            report.converged = true;
            return Ok(report);
        }
        let plane = st.nx * st.ny;
        let neighbours = |idx: usize| [idx.wrapping_sub(1), idx + 1, idx.wrapping_sub(st.nx), idx + st.nx, idx.wrapping_sub(plane), idx + plane];

        if let Some(pipes) = &self.pipelines.solve64 {
            // Doubles on the device: the CPU algorithm as it is, one pass.
            self.check_size(6 * 8 * n as u64, "the thermal stencil")?;
            let diag: Vec<f64> = st.diag.clone();
            let invdiag: Vec<f64> = st.diag.iter().map(|d| if d.abs() > 0.0 { 1.0 / d } else { 0.0 }).collect();
            let mut off = vec![0.0f64; 6 * n];
            for (idx, o) in st.off.iter().enumerate() {
                for (side, v) in o.iter().enumerate() {
                    if let Some(slot) = off.get_mut(side * n + idx) {
                        *slot = *v;
                    }
                }
            }
            let sv = Solver {
                gpu: self,
                pipes,
                n,
                gx,
                gy,
                groups,
                diag: self.storage("sim diag", &elems(&diag, 8)),
                off: self.storage("sim off", &elems(&off, 8)),
                invdiag: self.storage("sim invdiag", &elems(&invdiag, 8)),
                partials: self.empty("sim dot partials", 8 * groups as u64),
                params: self.uniform("sim solve params", &[0u8; 48]),
            };
            let bytes = 8 * n as u64;
            let scaled: Vec<f64> = system.rhs.iter().map(|v| v / bnorm_true).collect();
            let b = self.storage("sim rhs", &elems(&scaled, 8));
            let x = self.empty("sim x", bytes);
            let r = self.empty("sim r", bytes);
            let inner = self.solve_pass(&sv, system.symmetric, nx, ny, &b, &x, &r, system.tolerance, system.max_iterations, progress)?;
            report.x = self.read_elems(&x, pipes, n)?.into_iter().map(|v| v * bnorm_true).collect();
            if report.x.len() != n {
                return Err(internal("temperature readback is short"));
            }
            report.iterations = inner.iterations;
            report.residual = inner.residual;
            report.converged = inner.converged;
            report.method = format!("gpu-{method}64");
            if !report.x.iter().all(|v| v.is_finite()) {
                return Err(SimError::NotConverged("the temperature solve produced non-finite values".into()));
            }
            return Ok(report);
        }

        // Single precision on the device. Symmetric diagonal scaling, S A S with
        // S = diag(A)^-1/2, before the system meets f32: a metal cell's row holds conductances
        // of hundreds of W/K beside a film coupling of microwatts per kelvin, which plain f32
        // would round away; scaled, every diagonal is 1 and the couplings are ratios (the Jacobi
        // preconditioner is then the identity). Then mixed-precision refinement: the device
        // solves for a correction to the f64 solution held here, and the true residual `b - A x`
        // is formed in f64 for the next pass.
        let pipes = &self.pipelines.solve32;
        self.check_size(6 * 4 * n as u64, "the thermal stencil")?;
        let scale: Vec<f64> = st.diag.iter().map(|d| if d.abs() > 0.0 { 1.0 / d.abs().sqrt() } else { 1.0 }).collect();
        let diag: Vec<f64> = st.diag.iter().zip(&scale).map(|(d, s)| d * s * s).collect();
        let mut off = vec![0.0f64; 6 * n];
        for (idx, o) in st.off.iter().enumerate() {
            let nb = neighbours(idx);
            let si = scale.get(idx).copied().unwrap_or(0.0);
            for (side, v) in o.iter().enumerate() {
                let sj = nb.get(side).and_then(|j| scale.get(*j)).copied().unwrap_or(0.0);
                if let Some(slot) = off.get_mut(side * n + idx) {
                    *slot = v * si * sj;
                }
            }
        }
        let sv = Solver {
            gpu: self,
            pipes,
            n,
            gx,
            gy,
            groups,
            diag: self.storage("sim diag", &elems(&diag, 4)),
            off: self.storage("sim off", &elems(&off, 4)),
            invdiag: self.storage("sim invdiag", &elems(&vec![1.0; n], 4)),
            partials: self.empty("sim dot partials", 4 * groups as u64),
            params: self.uniform("sim solve params", &[0u8; 48]),
        };
        let bytes = 4 * n as u64;
        let b = self.empty("sim rhs", bytes);
        let x = self.empty("sim x", bytes);
        let r = self.empty("sim r", bytes);
        let mut residual = system.rhs.clone();
        let mut ax = vec![0.0f64; n];
        let mut used = 0usize;
        let mut inner_converged = false;
        for pass in 0..4 {
            let rnorm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
            report.residual = rnorm / bnorm_true;
            if report.residual < system.tolerance || (pass > 0 && report.residual > 0.5) {
                // converged, or f32 cannot represent this system (the correction did not help)
                break;
            }
            let norm = rnorm.max(f64::MIN_POSITIVE);
            let rhs: Vec<f64> = residual.iter().zip(&scale).map(|(v, s)| v * s / norm).collect();
            self.queue.write_buffer(&b, 0, &elems(&rhs, 4));
            // f32 on the device: a tighter target than this cannot be told from rounding.
            let inner =
                self.solve_pass(&sv, system.symmetric, nx, ny, &b, &x, &r, 1e-6, system.max_iterations.saturating_sub(used).max(50), progress)?;
            used += inner.iterations;
            inner_converged = inner.converged;
            let d = self.read_elems(&x, pipes, n)?;
            if d.len() != n {
                return Err(internal("temperature readback is short"));
            }
            for ((xi, di), s) in report.x.iter_mut().zip(&d).zip(&scale) {
                *xi += di * s * norm;
            }
            st.apply(&report.x, &mut ax);
            for ((ri, bi), ai) in residual.iter_mut().zip(&system.rhs).zip(&ax) {
                *ri = bi - ai;
            }
            report.iterations = used;
            if !inner.converged {
                break;
            }
        }
        let rnorm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        report.residual = rnorm / bnorm_true;
        // The f64 residual is the verdict; a pass that converged in f32 counts when it got close.
        report.converged = report.residual < system.tolerance || (inner_converged && report.residual < 1e-5);
        if !report.converged || !report.x.iter().all(|v| v.is_finite()) {
            // f32 could not carry this system: the f64 reference finishes it (and says so), so
            // the study still gets its temperatures. The airflow stays on the GPU.
            let mut cpu = solidcraft_sim::CpuAccelerator.thermal(system, progress)?;
            cpu.method = format!("cpu-{} (GPU f32 insufficient)", cpu.method);
            cpu.iterations += used;
            return Ok(cpu);
        }
        Ok(report)
    }

    /// One solve of `A x = b` on the device from `x = 0`, in the solver's precision (the
    /// right-hand side is pre-scaled to unit norm, so `tol` is a relative residual).
    fn solve_pass(
        &self,
        sv: &Solver<'_>,
        symmetric: bool,
        nx: u32,
        ny: u32,
        b: &wgpu::Buffer,
        x: &wgpu::Buffer,
        r: &wgpu::Buffer,
        tol: f64,
        max_iter: usize,
        progress: Progress,
    ) -> Result<ThermalSolution> {
        let bytes = sv.pipes.elem * sv.n as u64;
        let mut report = ThermalSolution { x: Vec::new(), iterations: 0, residual: f64::INFINITY, converged: false, method: String::new() };
        let mut cancelled = false;
        let mut check = |it: usize, cancelled: &mut bool| -> bool {
            if it.is_multiple_of(25) && !progress("thermal", it, max_iter) {
                *cancelled = true;
                return false;
            }
            true
        };
        // x = 0, r = b
        sv.combine(x, 0.0, b, 0.0, b, 0.0);
        sv.combine(r, 0.0, b, 1.0, b, 0.0);
        if symmetric {
            let z = self.empty("sim z", bytes);
            let p = self.empty("sim p", bytes);
            let q = self.empty("sim q", bytes);
            sv.precond(r, &z);
            sv.combine(&p, 0.0, &z, 1.0, &z, 0.0);
            let mut rz = sv.dot(r, &z)?;
            for it in 1..=max_iter {
                sv.matvec(nx, ny, &p, &q);
                let pq = sv.dot(&p, &q)?;
                if !(pq.abs() > 0.0) || !pq.is_finite() {
                    break;
                }
                let alpha = rz / pq;
                sv.combine(x, 1.0, &p, alpha, &p, 0.0);
                sv.combine(r, 1.0, &q, -alpha, &q, 0.0);
                let res = sv.dot(r, r)?.sqrt();
                report.iterations = it;
                report.residual = res;
                if !res.is_finite() {
                    break;
                }
                if res < tol {
                    report.converged = true;
                    break;
                }
                if !check(it, &mut cancelled) {
                    break;
                }
                sv.precond(r, &z);
                let rz_new = sv.dot(r, &z)?;
                let beta = if rz.abs() > 0.0 { rz_new / rz } else { 0.0 };
                rz = rz_new;
                sv.combine(&p, beta, &z, 1.0, &z, 0.0);
            }
        } else {
            let r0 = self.empty("sim r0", bytes);
            let p = self.empty("sim p", bytes);
            let v = self.empty("sim v", bytes);
            let s = self.empty("sim s", bytes);
            let t = self.empty("sim t", bytes);
            let ph = self.empty("sim ph", bytes);
            let sh = self.empty("sim sh", bytes);
            sv.combine(&r0, 0.0, r, 1.0, r, 0.0);
            let (mut rho, mut alpha, mut omega): (f64, f64, f64) = (1.0, 1.0, 1.0);
            for it in 1..=max_iter {
                let rho_new = sv.dot(&r0, r)?;
                if !(rho_new.abs() > 0.0) || !rho_new.is_finite() || !(omega.abs() > 0.0) {
                    break;
                }
                let beta = (rho_new / rho) * (alpha / omega);
                rho = rho_new;
                sv.combine(&p, beta, r, 1.0, &v, -beta * omega);
                sv.precond(&p, &ph);
                sv.matvec(nx, ny, &ph, &v);
                let r0v = sv.dot(&r0, &v)?;
                if !(r0v.abs() > 0.0) {
                    break;
                }
                alpha = rho / r0v;
                sv.combine(&s, 0.0, r, 1.0, &v, -alpha);
                let snorm = sv.dot(&s, &s)?.sqrt();
                if snorm < tol {
                    sv.combine(x, 1.0, &ph, alpha, &ph, 0.0);
                    report.iterations = it;
                    report.residual = snorm;
                    report.converged = true;
                    break;
                }
                sv.precond(&s, &sh);
                sv.matvec(nx, ny, &sh, &t);
                let tt = sv.dot(&t, &t)?;
                if !(tt > 0.0) {
                    break;
                }
                omega = sv.dot(&t, &s)? / tt;
                sv.combine(x, 1.0, &ph, alpha, &sh, omega);
                sv.combine(r, 0.0, &s, 1.0, &t, -omega);
                let res = sv.dot(r, r)?.sqrt();
                report.iterations = it;
                report.residual = res;
                if !res.is_finite() {
                    break;
                }
                if res < tol {
                    report.converged = true;
                    break;
                }
                if !check(it, &mut cancelled) {
                    break;
                }
            }
        }
        if cancelled {
            return Err(SimError::Cancelled);
        }
        Ok(report)
    }
}

impl Accelerator for GpuAccelerator {
    fn name(&self) -> String {
        format!("gpu: {}{}", self.adapter, if self.software { " (software)" } else { "" })
    }

    fn flow(&self, setup: &LatticeSetup, progress: Progress) -> Result<FlowFields> {
        let _guard = self.lock.lock().map_err(|_| internal("the GPU backend is poisoned"))?;
        self.guarded(|| self.flow_inner(setup, progress))
    }

    fn thermal(&self, system: &ThermalSystem, progress: Progress) -> Result<ThermalSolution> {
        let _guard = self.lock.lock().map_err(|_| internal("the GPU backend is poisoned"))?;
        self.guarded(|| self.thermal_inner(system, progress))
    }
}
