//! Simulation on the GPU. [`GpuAccelerator`] runs the lattice-Boltzmann airflow step
//! (`lbm.wgsl`, in-place AA-pattern streaming) and the temperature solve (`solve.wgsl`, a
//! 7-point stencil with host-driven conjugate gradients or BiCGSTAB) as wgpu compute shaders,
//! and implements [`solidcraft_sim::Accelerator`] so `sim.solve` can use it. Populations and
//! vectors are f32 on the device; sums are combined in f64 on the host.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

use std::sync::Mutex;

use solidcraft_sim::{Accelerator, FlowFields, LatticeSetup, Multigrid, Progress, SimError, Stencil, ThermalSolution, ThermalSystem};
use wgpu::util::DeviceExt;

mod krylov;
mod multigrid;
#[cfg(test)]
mod tests;

const LBM_SHADER: &str = include_str!("lbm.wgsl");
const REDUCE_SHADER: &str = include_str!("reduce.wgsl");

/// Lattice step kernel workgroup size (matches `lbm.wgsl`).
const LBM_WG: u32 = 128;
/// Reduction and solver kernels' workgroup size (matches `reduce.wgsl` and `solve.wgsl`).
pub(crate) const VEC_WG: u32 = 256;
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
pub(crate) fn groups(n: usize, wg: u32) -> Result<(u32, u32)> {
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

struct Pipelines {
    lbm_layout: wgpu::BindGroupLayout,
    lbm_step: wgpu::ComputePipeline,
    lbm_init: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    solve32: krylov::KrylovPipes,
    mg32: multigrid::MultigridPipes,
    /// The same kernels in f64, on devices with `SHADER_F64`.
    solve64: Option<krylov::KrylovPipes>,
    mg64: Option<multigrid::MultigridPipes>,
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
        let solve32 = krylov::KrylovPipes::new(device, false);
        let solve64 = f64_ok.then(|| krylov::KrylovPipes::new(device, true));
        let mg32 = multigrid::MultigridPipes::new(device, false);
        let mg64 = f64_ok.then(|| multigrid::MultigridPipes::new(device, true));
        Pipelines {
            lbm_step: pipeline("solidcraft sim lbm step", &lbm_layout, &lbm, "step"),
            lbm_init: pipeline("solidcraft sim lbm init", &lbm_layout, &lbm, "init"),
            lbm_layout,
            reduce: pipeline("solidcraft sim reduce", &reduce_layout, &reduce_mod, "reduce"),
            reduce_layout,
            solve32,
            mg32,
            solve64,
            mg64,
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

impl GpuAccelerator {
    fn thermal_inner(&self, system: &ThermalSystem, progress: Progress) -> Result<ThermalSolution> {
        let st = &system.stencil;
        let n = st.len();
        if n == 0 || system.rhs.len() != n || st.off.len() != n {
            return Err(SimError::Invalid("thermal system does not match its stencil".into()));
        }
        let (gx, gy) = groups(n, VEC_WG)?;
        let bnorm_true = system.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let method = if system.symmetric { "cg" } else { "bicgstab" };
        let mut report =
            ThermalSolution { x: vec![0.0; n], iterations: 0, residual: f64::INFINITY, converged: false, method: format!("gpu-{method}") };
        if bnorm_true == 0.0 {
            report.residual = 0.0;
            report.converged = true;
            return Ok(report);
        }
        let side_major = |st: &Stencil| -> Vec<f64> {
            let mut off = vec![0.0f64; 6 * n];
            for (idx, o) in st.off.iter().enumerate() {
                for (side, v) in o.iter().enumerate() {
                    if let Some(slot) = off.get_mut(side * n + idx) {
                        *slot = *v;
                    }
                }
            }
            off
        };
        let read = |b: &wgpu::Buffer, size: u64| self.read_f32(b, size);
        let check = |bytes: u64, what: &str| self.check_size(bytes, what);
        let trace = std::env::var_os("SOLIDCRAFT_SIM_DEBUG").is_some();

        if let (Some(pipes), Some(mg_pipes)) = (&self.pipelines.solve64, &self.pipelines.mg64) {
            // Doubles on the device: the CPU algorithm as it is, one pass, resident on the GPU.
            self.check_size(8 * n as u64, "the thermal vectors")?;
            let mg = Multigrid::new(st);
            if trace {
                eprintln!("  multigrid: {} levels", mg.levels());
            }
            let gmg = multigrid::GpuMultigrid::new(&self.device, mg_pipes, &mg, &check)?;
            let k = krylov::Krylov::new(&self.device, &self.queue, pipes, &gmg, st.nx, st.ny, &st.diag, &side_major(st), gx, gy)?;
            let scaled: Vec<f64> = system.rhs.iter().map(|v| v / bnorm_true).collect();
            let solved = k.solve(&scaled, system.symmetric, system.tolerance, system.max_iterations, &read, progress)?;
            report.x = solved.x.into_iter().map(|v| v * bnorm_true).collect();
            report.iterations = solved.iterations;
            report.residual = solved.residual;
            report.converged = solved.converged;
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
        self.check_size(4 * n as u64, "the thermal vectors")?;
        let scale: Vec<f64> = st.diag.iter().map(|d| if d.abs() > 0.0 { 1.0 / d.abs().sqrt() } else { 1.0 }).collect();
        let plane = st.nx * st.ny;
        let scaled = Stencil {
            nx: st.nx,
            ny: st.ny,
            diag: st.diag.iter().zip(&scale).map(|(d, s)| d * s * s).collect(),
            off: st
                .off
                .iter()
                .enumerate()
                .map(|(idx, o)| {
                    let nb = [idx.wrapping_sub(1), idx + 1, idx.wrapping_sub(st.nx), idx + st.nx, idx.wrapping_sub(plane), idx + plane];
                    let si = scale.get(idx).copied().unwrap_or(1.0);
                    let mut row = [0.0; 6];
                    for (side, v) in o.iter().enumerate() {
                        let sj = nb.get(side).and_then(|j| scale.get(*j)).copied().unwrap_or(0.0);
                        row[side] = v * si * sj;
                    }
                    row
                })
                .collect(),
        };
        let mg = Multigrid::new(&scaled);
        let gmg = multigrid::GpuMultigrid::new(&self.device, &self.pipelines.mg32, &mg, &check)?;
        let k = krylov::Krylov::new(&self.device, &self.queue, pipes, &gmg, st.nx, st.ny, &scaled.diag, &side_major(&scaled), gx, gy)?;
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
            // f32 on the device: a tighter target than this cannot be told from rounding.
            let inner = k.solve(&rhs, system.symmetric, 1e-6, system.max_iterations.saturating_sub(used).max(50), &read, progress)?;
            used += inner.iterations;
            inner_converged = inner.converged;
            for ((xi, di), s) in report.x.iter_mut().zip(&inner.x).zip(&scale) {
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
