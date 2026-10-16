//! The GPU kernels against the CPU reference on the same problems (skipped without an adapter).

use solidcraft_geom::{Aabb3, Vec3};
use solidcraft_sim::{Accelerator, Air, CpuAccelerator, Fan, FlowSettings, Limits, Problem, SolidBody, Thermal, ThermalSystem, Wall, solve_with};

use super::{GpuAccelerator, GpuSimError};

fn boxed(name: &str, min: [f64; 3], max: [f64; 3], k: f64, power: f64) -> SolidBody {
    let (a, b) = (Vec3::from(min), Vec3::from(max));
    let p = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
    let positions = vec![
        p(a.x, a.y, a.z),
        p(b.x, a.y, a.z),
        p(b.x, b.y, a.z),
        p(a.x, b.y, a.z),
        p(a.x, a.y, b.z),
        p(b.x, a.y, b.z),
        p(b.x, b.y, b.z),
        p(a.x, b.y, b.z),
    ];
    let triangles =
        vec![[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]];
    SolidBody { name: name.into(), positions, triangles, conductivity: k, power_w: power, fixed_temperature_c: None, fluid: None }
}

fn domain(min: [f64; 3], max: [f64; 3]) -> Aabb3 {
    Aabb3 { min: Vec3::from(min), max: Vec3::from(max) }
}

fn problem(domain: Aabb3, cell: f64, bodies: Vec<SolidBody>) -> Problem {
    Problem {
        domain,
        cell_size: cell,
        bodies,
        walls: [Wall::Open; 6],
        air: Air::default(),
        fluids: Vec::new(),
        thermal: Thermal::default(),
        fans: Vec::new(),
        flow: None,
        limits: Limits::default(),
    }
}

fn gpu() -> Option<GpuAccelerator> {
    match GpuAccelerator::headless() {
        Ok(g) => {
            eprintln!("GPU: {}", g.adapter_name());
            Some(g)
        }
        // only a missing adapter is a reason to skip: a kernel that fails to build is a failure
        Err(GpuSimError::NoAdapter(e)) => {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("the GPU backend did not build: {e}"),
    }
}

fn go() -> impl FnMut(&str, usize, usize) -> bool {
    |_, _, _| true
}

#[test]
fn conduction_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 1e4, 1.0);
    let mut p = problem(domain([0.0; 3], [60.0; 3]), 2.0, vec![cube]);
    p.thermal.film_coefficient = 10.0;
    p.thermal.ambient_c = 25.0;
    let cpu = solve_with(&p, &CpuAccelerator, &mut go()).unwrap();
    let dev = solve_with(&p, &gpu, &mut go()).unwrap();
    assert!(dev.summary.thermal_method.starts_with("gpu-cg"), "{}", dev.summary.thermal_method);
    assert!(dev.summary.thermal_converged, "{:?}", dev.summary);
    let worst = cpu.thermal.temperature.iter().zip(&dev.thermal.temperature).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 0.05, "largest temperature difference {worst} K");
    assert!((dev.summary.heat_out_w - 1.0).abs() < 1e-2, "{:?}", dev.summary);
}

#[test]
fn fan_driven_channel_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let block = boxed("block", [25.0, 4.0, 0.0], [29.0, 8.0, 4.0], 200.0, 0.05);
    let mut p = problem(domain([0.0, 0.0, 0.0], [60.0, 12.0, 12.0]), 1.0, vec![block]);
    p.walls = [Wall::Open, Wall::Open, Wall::Closed, Wall::Closed, Wall::Closed, Wall::Closed];
    p.thermal.film_coefficient = 20.0;
    p.fans.push(Fan { name: "fan".into(), region: domain([1.0, 0.0, 0.0], [4.0, 12.0, 12.0]), velocity: Vec3::new(1.0, 0.0, 0.0) });
    p.flow = Some(FlowSettings { max_steps: 1000, buoyancy: false, tolerance: 1e-9, ..Default::default() });
    let cpu = solve_with(&p, &CpuAccelerator, &mut go()).unwrap();
    let dev = solve_with(&p, &gpu, &mut go()).unwrap();
    let (fc, fd) = (cpu.flow.as_ref().unwrap(), dev.flow.as_ref().unwrap());
    assert_eq!(fc.steps, fd.steps);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in fc.velocity.iter().zip(&fd.velocity) {
        for k in 0..3 {
            num += f64::from((a[k] - b[k]) * (a[k] - b[k]));
            den += f64::from(a[k] * a[k]);
        }
    }
    let rel = (num / den).sqrt();
    assert!(rel < 1e-3, "velocity fields differ by {rel} (relative L2)");
    assert!((fc.max_lattice_speed - fd.max_lattice_speed).abs() < 1e-3);
    assert!(dev.summary.thermal_method.starts_with("gpu-bicgstab"), "{}", dev.summary.thermal_method);
    let worst = cpu.thermal.temperature.iter().zip(&dev.thermal.temperature).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 0.1, "largest temperature difference {worst} K");
    assert!(dev.summary.flow.as_ref().unwrap().backend.starts_with("gpu"));
}

#[test]
fn buoyant_plume_runs_on_the_gpu() {
    let Some(gpu) = gpu() else { return };
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 200.0, 2.0);
    let mut p = problem(domain([0.0, 0.0, 0.0], [60.0, 60.0, 100.0]), 2.5, vec![cube]);
    p.flow = Some(FlowSettings { max_steps: 600, buoyancy: true, coupling_iterations: 1, ..Default::default() });
    let dev = solve_with(&p, &gpu, &mut go()).unwrap();
    let f = dev.summary.flow.as_ref().unwrap();
    assert!(f.max_speed_m_s > 0.01, "warm air rises: {f:?}");
    let above = dev.velocity_at(Vec3::new(30.0, 30.0, 60.0)).unwrap();
    assert!(above.z > 0.0, "upward over the cube: {above:?}");
}

/// The solver alone on a 1-D Poisson system with a known answer, against the CPU reference.
#[test]
fn stencil_solver_matches_the_cpu_reference_on_a_poisson_line() {
    use solidcraft_sim::{Accelerator, Stencil, ThermalSystem};
    let Some(gpu) = gpu() else { return };
    let n = 300usize;
    let mut diag: Vec<f64> = vec![2.0; n];
    let mut off: Vec<[f64; 6]> = vec![[0.0; 6]; n];
    for (i, o) in off.iter_mut().enumerate() {
        if i > 0 {
            o[0] = -1.0;
        }
        if i + 1 < n {
            o[1] = -1.0;
        }
    }
    // a stiff variant: alternate a huge and a tiny conductance
    for (i, d) in diag.iter_mut().enumerate() {
        let k = if i % 7 == 0 { 1e4 } else { 1.0 };
        *d = 2.0 * k;
        off[i][0] *= k;
        off[i][1] *= k;
    }
    // symmetrise: off[i][1] must equal off[i+1][0]
    for i in 0..n - 1 {
        let g = off[i][1].min(off[i + 1][0]);
        off[i][1] = g;
        off[i + 1][0] = g;
        diag[i] = 0.0;
    }
    for i in 0..n {
        diag[i] = -off[i][0] - off[i][1] + if i == 0 || i == n - 1 { 1.0 } else { 0.0 };
    }
    let rhs: Vec<f64> = (0..n).map(|i| if i == n / 2 { 1.0 } else { 0.0 }).collect();
    let system = ThermalSystem { stencil: Stencil { nx: n, ny: 1, diag, off }, rhs, symmetric: true, tolerance: 1e-7, max_iterations: 5000 };
    let cpu = CpuAccelerator.thermal(&system, &mut go()).unwrap();
    let dev = gpu.thermal(&system, &mut go()).unwrap();
    eprintln!("cpu {} it res {:.2e}; gpu {} {} it res {:.2e}", cpu.iterations, cpu.residual, dev.method, dev.iterations, dev.residual);
    let worst = cpu.x.iter().zip(&dev.x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let scale = cpu.x.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(dev.method.starts_with("gpu"), "{}", dev.method);
    assert!(worst < 1e-5 * scale, "worst {worst} of {scale}");
}

#[test]
#[ignore]
fn probe_f64_support() {
    use solidcraft_gpu::select::{Policy, pick};
    let (adapter, report) = pick(Policy::Auto).unwrap();
    eprintln!("adapter {} f64 {}", report.name, adapter.features().contains(wgpu::Features::SHADER_F64));
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("probe"),
        required_features: wgpu::Features::SHADER_F64,
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .unwrap();
    let scopes = [wgpu::ErrorFilter::Validation].map(|f| device.push_error_scope(f));
    let _m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f64 probe"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read_write> x: array<f64>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { x[g.x] = x[g.x] * 2.0lf + 1.0lf; }"
                .into(),
        ),
    });
    for s in scopes {
        let e = pollster::block_on(s.pop());
        eprintln!("validation: {e:?}");
    }
}

/// Without doubles (Apple GPUs), the scaled f32 solve with refinement carries a moderately
/// stiff system, and a system it cannot carry is finished by the f64 reference, which the
/// report says.
#[test]
fn single_precision_path_refines_or_admits_defeat() {
    let Some(gpu) = gpu() else { return };
    let gpu = gpu.without_f64();
    assert!(!gpu.has_f64());
    // conductive block in moving air: moderate stiffness
    let block = boxed("block", [25.0, 4.0, 0.0], [29.0, 8.0, 4.0], 200.0, 0.05);
    let mut p = problem(domain([0.0, 0.0, 0.0], [60.0, 12.0, 12.0]), 1.0, vec![block]);
    p.walls = [Wall::Open, Wall::Open, Wall::Closed, Wall::Closed, Wall::Closed, Wall::Closed];
    p.thermal.film_coefficient = 20.0;
    p.fans.push(Fan { name: "fan".into(), region: domain([1.0, 0.0, 0.0], [4.0, 12.0, 12.0]), velocity: Vec3::new(1.0, 0.0, 0.0) });
    p.flow = Some(FlowSettings { max_steps: 600, buoyancy: false, ..Default::default() });
    let cpu = solve_with(&p, &CpuAccelerator, &mut go()).unwrap();
    let dev = solve_with(&p, &gpu, &mut go()).unwrap();
    eprintln!(
        "f32 path: {} in {} iterations, residual {:.2e}",
        dev.summary.thermal_method, dev.summary.thermal_iterations, dev.summary.thermal_residual
    );
    assert!(dev.summary.thermal_converged, "{:?}", dev.summary);
    let worst = cpu.thermal.temperature.iter().zip(&dev.thermal.temperature).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 0.1, "largest temperature difference {worst} K");
    // a hot metal cube in still air: too stiff for f32, and the report says who finished it
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 1e4, 1.0);
    let p = problem(domain([0.0; 3], [60.0; 3]), 2.0, vec![cube]);
    let cpu = solve_with(&p, &CpuAccelerator, &mut go()).unwrap();
    let dev = solve_with(&p, &gpu, &mut go()).unwrap();
    eprintln!("f32 path, stiff: {}", dev.summary.thermal_method);
    assert!(dev.summary.thermal_converged);
    assert!(dev.summary.thermal_method.contains("cpu-cg (GPU f32 insufficient)"), "{}", dev.summary.thermal_method);
    let (a, b) = (cpu.summary.bodies[0].mean_c, dev.summary.bodies[0].mean_c);
    assert!((a - b).abs() < 0.01, "{a} vs {b}");
}

/// Two lattices (air and water) on the device agree with the CPU reference.
#[test]
fn water_loop_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let water = |name: &str, min: [f64; 3], max: [f64; 3]| {
        let mut b = boxed(name, min, max, 0.6, 0.0);
        b.fluid = Some(0);
        b
    };
    let bodies = vec![
        water("leg_a", [10.0, 6.0, 14.0], [90.0, 12.0, 20.0]),
        water("leg_b", [10.0, 28.0, 14.0], [90.0, 34.0, 20.0]),
        water("leg_c", [10.0, 6.0, 14.0], [16.0, 34.0, 20.0]),
        water("leg_d", [84.0, 6.0, 14.0], [90.0, 34.0, 20.0]),
        boxed("block", [40.0, 6.0, 6.0], [60.0, 14.0, 14.0], 385.0, 2.0),
    ];
    let mut p = problem(domain([0.0, 0.0, 0.0], [100.0, 40.0, 30.0]), 2.0, bodies);
    p.fluids.push(solidcraft_sim::Fluid::water());
    p.fans.push(Fan { name: "pump".into(), region: domain([30.0, 6.0, 14.0], [36.0, 12.0, 20.0]), velocity: Vec3::new(0.5, 0.0, 0.0) });
    p.flow = Some(FlowSettings { max_steps: 1500, buoyancy: false, tolerance: 1e-9, ..Default::default() });
    let cpu = solve_with(&p, &CpuAccelerator, &mut go()).unwrap();
    let dev = solve_with(&p, &gpu, &mut go()).unwrap();
    assert_eq!(dev.summary.fluids.len(), 2);
    let (fc, fd) = (cpu.flow.as_ref().unwrap(), dev.flow.as_ref().unwrap());
    assert_eq!(fc.zones.len(), fd.zones.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in fc.velocity.iter().zip(&fd.velocity) {
        for k in 0..3 {
            num += f64::from((a[k] - b[k]) * (a[k] - b[k]));
            den += f64::from(a[k] * a[k]);
        }
    }
    let rel = (num / den).sqrt();
    assert!(rel < 1e-3, "velocity fields differ by {rel} (relative L2)");
    let worst = cpu.thermal.temperature.iter().zip(&dev.thermal.temperature).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 0.1, "largest temperature difference {worst} K");
    assert!(dev.summary.fluids[1].max_speed_m_s > 0.3, "{:?}", dev.summary.fluids);
}

#[test]
#[ignore]
fn print_adapter_limits() {
    use solidcraft_gpu::select::{Policy, pick};
    let (adapter, report) = pick(Policy::Auto).unwrap();
    let l = adapter.limits();
    eprintln!(
        "{}: max_buffer_size {} MB, max_storage_buffer_binding_size {} MB, storage buffers per stage {}",
        report.name,
        l.max_buffer_size >> 20,
        l.max_storage_buffer_binding_size >> 20,
        l.max_storage_buffers_per_shader_stage
    );
}

/// Iterating far past convergence (a tolerance nothing reaches) drives BiCGSTAB into breakdown:
/// the recurrence scalars collapse and round-off takes over. The resident solver must restart
/// and hand back its best iterate, never a diverged one.
#[test]
fn iterating_past_convergence_never_returns_garbage() {
    use solidcraft_sim::{Accelerator, Stencil, ThermalSystem};
    let Some(gpu) = gpu() else { return };
    let (nx, ny, nz) = (40usize, 30usize, 20usize);
    let n = nx * ny * nz;
    let mut diag = vec![0.0f64; n];
    let mut off = vec![[0.0f64; 6]; n];
    for idx in 0..n {
        let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
        let has = [i > 0, i + 1 < nx, j > 0, j + 1 < ny, k > 0, k + 1 < nz];
        // conduction plus an upwind advection along +x (one-sided, like moving air)
        for (side, h) in has.iter().enumerate() {
            if *h {
                let mut g = 1.0;
                if side == 0 {
                    g += 2.0;
                }
                off[idx][side] = -g;
                diag[idx] += if side == 1 { 1.0 + 2.0 } else { 1.0 };
            }
        }
        diag[idx] += 1e-3;
    }
    let rhs: Vec<f64> = (0..n).map(|i| if i % 97 == 0 { 1.0 } else { 0.0 }).collect();
    for symmetric in [false, true] {
        let mut sys = ThermalSystem {
            stencil: Stencil { nx, ny, diag: diag.clone(), off: off.clone() },
            rhs: rhs.clone(),
            symmetric,
            tolerance: 1e-30,
            max_iterations: 3000,
        };
        if symmetric {
            // the symmetric variant: conduction only
            for (idx, o) in sys.stencil.off.iter_mut().enumerate() {
                let total: f64 = o.iter().map(|v| -v).sum();
                for v in o.iter_mut() {
                    if *v != 0.0 {
                        *v = -1.0;
                    }
                }
                sys.stencil.diag[idx] = o.iter().map(|v| -v).sum::<f64>() + 1e-3;
                let _ = total;
            }
        }
        let dev = gpu.thermal(&sys, &mut go()).unwrap();
        assert!(dev.x.iter().all(|v| v.is_finite()), "{}: finite", dev.method);
        assert!(dev.residual < 1e-8, "{}: best residual {}", dev.method, dev.residual);
        let mut ax = vec![0.0; n];
        sys.stencil.apply(&dev.x, &mut ax);
        let res = ax.iter().zip(&rhs).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt() / rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
        assert!(res < 1e-6, "{}: true residual {res}", dev.method);
    }
}

/// The GPU solver on a system `sim-bench --dump-thermal` wrote: the path in
/// `SOLIDCRAFT_THERMAL_DUMP`, residuals traced with `SOLIDCRAFT_SIM_DEBUG`.
#[test]
#[ignore]
fn dumped_system_on_the_gpu() {
    let Some(path) = std::env::var_os("SOLIDCRAFT_THERMAL_DUMP") else { return };
    let bytes = std::fs::read(path).expect("read the dump");
    let system = ThermalSystem::from_bytes(&bytes, 1e-8, 20_000).expect("parse the dump");
    let gpu = GpuAccelerator::headless().expect("a GPU");
    let started = std::time::Instant::now();
    let solved = gpu.thermal(&system, &mut |_, _, _| true).expect("solve");
    let mut ax = vec![0.0; system.rhs.len()];
    system.stencil.apply(&solved.x, &mut ax);
    let bnorm = system.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let true_res = system.rhs.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum::<f64>().sqrt() / bnorm;
    let hottest = solved.x.iter().cloned().fold(f64::MIN, f64::max);
    eprintln!(
        "{} {}: {} iterations, converged {}, residual {:.3e}, true residual {true_res:.3e}, hottest {hottest:.2} K above ambient, {:.1} s",
        gpu.name(),
        solved.method,
        solved.iterations,
        solved.converged,
        solved.residual,
        started.elapsed().as_secs_f64()
    );
}
