//! GPU-resident Krylov solvers for the temperature system: conjugate gradients (symmetric) and
//! BiCGSTAB (advection makes it one-sided), Jacobi preconditioned. Vectors and the scalar
//! recurrences (ρ, α, ω, …) stay on the device: dot products reduce in two stages (workgroup
//! partials, then one workgroup into a scalar slot) and the update kernels read their
//! coefficients from those slots. A batch of iterations is encoded in one submission and the
//! host reads the residual back once per batch, instead of synchronising on every dot product.
//!
//! The kernels are generated in one precision; the f64 set is the same text with `f32`
//! replaced by `f64` (devices with `SHADER_F64`).

use std::collections::HashMap;

use solidcraft_sim::{Progress, SimError};
use wgpu::util::DeviceExt;

type Result<T> = std::result::Result<T, SimError>;

/// Iterations encoded per submission (and between residual checks).
pub(crate) const BATCH: usize = 25;

/// Scalar slots.
const RZ: u32 = 0;
const PQ: u32 = 1;
const RR: u32 = 2;
const RZN: u32 = 3;
const RHO: u32 = 4;
const RHON: u32 = 5;
const ALPHA: u32 = 6;
const OMEGA: u32 = 7;
const R0V: u32 = 8;
const TT: u32 = 9;
const TS: u32 = 10;
const SLOTS: u64 = 16;

/// Vectors and their bindings (after params, diag, off, invdiag, b at 0..=4).
const VECTORS: [&str; 9] = ["x", "r", "p", "q", "z", "s", "t", "sh", "r0"];

const CG_INIT: &[&str] = &["init_cg", "dot_r_z", "fin_0"];
const CG_STEP: &[&str] = &["matvec_p_q", "dot_p_q", "fin_1", "cg_xr", "dot_r_r", "fin_2", "precond_r_z", "dot_r_z", "fin_3", "cg_p", "cg_shift"];
const BI_INIT: &[&str] = &["init_bi"];
const BI_STEP: &[&str] = &[
    "dot_r0_r",
    "fin_5",
    "bi_p",
    "bi_shift_rho",
    "precond_p_z",
    "matvec_z_q",
    "dot_r0_q",
    "fin_8",
    "bi_alpha",
    "bi_s",
    "precond_s_sh",
    "matvec_sh_t",
    "dot_t_t",
    "fin_9",
    "dot_t_s",
    "fin_10",
    "bi_omega",
    "bi_xr",
    "dot_r_r",
    "fin_2",
];

/// The kernels a solve uses: (entry point, runs on one workgroup).
fn entry_points() -> Vec<(String, bool)> {
    let mut v: Vec<(String, bool)> = Vec::new();
    for list in [CG_INIT, CG_STEP, BI_INIT, BI_STEP] {
        for k in list.iter() {
            let single = k.starts_with("fin_") || matches!(*k, "cg_shift" | "bi_shift_rho" | "bi_alpha" | "bi_omega");
            if !v.iter().any(|(n, _)| n == k) {
                v.push((k.to_string(), single));
            }
        }
    }
    v
}

/// The WGSL of every kernel, in f32 (callers substitute f64).
pub(crate) fn source() -> String {
    let mut s = String::new();
    s.push_str(
        r#"
const WG: u32 = 256u;
struct Params { nx: u32, ny: u32, n: u32, groups_x: u32, groups: u32, _a: u32, _b: u32, _c: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> diag: array<f32>;
@group(0) @binding(2) var<storage, read> off: array<f32>;
@group(0) @binding(3) var<storage, read> invdiag: array<f32>;
@group(0) @binding(4) var<storage, read> b: array<f32>;
"#,
    );
    for (i, v) in VECTORS.iter().enumerate() {
        s.push_str(&format!("@group(0) @binding({}) var<storage, read_write> {v}: array<f32>;\n", 5 + i));
    }
    s.push_str(&format!(
        r#"@group(0) @binding({}) var<storage, read_write> partials: array<f32>;
@group(0) @binding({}) var<storage, read_write> scal: array<f32>;
var<workgroup> red: array<f32, 256>;

fn cell(lid: u32, wg: vec3<u32>) -> u32 {{ return (wg.y * params.groups_x + wg.x) * WG + lid; }}
fn sdiv(a: f32, b: f32) -> f32 {{ if abs(b) > f32(0.0) {{ return a / b; }} return f32(0.0); }}
"#,
        5 + VECTORS.len(),
        6 + VECTORS.len()
    ));
    let matvec = |src: &str, dst: &str| {
        format!(
            r#"
@compute @workgroup_size(256)
fn matvec_{src}_{dst}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {{
    let i = cell(lid, wg);
    let n = params.n;
    if i >= n {{ return; }}
    let nx = params.nx;
    let plane = params.nx * params.ny;
    let ix = i % nx;
    let iy = (i / nx) % params.ny;
    var acc = diag[i] * {src}[i];
    if ix > 0u {{ acc = acc + off[i] * {src}[i - 1u]; }}
    if ix + 1u < nx {{ acc = acc + off[n + i] * {src}[i + 1u]; }}
    if iy > 0u {{ acc = acc + off[2u * n + i] * {src}[i - nx]; }}
    if iy + 1u < params.ny {{ acc = acc + off[3u * n + i] * {src}[i + nx]; }}
    if i >= plane {{ acc = acc + off[4u * n + i] * {src}[i - plane]; }}
    if i + plane < n {{ acc = acc + off[5u * n + i] * {src}[i + plane]; }}
    {dst}[i] = acc;
}}
"#
        )
    };
    let dot = |a: &str, bb: &str| {
        format!(
            r#"
@compute @workgroup_size(256)
fn dot_{a}_{bb}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {{
    let group = wg.y * params.groups_x + wg.x;
    let i = group * WG + lid;
    var v = f32(0.0);
    if i < params.n {{ v = {a}[i] * {bb}[i]; }}
    red[lid] = v;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {{
        if lid < k {{ red[lid] = red[lid] + red[lid + k]; }}
        workgroupBarrier();
    }}
    if lid == 0u {{ partials[group] = red[0]; }}
}}
"#
        )
    };
    let fin = |slot: u32| {
        format!(
            r#"
@compute @workgroup_size(256)
fn fin_{slot}(@builtin(local_invocation_index) lid: u32) {{
    var v = f32(0.0);
    for (var g = lid; g < params.groups; g = g + WG) {{ v = v + partials[g]; }}
    red[lid] = v;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {{
        if lid < k {{ red[lid] = red[lid] + red[lid + k]; }}
        workgroupBarrier();
    }}
    if lid == 0u {{ scal[{slot}u] = red[0]; }}
}}
"#
        )
    };
    let precond = |src: &str, dst: &str| {
        format!(
            r#"
@compute @workgroup_size(256)
fn precond_{src}_{dst}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {{
    let i = cell(lid, wg);
    if i >= params.n {{ return; }}
    {dst}[i] = invdiag[i] * {src}[i];
}}
"#
        )
    };
    let vector = |name: &str, body: &str| {
        format!(
            r#"
@compute @workgroup_size(256)
fn {name}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {{
    let i = cell(lid, wg);
    if i >= params.n {{ return; }}
    {body}
}}
"#
        )
    };
    let single = |name: &str, body: &str| {
        format!(
            r#"
@compute @workgroup_size(256)
fn {name}(@builtin(local_invocation_index) lid: u32) {{
    if lid != 0u {{ return; }}
    {body}
}}
"#
        )
    };
    for (a, d) in [("p", "q"), ("z", "q"), ("sh", "t")] {
        s.push_str(&matvec(a, d));
    }
    for (a, d) in [("p", "q"), ("r", "r"), ("r", "z"), ("r0", "r"), ("r0", "q"), ("t", "t"), ("t", "s")] {
        s.push_str(&dot(a, d));
    }
    for slot in [RZ, PQ, RR, RZN, RHON, R0V, TT, TS] {
        s.push_str(&fin(slot));
    }
    for (a, d) in [("r", "z"), ("p", "z"), ("s", "sh")] {
        s.push_str(&precond(a, d));
    }
    s.push_str(&vector("init_cg", "x[i] = f32(0.0); r[i] = b[i]; z[i] = invdiag[i] * b[i]; p[i] = z[i];"));
    s.push_str(&vector(
        "init_bi",
        &format!(
            "x[i] = f32(0.0); r[i] = b[i]; r0[i] = b[i]; p[i] = f32(0.0); q[i] = f32(0.0); \
             if i == 0u {{ scal[{RHO}u] = f32(1.0); scal[{ALPHA}u] = f32(1.0); scal[{OMEGA}u] = f32(1.0); }}"
        ),
    ));
    s.push_str(&vector("cg_xr", &format!("let alpha = sdiv(scal[{RZ}u], scal[{PQ}u]); x[i] = x[i] + alpha * p[i]; r[i] = r[i] - alpha * q[i];")));
    s.push_str(&vector("cg_p", &format!("let beta = sdiv(scal[{RZN}u], scal[{RZ}u]); p[i] = z[i] + beta * p[i];")));
    s.push_str(&single("cg_shift", &format!("scal[{RZ}u] = scal[{RZN}u];")));
    s.push_str(&vector(
        "bi_p",
        &format!(
            "let beta = sdiv(scal[{RHON}u], scal[{RHO}u]) * sdiv(scal[{ALPHA}u], scal[{OMEGA}u]); \
             p[i] = r[i] + beta * (p[i] - scal[{OMEGA}u] * q[i]);"
        ),
    ));
    s.push_str(&single("bi_shift_rho", &format!("scal[{RHO}u] = scal[{RHON}u];")));
    s.push_str(&single("bi_alpha", &format!("scal[{ALPHA}u] = sdiv(scal[{RHO}u], scal[{R0V}u]);")));
    s.push_str(&vector("bi_s", &format!("s[i] = r[i] - scal[{ALPHA}u] * q[i];")));
    s.push_str(&single("bi_omega", &format!("scal[{OMEGA}u] = sdiv(scal[{TS}u], scal[{TT}u]);")));
    s.push_str(&vector("bi_xr", &format!("x[i] = x[i] + scal[{ALPHA}u] * z[i] + scal[{OMEGA}u] * sh[i]; r[i] = s[i] - scal[{OMEGA}u] * t[i];")));
    s
}

/// The kernels in one precision on one device.
pub(crate) struct KrylovPipes {
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<String, (wgpu::ComputePipeline, bool)>,
    /// Bytes per element: 4 or 8.
    pub(crate) elem: u64,
}

impl KrylovPipes {
    pub(crate) fn new(device: &wgpu::Device, f64: bool) -> KrylovPipes {
        let src = source();
        let src = if f64 { src.replace("f32", "f64") } else { src };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solidcraft sim krylov"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let mut entries = vec![entry(0, wgpu::BufferBindingType::Uniform), entry(1, read), entry(2, read), entry(3, read), entry(4, read)];
        for k in 0..VECTORS.len() as u32 + 2 {
            entries.push(entry(5 + k, write));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("solidcraft sim krylov"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solidcraft sim krylov"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut pipelines = HashMap::new();
        for (name, single) in entry_points() {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(&name),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(name, (p, single));
        }
        KrylovPipes { layout, pipelines, elem: if f64 { 8 } else { 4 } }
    }
}

/// Vector bytes in a precision.
pub(crate) fn elems(values: &[f64], elem: u64) -> Vec<u8> {
    if elem == 8 { values.iter().flat_map(|v| v.to_le_bytes()).collect() } else { values.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect() }
}

/// One system `A x = b` on the device, ready to solve for several right-hand sides.
pub(crate) struct Krylov<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipes: &'a KrylovPipes,
    n: usize,
    gx: u32,
    gy: u32,
    bind: wgpu::BindGroup,
    b: wgpu::Buffer,
    x: wgpu::Buffer,
    scal: wgpu::Buffer,
}

/// The outcome of one solve.
pub(crate) struct Solved {
    pub x: Vec<f64>,
    pub iterations: usize,
    /// `|r|` at the last check (the right-hand side has unit norm, so this is relative).
    pub residual: f64,
    pub converged: bool,
}

impl<'a> Krylov<'a> {
    /// `off` is side-major: `off[side * n + cell]`.
    pub(crate) fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        pipes: &'a KrylovPipes,
        nx: usize,
        ny: usize,
        diag: &[f64],
        off: &[f64],
        invdiag: &[f64],
        gx: u32,
        gy: u32,
    ) -> Result<Krylov<'a>> {
        let n = diag.len();
        if off.len() != 6 * n || invdiag.len() != n || n == 0 {
            return Err(SimError::Invalid("stencil arrays do not match".into()));
        }
        let groups = (gx as usize) * (gy as usize);
        let e = pipes.elem;
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        let empty = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let u = |v: usize| u32::try_from(v).map_err(|_| SimError::TooLarge("too many cells for the GPU solver".into()));
        let mut params = Vec::with_capacity(32);
        for v in [u(nx)?, u(ny)?, u(n)?, gx, u(groups)?, 0, 0, 0] {
            params.extend_from_slice(&v.to_le_bytes());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("krylov params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let diag = storage("krylov diag", &elems(diag, e));
        let off = storage("krylov off", &elems(off, e));
        let invdiag = storage("krylov invdiag", &elems(invdiag, e));
        let b = empty("krylov b", e * n as u64);
        let vectors: Vec<wgpu::Buffer> = VECTORS.iter().map(|v| empty(v, e * n as u64)).collect();
        let partials = empty("krylov partials", e * groups as u64);
        let scal = empty("krylov scalars", e * SLOTS);
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: diag.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: off.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: invdiag.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b.as_entire_binding() },
        ];
        for (k, v) in vectors.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: 5 + k as u32, resource: v.as_entire_binding() });
        }
        entries.push(wgpu::BindGroupEntry { binding: 5 + VECTORS.len() as u32, resource: partials.as_entire_binding() });
        entries.push(wgpu::BindGroupEntry { binding: 6 + VECTORS.len() as u32, resource: scal.as_entire_binding() });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("krylov"), layout: &pipes.layout, entries: &entries });
        let x = vectors.into_iter().next().ok_or_else(|| SimError::Internal("no x vector".into()))?;
        Ok(Krylov { device, queue, pipes, n, gx, gy, bind, b, x, scal })
    }

    fn encode(&self, pass: &mut wgpu::ComputePass<'_>, kernels: &[&str]) -> Result<()> {
        for k in kernels {
            let (p, single) = self.pipes.pipelines.get(*k).ok_or_else(|| SimError::Internal(format!("no kernel {k}")))?;
            pass.set_pipeline(p);
            if *single {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.dispatch_workgroups(self.gx, self.gy, 1);
            }
        }
        Ok(())
    }

    /// Solve with `rhs` (unit norm) from `x = 0`; `tol` is the relative residual to reach.
    pub(crate) fn solve(
        &self,
        rhs: &[f64],
        symmetric: bool,
        tol: f64,
        max_iter: usize,
        read: &dyn Fn(&wgpu::Buffer, u64) -> Result<Vec<f32>>,
        progress: Progress,
    ) -> Result<Solved> {
        if rhs.len() != self.n {
            return Err(SimError::Invalid("right-hand side does not match the system".into()));
        }
        let e = self.pipes.elem;
        self.queue.write_buffer(&self.b, 0, &elems(rhs, e));
        let (init, step) = if symmetric { (CG_INIT, CG_STEP) } else { (BI_INIT, BI_STEP) };
        let read_scalar = |slot: u32| -> Result<f64> {
            let words = read(&self.scal, e * SLOTS)?;
            let at = slot as usize;
            if e == 8 {
                let lo = words.get(2 * at).copied().unwrap_or(f32::NAN).to_bits();
                let hi = words.get(2 * at + 1).copied().unwrap_or(f32::NAN).to_bits();
                Ok(f64::from_bits(u64::from(lo) | (u64::from(hi) << 32)))
            } else {
                Ok(f64::from(words.get(at).copied().unwrap_or(f32::NAN)))
            }
        };
        let mut done = 0usize;
        let mut residual = f64::INFINITY;
        let mut converged = false;
        let mut first = true;
        while done < max_iter {
            let k = BATCH.min(max_iter - done);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("krylov batch") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("krylov"), timestamp_writes: None });
                pass.set_bind_group(0, &self.bind, &[]);
                if first {
                    self.encode(&mut pass, init)?;
                    first = false;
                }
                for _ in 0..k {
                    self.encode(&mut pass, step)?;
                }
            }
            self.queue.submit([enc.finish()]);
            done += k;
            residual = read_scalar(RR)?.max(0.0).sqrt();
            if !residual.is_finite() {
                break;
            }
            if residual < tol {
                converged = true;
                break;
            }
            if !progress("thermal", done, max_iter) {
                return Err(SimError::Cancelled);
            }
        }
        let words = read(&self.x, e * self.n as u64)?;
        let x: Vec<f64> = if e == 8 {
            words.as_chunks::<2>().0.iter().map(|w| f64::from_bits(u64::from(w[0].to_bits()) | (u64::from(w[1].to_bits()) << 32))).collect()
        } else {
            words.into_iter().map(f64::from).collect()
        };
        if x.len() != self.n {
            return Err(SimError::Internal("solution readback is short".into()));
        }
        Ok(Solved { x, iterations: done, residual, converged })
    }
}
