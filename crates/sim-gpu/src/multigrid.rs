//! The V-cycle of `solidcraft_sim::Multigrid` on the device: every level's stencil and vectors
//! stay resident, and one cycle encodes as a run of dispatches that the Krylov batches splice in
//! as their preconditioner. The kernels are one text in f32, substituted to f64 like the Krylov
//! ones; the sweeps, levels and counts are the CPU cycle's, so both backends apply the same
//! operator.

use std::collections::HashMap;

use solidcraft_sim::multigrid::{COARSE_SWEEPS, SMOOTH};
use solidcraft_sim::{Multigrid, SimError, Stencil};
use wgpu::util::DeviceExt;

use crate::krylov::elems;

type Result<T> = std::result::Result<T, SimError>;

/// The WGSL of the level kernels, in f32 (callers substitute f64).
fn source() -> String {
    let mut s = String::from(
        r#"
const WG: u32 = 256u;
struct Params { nx: u32, ny: u32, n: u32, groups_x: u32, fnx: u32, fny: u32, fnz: u32, cnx: u32, cny: u32, _a: u32, _b: u32, _c: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> diag: array<f32>;
"#,
    );
    for side in 0..6 {
        s.push_str(&format!("@group(0) @binding({}) var<storage, read> off{side}: array<f32>;\n", 2 + side));
    }
    s.push_str(
        r#"@group(0) @binding(8) var<storage, read_write> x: array<f32>;
@group(0) @binding(9) var<storage, read_write> b: array<f32>;
@group(0) @binding(10) var<storage, read_write> r: array<f32>;
@group(0) @binding(11) var<storage, read> fine_r: array<f32>;
@group(0) @binding(12) var<storage, read> coarse_x: array<f32>;

fn cell(lid: u32, wg: vec3<u32>) -> u32 { return (wg.y * params.groups_x + wg.x) * WG + lid; }

// the off-diagonal part of row i applied to x
fn neighbours(i: u32) -> f32 {
    let n = params.n;
    let nx = params.nx;
    let plane = params.nx * params.ny;
    let ix = i % nx;
    let iy = (i / nx) % params.ny;
    var acc = f32(0.0);
    if ix > 0u { acc = acc + off0[i] * x[i - 1u]; }
    if ix + 1u < nx { acc = acc + off1[i] * x[i + 1u]; }
    if iy > 0u { acc = acc + off2[i] * x[i - nx]; }
    if iy + 1u < params.ny { acc = acc + off3[i] * x[i + nx]; }
    if i >= plane { acc = acc + off4[i] * x[i - plane]; }
    if i + plane < n { acc = acc + off5[i] * x[i + plane]; }
    return acc;
}

// Gauss-Seidel on the cells of one colour: they read only the other colour, so a sweep is one
// dispatch writing x in place.
fn sweep(i: u32, colour: u32) {
    let nx = params.nx;
    let plane = nx * params.ny;
    let c = (i % nx + (i / nx) % params.ny + i / plane) & 1u;
    if c != colour { return; }
    let d = diag[i];
    if abs(d) > f32(0.0) { x[i] = (b[i] - neighbours(i)) / d; } else { x[i] = f32(0.0); }
}

@compute @workgroup_size(256)
fn zero_x(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = cell(lid, wg);
    if i >= params.n { return; }
    x[i] = f32(0.0);
}

@compute @workgroup_size(256)
fn gs_red(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = cell(lid, wg);
    if i >= params.n { return; }
    sweep(i, 0u);
}

@compute @workgroup_size(256)
fn gs_black(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = cell(lid, wg);
    if i >= params.n { return; }
    sweep(i, 1u);
}

@compute @workgroup_size(256)
fn residual(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = cell(lid, wg);
    if i >= params.n { return; }
    r[i] = b[i] - diag[i] * x[i] - neighbours(i);
}

// this level's b = the sum of the finer level's residual over each 2 x 2 x 2 block
@compute @workgroup_size(256)
fn restrict_sum(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = cell(lid, wg);
    if i >= params.n { return; }
    let cx = params.nx;
    let ic = i % cx;
    let jc = (i / cx) % params.ny;
    let kc = i / (cx * params.ny);
    var s = f32(0.0);
    for (var k = 2u * kc; k < min(2u * kc + 2u, params.fnz); k = k + 1u) {
        for (var j = 2u * jc; j < min(2u * jc + 2u, params.fny); j = j + 1u) {
            for (var ii = 2u * ic; ii < min(2u * ic + 2u, params.fnx); ii = ii + 1u) {
                s = s + fine_r[ii + params.fnx * (j + params.fny * k)];
            }
        }
    }
    b[i] = s;
}

// x += the coarser level's x of the block this cell is in
@compute @workgroup_size(256)
fn prolong_add(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = cell(lid, wg);
    if i >= params.n { return; }
    let nx = params.nx;
    let plane = nx * params.ny;
    let ix = i % nx;
    let iy = (i / nx) % params.ny;
    let iz = i / plane;
    let block = (ix / 2u) + params.cnx * ((iy / 2u) + params.cny * (iz / 2u));
    x[i] = x[i] + coarse_x[block];
}
"#,
    );
    s
}

const KERNELS: [&str; 7] = ["zero_x", "gs_red", "gs_black", "residual", "restrict_sum", "prolong_add", "zero_x"];

/// The level kernels in one precision on one device.
pub(crate) struct MultigridPipes {
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Bytes per element: 4 or 8.
    elem: u64,
}

impl MultigridPipes {
    pub(crate) fn new(device: &wgpu::Device, f64: bool) -> MultigridPipes {
        let src = source();
        let src = if f64 { src.replace("f32", "f64") } else { src };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solidcraft sim multigrid"),
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
        let mut entries = vec![entry(0, wgpu::BufferBindingType::Uniform)];
        entries.extend((1..8).map(|b| entry(b, read)));
        entries.extend((8..11).map(|b| entry(b, write)));
        entries.extend((11..13).map(|b| entry(b, read)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("solidcraft sim multigrid"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solidcraft sim multigrid"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut pipelines = HashMap::new();
        for name in KERNELS {
            if pipelines.contains_key(name) {
                continue;
            }
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(name.to_string(), p);
        }
        MultigridPipes { layout, pipelines, elem: if f64 { 8 } else { 4 } }
    }
}

struct Level {
    gx: u32,
    gy: u32,
    bind: wgpu::BindGroup,
    x: wgpu::Buffer,
    b: wgpu::Buffer,
}

/// One hierarchy, resident on the device.
pub(crate) struct GpuMultigrid<'a> {
    pipes: &'a MultigridPipes,
    levels: Vec<Level>,
}

fn dims(a: &Stencil) -> (usize, usize, usize) {
    (a.nx, a.ny, a.len().checked_div(a.nx * a.ny).unwrap_or(0))
}

impl<'a> GpuMultigrid<'a> {
    /// Upload `mg`; `check` vets each buffer's size against the device.
    pub(crate) fn new(
        device: &wgpu::Device,
        pipes: &'a MultigridPipes,
        mg: &Multigrid,
        check: &dyn Fn(u64, &str) -> Result<()>,
    ) -> Result<GpuMultigrid<'a>> {
        let e = pipes.elem;
        let stencils = mg.stencils();
        if stencils.is_empty() {
            return Err(SimError::Internal("an empty multigrid hierarchy".into()));
        }
        let u = |v: usize| u32::try_from(v).map_err(|_| SimError::TooLarge("too many cells for the GPU solver".into()));
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16][..] } else { bytes },
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
        struct Raw {
            gx: u32,
            gy: u32,
            params: wgpu::Buffer,
            diag: wgpu::Buffer,
            offs: Vec<wgpu::Buffer>,
            x: wgpu::Buffer,
            b: wgpu::Buffer,
            r: wgpu::Buffer,
        }
        let mut raws: Vec<Raw> = Vec::with_capacity(stencils.len());
        for (l, a) in stencils.iter().enumerate() {
            let n = a.len();
            check(e * n as u64, "a multigrid level")?;
            let (gx, gy) = crate::groups(n, crate::VEC_WG)?;
            let (nx, ny, _) = dims(a);
            let (fnx, fny, fnz) = l.checked_sub(1).and_then(|f| stencils.get(f)).map_or((0, 0, 0), dims);
            let (cnx, cny, _) = stencils.get(l + 1).map_or((0, 0, 0), dims);
            let mut params = Vec::with_capacity(48);
            for v in [u(nx)?, u(ny)?, u(n)?, gx, u(fnx)?, u(fny)?, u(fnz)?, u(cnx)?, u(cny)?, 0, 0, 0] {
                params.extend_from_slice(&v.to_le_bytes());
            }
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multigrid params"),
                contents: &params,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let diag = storage("multigrid diag", &elems(&a.diag, e));
            let offs = (0..6)
                .map(|side| {
                    let column: Vec<f64> = a.off.iter().map(|o| o[side]).collect();
                    storage("multigrid off", &elems(&column, e))
                })
                .collect();
            raws.push(Raw {
                gx,
                gy,
                params,
                diag,
                offs,
                x: empty("multigrid x", e * n as u64),
                b: empty("multigrid b", e * n as u64),
                r: empty("multigrid r", e * n as u64),
            });
        }
        let dummy = empty("multigrid none", 16);
        let mut levels = Vec::with_capacity(raws.len());
        for (l, raw) in raws.iter().enumerate() {
            let fine_r = l.checked_sub(1).and_then(|f| raws.get(f)).map_or(&dummy, |f| &f.r);
            let coarse_x = raws.get(l + 1).map_or(&dummy, |c| &c.x);
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: raw.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: raw.diag.as_entire_binding() },
            ];
            for (k, o) in raw.offs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: 2 + k as u32, resource: o.as_entire_binding() });
            }
            entries.push(wgpu::BindGroupEntry { binding: 8, resource: raw.x.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 9, resource: raw.b.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 10, resource: raw.r.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 11, resource: fine_r.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 12, resource: coarse_x.as_entire_binding() });
            let bind =
                device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("multigrid level"), layout: &pipes.layout, entries: &entries });
            levels.push(Level { gx: raw.gx, gy: raw.gy, bind, x: raw.x.clone(), b: raw.b.clone() });
        }
        Ok(GpuMultigrid { pipes, levels })
    }

    /// The finest level's right-hand side (the cycle's input).
    pub(crate) fn input(&self) -> Option<&wgpu::Buffer> {
        self.levels.first().map(|l| &l.b)
    }

    /// The finest level's x (the cycle's output).
    pub(crate) fn output(&self) -> Option<&wgpu::Buffer> {
        self.levels.first().map(|l| &l.x)
    }

    fn run(&self, pass: &mut wgpu::ComputePass<'_>, level: usize, kernel: &str) -> Result<()> {
        let l = self.levels.get(level).ok_or_else(|| SimError::Internal("no such multigrid level".into()))?;
        let p = self.pipes.pipelines.get(kernel).ok_or_else(|| SimError::Internal(format!("no kernel {kernel}")))?;
        pass.set_bind_group(0, &l.bind, &[]);
        pass.set_pipeline(p);
        pass.dispatch_workgroups(l.gx, l.gy, 1);
        Ok(())
    }

    /// One V-cycle: `output = V(input)`.
    pub(crate) fn encode(&self, pass: &mut wgpu::ComputePass<'_>) -> Result<()> {
        let last = self.levels.len().checked_sub(1).ok_or_else(|| SimError::Internal("an empty multigrid hierarchy".into()))?;
        for l in 0..last {
            self.run(pass, l, "zero_x")?;
            for _ in 0..SMOOTH {
                self.run(pass, l, "gs_red")?;
                self.run(pass, l, "gs_black")?;
            }
            self.run(pass, l, "residual")?;
            self.run(pass, l + 1, "restrict_sum")?;
        }
        self.run(pass, last, "zero_x")?;
        for _ in 0..COARSE_SWEEPS {
            self.run(pass, last, "gs_red")?;
            self.run(pass, last, "gs_black")?;
        }
        for _ in 0..COARSE_SWEEPS {
            self.run(pass, last, "gs_black")?;
            self.run(pass, last, "gs_red")?;
        }
        for l in (0..last).rev() {
            self.run(pass, l, "prolong_add")?;
            for _ in 0..SMOOTH {
                self.run(pass, l, "gs_black")?;
                self.run(pass, l, "gs_red")?;
            }
        }
        Ok(())
    }
}
