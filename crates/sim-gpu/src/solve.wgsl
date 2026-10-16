// The linear solve of the temperature field: the 7-point stencil times a vector, in-place vector
// updates, Jacobi preconditioning and dot products as workgroup partial sums. The host drives
// conjugate gradients or BiCGSTAB with these.

const WG: u32 = 256u;

// The f64 variant of this shader is the same text with f32 replaced by f64 (see lib.rs).
struct Params {
    nx: u32,
    ny: u32,
    n: u32,
    groups_x: u32,
    a: f32,
    b: f32,
    c: f32,
    _p: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// matvec: diag, off (6 × n, side-major: -x, +x, -y, +y, -z, +z), x → y
@group(0) @binding(1) var<storage, read> diag: array<f32>;
@group(0) @binding(2) var<storage, read> off: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn cell_of(lid: u32, wg: vec3<u32>) -> u32 {
    return (wg.y * params.groups_x + wg.x) * WG + lid;
}

@compute @workgroup_size(256)
fn matvec(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let idx = cell_of(lid, wg);
    if idx >= params.n { return; }
    let n = params.n;
    let nx = params.nx;
    let plane = params.nx * params.ny;
    let i = idx % nx;
    let j = (idx / nx) % params.ny;
    var s = diag[idx] * x[idx];
    if i > 0u { s = s + off[0u * n + idx] * x[idx - 1u]; }
    if i + 1u < nx { s = s + off[1u * n + idx] * x[idx + 1u]; }
    if j > 0u { s = s + off[2u * n + idx] * x[idx - nx]; }
    if j + 1u < params.ny { s = s + off[3u * n + idx] * x[idx + nx]; }
    if idx >= plane { s = s + off[4u * n + idx] * x[idx - plane]; }
    if idx + plane < n { s = s + off[5u * n + idx] * x[idx + plane]; }
    y[idx] = s;
}

// y = a*y + b*x + c*diag   (diag doubles as a second vector input here)
@compute @workgroup_size(256)
fn combine(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let idx = cell_of(lid, wg);
    if idx >= params.n { return; }
    y[idx] = params.a * y[idx] + params.b * x[idx] + params.c * diag[idx];
}

// y = x * diag  (Jacobi: diag holds the inverse diagonal)
@compute @workgroup_size(256)
fn precond(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let idx = cell_of(lid, wg);
    if idx >= params.n { return; }
    y[idx] = x[idx] * diag[idx];
}

var<workgroup> sh: array<f32, 256>;

// partials[group] = Σ x·diag  (diag doubles as the second vector; y holds the partials)
@compute @workgroup_size(256)
fn dot_partial(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let group = wg.y * params.groups_x + wg.x;
    let idx = group * WG + lid;
    var v: f32 = 0.0;
    if idx < params.n { v = x[idx] * diag[idx]; }
    sh[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid < s { sh[lid] = sh[lid] + sh[lid + s]; }
        workgroupBarrier();
    }
    if lid == 0u { y[group] = sh[0]; }
}
