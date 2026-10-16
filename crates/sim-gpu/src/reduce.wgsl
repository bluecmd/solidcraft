// Convergence check of the lattice: per workgroup, the sums over air cells of |u - u_prev|² and
// |u|² and the largest |u|², for the host to combine.

const WG: u32 = 256u;

struct Params {
    n: u32,
    groups_x: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> u: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> u_prev: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> sd: array<f32, 256>;
var<workgroup> su: array<f32, 256>;
var<workgroup> mu: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let group = wg.y * params.groups_x + wg.x;
    let cell = group * WG + lid;
    var d2 = 0.0;
    var u2 = 0.0;
    if cell < params.n && (flags[cell] & 0xffu) != 0u {
        let a = u[cell].xyz;
        let d = a - u_prev[cell].xyz;
        d2 = dot(d, d);
        u2 = dot(a, a);
    }
    sd[lid] = d2;
    su[lid] = u2;
    mu[lid] = u2;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid < s {
            sd[lid] = sd[lid] + sd[lid + s];
            su[lid] = su[lid] + su[lid + s];
            mu[lid] = max(mu[lid], mu[lid + s]);
        }
        workgroupBarrier();
    }
    if lid == 0u {
        partials[group] = vec4<f32>(sd[0], su[0], mu[0], 0.0);
    }
}
