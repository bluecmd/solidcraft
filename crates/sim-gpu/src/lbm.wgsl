// D3Q19 lattice-Boltzmann step on the GPU: the twin of solidcraft_sim::lattice::collide and the
// CPU streaming in solidcraft_sim::backend, as one in-place kernel.
//
// Streaming is the AA pattern (Bailey et al. 2009): one population buffer, no copy. On an even
// step a cell reads its own 19 slots, collides, and writes each result into the slot of the
// opposite direction. On an odd step a cell pulls direction i from the neighbour at -c_i (its
// opposite slot, where the even step left it), collides, and pushes direction i into the
// neighbour at +c_i (its direct slot, where the next even step reads it). A solid or closed
// neighbour turns both the pull and the push onto the cell's own slots, which is bounce-back for
// free; an open wall supplies the ambient equilibrium at the cell's velocity.
//
// Populations are stored direction-major (f[a * n + cell]) so neighbouring threads read
// neighbouring words.

const Q: u32 = 19u;
const CELL_SOLID: u32 = 0u;
const CELL_FAN: u32 = 2u;
const FAN_SHIFT: u32 = 8u;
const WG: u32 = 128u;

struct Params {
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
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> f: array<f32>;
// Velocity (xyz) and density (w) per cell, from the last collision.
@group(0) @binding(3) var<storage, read_write> u: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> fans: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> dtemp: array<f32>;

fn c_of(a: u32) -> vec3<i32> {
    switch a {
        case 0u: { return vec3<i32>(0, 0, 0); }
        case 1u: { return vec3<i32>(1, 0, 0); }
        case 2u: { return vec3<i32>(-1, 0, 0); }
        case 3u: { return vec3<i32>(0, 1, 0); }
        case 4u: { return vec3<i32>(0, -1, 0); }
        case 5u: { return vec3<i32>(0, 0, 1); }
        case 6u: { return vec3<i32>(0, 0, -1); }
        case 7u: { return vec3<i32>(1, 1, 0); }
        case 8u: { return vec3<i32>(-1, -1, 0); }
        case 9u: { return vec3<i32>(1, -1, 0); }
        case 10u: { return vec3<i32>(-1, 1, 0); }
        case 11u: { return vec3<i32>(1, 0, 1); }
        case 12u: { return vec3<i32>(-1, 0, -1); }
        case 13u: { return vec3<i32>(1, 0, -1); }
        case 14u: { return vec3<i32>(-1, 0, 1); }
        case 15u: { return vec3<i32>(0, 1, 1); }
        case 16u: { return vec3<i32>(0, -1, -1); }
        case 17u: { return vec3<i32>(0, 1, -1); }
        default: { return vec3<i32>(0, -1, 1); }
    }
}

fn opp(a: u32) -> u32 {
    if a == 0u { return 0u; }
    // pairs (1,2) (3,4) ... : odd ↔ next
    if (a & 1u) == 1u { return a + 1u; }
    return a - 1u;
}

fn w_of(a: u32) -> f32 {
    if a == 0u { return 1.0 / 3.0; }
    if a < 7u { return 1.0 / 18.0; }
    return 1.0 / 36.0;
}

fn feq(a: u32, rho: f32, vel: vec3<f32>) -> f32 {
    let c = vec3<f32>(c_of(a));
    let cu = dot(c, vel);
    let usq = 1.5 * dot(vel, vel);
    return w_of(a) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
}

// The neighbour of `cell` at `d` (a lattice step), or the kind of boundary crossed:
// returns (index, kind) with kind 0 = air cell, 1 = bounce (solid or closed wall), 2 = open wall.
fn neighbour(i: i32, j: i32, k: i32, d: vec3<i32>) -> vec2<u32> {
    let ni = i + d.x;
    let nj = j + d.y;
    let nk = k + d.z;
    let nx = i32(params.nx);
    let ny = i32(params.ny);
    let nz = i32(params.nz);
    if ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz {
        var open = true;
        if ni < 0 && (params.open_walls & 1u) == 0u { open = false; }
        if ni >= nx && (params.open_walls & 2u) == 0u { open = false; }
        if nj < 0 && (params.open_walls & 4u) == 0u { open = false; }
        if nj >= ny && (params.open_walls & 8u) == 0u { open = false; }
        if nk < 0 && (params.open_walls & 16u) == 0u { open = false; }
        if nk >= nz && (params.open_walls & 32u) == 0u { open = false; }
        if open { return vec2<u32>(0u, 2u); }
        return vec2<u32>(0u, 1u);
    }
    let idx = u32((nk * ny + nj) * nx + ni);
    if (flags[idx] & 0xffu) == CELL_SOLID { return vec2<u32>(idx, 1u); }
    return vec2<u32>(idx, 0u);
}

@compute @workgroup_size(128)
fn step(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let cell = (wg.y * params.groups_x + wg.x) * WG + lid;
    if cell >= params.n { return; }
    let flag = flags[cell];
    if (flag & 0xffu) == CELL_SOLID { return; }
    let n = params.n;
    let i = i32(cell % params.nx);
    let j = i32((cell / params.nx) % params.ny);
    let k = i32(cell / (params.nx * params.ny));
    let odd = (params.step & 1u) == 1u;
    let u_old = u[cell].xyz;

    // ---- load ----
    var pop: array<f32, 19>;
    if odd {
        for (var a = 0u; a < Q; a = a + 1u) {
            let c = c_of(a);
            let nb = neighbour(i, j, k, -c);
            if nb.y == 0u {
                pop[a] = f[opp(a) * n + nb.x];
            } else if nb.y == 1u {
                pop[a] = f[a * n + cell];
            } else {
                pop[a] = feq(a, 1.0, u_old);
            }
        }
    } else {
        for (var a = 0u; a < Q; a = a + 1u) {
            pop[a] = f[a * n + cell];
        }
    }

    // ---- collide (TRT + Smagorinsky + Guo forcing; fans at equilibrium) ----
    var dens = 0.0;
    var mom = vec3<f32>(0.0);
    for (var a = 0u; a < Q; a = a + 1u) {
        dens = dens + pop[a];
        mom = mom + vec3<f32>(c_of(a)) * pop[a];
    }
    if !(dens > 0.0) { dens = 1.0; }
    let force = vec3<f32>(0.0, 0.0, params.g_lat * dtemp[cell]);
    var vel = (mom + 0.5 * force) / dens;
    var post: array<f32, 19>;
    if (flag & 0xffu) == CELL_FAN {
        var ramp = 1.0;
        if params.ramp_steps > 0u {
            ramp = min(1.0, f32(params.step + 1u) / f32(params.ramp_steps));
        }
        vel = fans[flag >> FAN_SHIFT].xyz * ramp;
        for (var a = 0u; a < Q; a = a + 1u) {
            post[a] = feq(a, dens, vel);
        }
    } else {
        var eq: array<f32, 19>;
        for (var a = 0u; a < Q; a = a + 1u) {
            eq[a] = feq(a, dens, vel);
        }
        var tau = params.tau0;
        if params.cs2 > 0.0 {
            var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
            for (var a = 0u; a < Q; a = a + 1u) {
                let neq = pop[a] - eq[a];
                let c = vec3<f32>(c_of(a));
                pxx = pxx + c.x * c.x * neq;
                pyy = pyy + c.y * c.y * neq;
                pzz = pzz + c.z * c.z * neq;
                pxy = pxy + c.x * c.y * neq;
                pxz = pxz + c.x * c.z * neq;
                pyz = pyz + c.y * c.z * neq;
            }
            let q = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
            tau = 0.5 * (params.tau0 + sqrt(params.tau0 * params.tau0 + 18.0 * params.cs2 * q / dens));
        }
        let omega_p = 1.0 / tau;
        let omega_m = 1.0 / (params.lambda / (tau - 0.5) + 0.5);
        for (var a = 0u; a < Q; a = a + 1u) {
            let o = opp(a);
            let sym = 0.5 * (pop[a] + pop[o]) - 0.5 * (eq[a] + eq[o]);
            let anti = 0.5 * (pop[a] - pop[o]) - 0.5 * (eq[a] - eq[o]);
            let c = vec3<f32>(c_of(a));
            let cu = dot(c, vel);
            let fterm = (c - vel) * 3.0 + cu * 9.0 * c;
            let forcing = (1.0 - 0.5 * omega_p) * w_of(a) * dot(fterm, force);
            post[a] = pop[a] - omega_p * sym - omega_m * anti + forcing;
        }
    }
    u[cell] = vec4<f32>(vel, dens);

    // ---- store ----
    if odd {
        for (var a = 0u; a < Q; a = a + 1u) {
            let c = c_of(a);
            let nb = neighbour(i, j, k, c);
            if nb.y == 0u {
                f[a * n + nb.x] = post[a];
            } else if nb.y == 1u {
                f[opp(a) * n + cell] = post[a];
            } else {
                f[opp(a) * n + cell] = feq(opp(a), 1.0, vel);
            }
        }
    } else {
        for (var a = 0u; a < Q; a = a + 1u) {
            f[opp(a) * n + cell] = post[a];
        }
    }
}

// Equilibrium at rest everywhere (even-step layout: direct slots).
@compute @workgroup_size(128)
fn init(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let cell = (wg.y * params.groups_x + wg.x) * WG + lid;
    if cell >= params.n { return; }
    for (var a = 0u; a < Q; a = a + 1u) {
        f[a * params.n + cell] = w_of(a);
    }
    u[cell] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
