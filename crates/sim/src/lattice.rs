//! The D3Q19 lattice shared by every backend: velocities, weights, opposites, equilibrium.

pub const Q: usize = 19;
pub const C: [[i32; 3]; Q] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
];
pub const OPP: [usize; Q] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];
pub const W: [f32; Q] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

#[inline]
pub fn equilibrium(rho: f32, u: [f32; 3]) -> [f32; Q] {
    let usq = 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    let mut f = [0.0; Q];
    for a in 0..Q {
        let cu = C[a][0] as f32 * u[0] + C[a][1] as f32 * u[1] + C[a][2] as f32 * u[2];
        f[a] = W[a] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
    }
    f
}

/// Collision with a Smagorinsky eddy viscosity and Guo forcing, on one cell: `pop` in,
/// post-collision `pop` out; returns the velocity and density used.
///
/// `lambda > 0`: two-relaxation-time (TRT) with that magic parameter. `lambda == 0`: regularized
/// BGK (Latt & Chopard 2006): the non-equilibrium part is replaced by its projection on the
/// second-order Hermite moment before relaxing, which removes the ghost modes that grow where the
/// local cell Reynolds number is high (jets out of fans, flow through fin gaps).
#[inline]
pub fn collide(pop: &mut [f32; Q], force: [f32; 3], tau0: f32, cs2: f32, lambda: f32, fan: Option<[f32; 3]>) -> ([f32; 3], f32) {
    let mut dens = 0.0f32;
    let mut mom = [0.0f32; 3];
    for a in 0..Q {
        dens += pop[a];
        mom[0] += C[a][0] as f32 * pop[a];
        mom[1] += C[a][1] as f32 * pop[a];
        mom[2] += C[a][2] as f32 * pop[a];
    }
    if !(dens > 0.0) {
        dens = 1.0;
    }
    let mut vel = [(mom[0] + 0.5 * force[0]) / dens, (mom[1] + 0.5 * force[1]) / dens, (mom[2] + 0.5 * force[2]) / dens];
    if let Some(fv) = fan {
        // A fan cell is held at its velocity: its populations are the equilibrium.
        vel = fv;
        *pop = equilibrium(dens, vel);
        return (vel, dens);
    }
    let feq = equilibrium(dens, vel);
    let mut tau = tau0;
    let mut pi = [0.0f32; 6]; // xx, yy, zz, xy, xz, yz of the non-equilibrium stress
    for a in 0..Q {
        let neq = pop[a] - feq[a];
        let (cx, cy, cz) = (C[a][0] as f32, C[a][1] as f32, C[a][2] as f32);
        pi[0] += cx * cx * neq;
        pi[1] += cy * cy * neq;
        pi[2] += cz * cz * neq;
        pi[3] += cx * cy * neq;
        pi[4] += cx * cz * neq;
        pi[5] += cy * cz * neq;
    }
    if cs2 > 0.0 {
        let q = (pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2] + 2.0 * (pi[3] * pi[3] + pi[4] * pi[4] + pi[5] * pi[5])).sqrt();
        tau = 0.5 * (tau0 + (tau0 * tau0 + 18.0 * cs2 * q / dens).sqrt());
    }
    let omega_p = 1.0 / tau;
    if lambda <= 0.0 {
        // regularized: f = feq + (1 - ω) f_neq(Π) + forcing, f_neq(Π)_a = w_a 9/2 (c_a c_a - I/3) : Π
        let trace = (pi[0] + pi[1] + pi[2]) / 3.0;
        for a in 0..Q {
            let (cx, cy, cz) = (C[a][0] as f32, C[a][1] as f32, C[a][2] as f32);
            let qpi = cx * cx * pi[0] + cy * cy * pi[1] + cz * cz * pi[2] + 2.0 * (cx * cy * pi[3] + cx * cz * pi[4] + cy * cz * pi[5]) - trace;
            let fneq = W[a] * 4.5 * qpi;
            let cu = cx * vel[0] + cy * vel[1] + cz * vel[2];
            let fx = (cx - vel[0]) * 3.0 + cu * 9.0 * cx;
            let fy = (cy - vel[1]) * 3.0 + cu * 9.0 * cy;
            let fz = (cz - vel[2]) * 3.0 + cu * 9.0 * cz;
            let forcing = (1.0 - 0.5 * omega_p) * W[a] * (fx * force[0] + fy * force[1] + fz * force[2]);
            pop[a] = feq[a] + (1.0 - omega_p) * fneq + forcing;
        }
        return (vel, dens);
    }
    let omega_m = 1.0 / (lambda / (tau - 0.5) + 0.5);
    let old = *pop;
    for a in 0..Q {
        let o = OPP[a];
        let sym = 0.5 * (old[a] + old[o]) - 0.5 * (feq[a] + feq[o]);
        let anti = 0.5 * (old[a] - old[o]) - 0.5 * (feq[a] - feq[o]);
        let (cx, cy, cz) = (C[a][0] as f32, C[a][1] as f32, C[a][2] as f32);
        let cu = cx * vel[0] + cy * vel[1] + cz * vel[2];
        let fx = (cx - vel[0]) * 3.0 + cu * 9.0 * cx;
        let fy = (cy - vel[1]) * 3.0 + cu * 9.0 * cy;
        let fz = (cz - vel[2]) * 3.0 + cu * 9.0 * cz;
        let forcing = (1.0 - 0.5 * omega_p) * W[a] * (fx * force[0] + fy * force[1] + fz * force[2]);
        pop[a] = old[a] - omega_p * sym - omega_m * anti + forcing;
    }
    (vel, dens)
}
