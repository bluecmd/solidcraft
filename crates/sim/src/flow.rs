//! Airflow by the lattice-Boltzmann method (D3Q19, BGK collision with a Smagorinsky eddy
//! viscosity so coarse grids at realistic air viscosities stay stable).
//!
//! Air cells carry 19 populations; solid cells and closed walls bounce back; open walls supply
//! equilibrium populations at ambient density with the boundary cell's velocity (a pressure
//! outlet that also lets air in). Fans impose a velocity on the air cells in their region.
//! Buoyancy (Boussinesq, gravity along -z) acts through Guo forcing from a temperature field.
//! Time is stepped until the velocity field stops changing or a step cap is reached: the result
//! is a steady (or quasi-steady, time-averaged) flow, which is what a thermal study needs.

use serde::{Deserialize, Serialize};

use crate::grid::{FLUID, Grid};
use crate::problem::{Problem, Wall};
use crate::voxel::Voxels;
use crate::{Progress, Result, SimError, finite, par_chunks_mut};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlowSettings {
    /// Lattice time steps before giving up on a steady state.
    pub max_steps: usize,
    /// Relative change of the velocity field (over 50 steps) below which the flow is steady.
    pub tolerance: f64,
    /// Smagorinsky constant; 0 turns the sub-grid model off (fine grids, slow flow only).
    pub smagorinsky: f64,
    /// Let heated air rise (needs the thermal field: the two solvers alternate).
    pub buoyancy: bool,
    /// Flow/thermal alternations when buoyancy is on.
    pub coupling_iterations: usize,
    /// Target lattice speed for the fastest expected flow (sets the time step); below 0.1 for accuracy.
    pub lattice_velocity: f64,
    /// The viscosity is raised so that `u·dx/ν` stays below this: on a grid too coarse to resolve
    /// turbulence the extra viscosity stands for its mixing (and the same eddy diffusivity carries
    /// heat in the air). Larger is more faithful at fine resolution but less stable.
    pub max_cell_reynolds: f64,
    /// Two-relaxation-time "magic" parameter Λ. 1/12 cancels the third-order spatial error and
    /// keeps inertial flows past obstacles steady; 1/4 (best for creeping flow through porous
    /// media) lets a shear layer at a fan exit grow until the solve blows up.
    pub trt_lambda: f64,
}

impl Default for FlowSettings {
    fn default() -> Self {
        FlowSettings {
            max_steps: 4000,
            tolerance: 1e-3,
            smagorinsky: 0.12,
            buoyancy: true,
            coupling_iterations: 2,
            lattice_velocity: 0.05,
            max_cell_reynolds: 10.0,
            trt_lambda: 1.0 / 12.0,
        }
    }
}

impl FlowSettings {
    pub fn validate(&self) -> Result<()> {
        if self.max_steps == 0 || self.max_steps > 10_000_000 {
            return Err(SimError::Invalid("flow max_steps must be in 1..=10000000".into()));
        }
        finite(self.tolerance, 1e-12, 1.0, "flow tolerance")?;
        finite(self.smagorinsky, 0.0, 1.0, "smagorinsky constant")?;
        if self.coupling_iterations == 0 || self.coupling_iterations > 100 {
            return Err(SimError::Invalid("coupling_iterations must be in 1..=100".into()));
        }
        finite(self.lattice_velocity, 1e-4, 0.2, "lattice velocity")?;
        finite(self.max_cell_reynolds, 0.1, 1e6, "max cell Reynolds number")?;
        finite(self.trt_lambda, 1e-4, 10.0, "TRT lambda")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowReport {
    /// m/s per grid cell (zero in solids).
    pub velocity: Vec<[f32; 3]>,
    /// Pa relative to ambient, per grid cell.
    pub pressure: Vec<f32>,
    pub steps: usize,
    pub converged: bool,
    /// Last relative change of the velocity field.
    pub change: f64,
    /// Physical time step, s.
    pub dt: f64,
    /// Fastest lattice speed seen (stability needs it well below 0.3).
    pub max_lattice_speed: f64,
    pub air_cells: usize,
    /// The kinematic viscosity the air was solved with, m²/s (molecular, or raised for the grid).
    pub effective_viscosity: f64,
}

const Q: usize = 19;
const C: [[i32; 3]; Q] = [
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
const OPP: [usize; Q] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];
const W: [f32; Q] = [
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
fn equilibrium(rho: f32, u: [f32; 3]) -> [f32; Q] {
    let usq = 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    let mut f = [0.0; Q];
    for a in 0..Q {
        let cu = C[a][0] as f32 * u[0] + C[a][1] as f32 * u[1] + C[a][2] as f32 * u[2];
        f[a] = W[a] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
    }
    f
}

/// Where a pulled population comes from.
#[derive(Clone, Copy)]
enum Source {
    /// Another air cell (compact index).
    Air(u32),
    /// A solid cell or a closed wall: bounce back.
    Bounce,
    /// Past an open wall: ambient equilibrium.
    Open,
}

/// The lattice: air cells only, with the per-direction neighbour table resolved once.
struct Lattice {
    /// Grid index per air cell.
    cells: Vec<u32>,
    /// `Q` sources per air cell.
    sources: Vec<[Source; Q]>,
    /// Fan velocity (lattice units) per air cell, if inside a fan region.
    fan: Vec<Option<[f32; 3]>>,
}

fn build_lattice(grid: &Grid, voxels: &Voxels, problem: &Problem, u_scale: f32) -> Result<Lattice> {
    let n = grid.len();
    let mut compact = vec![u32::MAX; n];
    let mut cells = Vec::new();
    for (idx, slot) in compact.iter_mut().enumerate() {
        if voxels.label.get(idx) == Some(&FLUID) {
            if cells.len() >= u32::MAX as usize - 1 {
                return Err(SimError::TooLarge("too many air cells".into()));
            }
            *slot = cells.len() as u32;
            cells.push(idx as u32);
        }
    }
    if cells.is_empty() {
        return Err(SimError::Invalid("no air in the domain: nothing to flow".into()));
    }
    let wall_of = |side: usize| problem.walls.get(side).copied().unwrap_or_default();
    let mut sources = Vec::with_capacity(cells.len());
    let mut fan = Vec::with_capacity(cells.len());
    for &idx in &cells {
        let [i, j, k] = grid.coords(idx as usize);
        let mut s = [Source::Bounce; Q];
        for a in 0..Q {
            // pull from the cell at -c_a
            let (di, dj, dk) = (-C[a][0], -C[a][1], -C[a][2]);
            let (ni, nj, nk) = (i as i64 + di as i64, j as i64 + dj as i64, k as i64 + dk as i64);
            let outside = |v: i64, n: usize| v < 0 || v >= n as i64;
            if outside(ni, grid.nx) || outside(nj, grid.ny) || outside(nk, grid.nz) {
                // which wall(s) are crossed; closed wins over open
                let mut open = true;
                if ni < 0 && wall_of(0) == Wall::Closed {
                    open = false;
                }
                if ni >= grid.nx as i64 && wall_of(1) == Wall::Closed {
                    open = false;
                }
                if nj < 0 && wall_of(2) == Wall::Closed {
                    open = false;
                }
                if nj >= grid.ny as i64 && wall_of(3) == Wall::Closed {
                    open = false;
                }
                if nk < 0 && wall_of(4) == Wall::Closed {
                    open = false;
                }
                if nk >= grid.nz as i64 && wall_of(5) == Wall::Closed {
                    open = false;
                }
                s[a] = if open { Source::Open } else { Source::Bounce };
                continue;
            }
            let nidx = grid.index([ni as usize, nj as usize, nk as usize]);
            s[a] = match compact.get(nidx) {
                Some(&c) if c != u32::MAX => Source::Air(c),
                _ => Source::Bounce,
            };
        }
        sources.push(s);
        let centre = grid.centre([i, j, k]);
        let mut f = None;
        for fan_def in &problem.fans {
            let r = &fan_def.region;
            if centre.x >= r.min.x && centre.x <= r.max.x && centre.y >= r.min.y && centre.y <= r.max.y && centre.z >= r.min.z && centre.z <= r.max.z
            {
                let v = fan_def.velocity * f64::from(u_scale);
                f = Some([v.x as f32, v.y as f32, v.z as f32]);
                break;
            }
        }
        fan.push(f);
    }
    Ok(Lattice { cells, sources, fan })
}

pub(crate) fn solve(
    grid: &Grid,
    voxels: &Voxels,
    problem: &Problem,
    settings: &FlowSettings,
    temperature: Option<&[f64]>,
    progress: Progress,
) -> Result<FlowReport> {
    let dx = grid.dx_m();
    // Fastest physical speed to expect: fans, or a buoyant plume scale.
    let fan_max = problem.fans.iter().map(|f| f.velocity.len()).fold(0.0, f64::max);
    let height = (grid.nz as f64) * dx;
    let plume = (9.81 * problem.air.thermal_expansion * 30.0 * height.max(0.01)).sqrt();
    let u_ref = if fan_max > 0.0 { fan_max.max(plume * 0.5) } else { plume.max(0.05) };
    let dt = settings.lattice_velocity * dx / u_ref;
    let u_scale = (dt / dx) as f32; // physical m/s → lattice
    let nu_lat = problem.air.kinematic_viscosity * dt / (dx * dx);
    // Eddy-viscosity floor: the grid Reynolds number u·dx/ν stays below the limit.
    let nu_floor = settings.lattice_velocity / settings.max_cell_reynolds;
    let nu_used = nu_lat.max(nu_floor);
    let effective_viscosity = nu_used * dx * dx / dt;
    let tau0 = (3.0 * nu_used + 0.5) as f32;
    // Two-relaxation-time collision: the odd moments relax with their own rate, set by the
    // "magic" Λ, which keeps the scheme stable near τ = 1/2.
    let lambda = settings.trt_lambda as f32;
    let cs2 = settings.smagorinsky as f32 * settings.smagorinsky as f32;
    // buoyancy force (lattice units) per kelvin above ambient
    let g_lat = (9.81 * problem.air.thermal_expansion * dt * dt / dx) as f32;

    let lattice = build_lattice(grid, voxels, problem, u_scale)?;
    let m = lattice.cells.len();
    let mut f: Vec<[f32; Q]> = vec![equilibrium(1.0, [0.0; 3]); m];
    let mut f2 = f.clone();
    let mut u: Vec<[f32; 3]> = vec![[0.0; 3]; m];
    let mut rho: Vec<f32> = vec![1.0; m];
    let mut u_prev = u.clone();
    // Temperature rise per air cell, for buoyancy.
    let dtemp: Vec<f32> = if settings.buoyancy {
        lattice
            .cells
            .iter()
            .map(|&idx| temperature.and_then(|t| t.get(idx as usize)).map_or(0.0, |t| (*t - problem.thermal.ambient_c) as f32))
            .collect()
    } else {
        vec![0.0; m]
    };

    let mut steps = 0;
    let mut converged = false;
    let mut change = f64::INFINITY;
    let mut max_speed = 0.0f64;
    let chunk = (m / 64).max(4096);
    for step in 1..=settings.max_steps {
        let ramp = (step as f32 / 200.0).min(1.0);
        // collision (in place), with macroscopic update
        {
            let dtemp = &dtemp;
            let fan = &lattice.fan;
            let u_all = &mut u;
            let rho_all = &mut rho;
            // split borrows: collide per chunk writing f, u, rho for the same cells
            let f_chunks = f.chunks_mut(chunk);
            let u_chunks = u_all.chunks_mut(chunk);
            let r_chunks = rho_all.chunks_mut(chunk);
            let mut work: Vec<(&mut [[f32; Q]], &mut [[f32; 3]], &mut [f32])> =
                f_chunks.zip(u_chunks).zip(r_chunks).map(|((a, b), c)| (a, b, c)).collect();
            par_chunks_mut(&mut work, 1, |ci, items| {
                let base = ci * chunk;
                for (fc, uc, rc) in items.iter_mut() {
                    for (r, pop) in fc.iter_mut().enumerate() {
                        let cell = base + r;
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
                        let force = [0.0, 0.0, g_lat * dtemp.get(cell).copied().unwrap_or(0.0)];
                        let mut vel = [(mom[0] + 0.5 * force[0]) / dens, (mom[1] + 0.5 * force[1]) / dens, (mom[2] + 0.5 * force[2]) / dens];
                        if let Some(Some(fv)) = fan.get(cell) {
                            // A fan cell is held at its velocity: its populations are the
                            // equilibrium (ramped in over the first steps so the start is quiet).
                            vel = [fv[0] * ramp, fv[1] * ramp, fv[2] * ramp];
                            *pop = equilibrium(dens, vel);
                            if let Some(o) = uc.get_mut(r) {
                                *o = vel;
                            }
                            if let Some(o) = rc.get_mut(r) {
                                *o = dens;
                            }
                            continue;
                        }
                        let feq = equilibrium(dens, vel);
                        // Smagorinsky: local relaxation time from the non-equilibrium stress
                        let mut tau = tau0;
                        if cs2 > 0.0 {
                            let mut pi = [0.0f32; 6]; // xx, yy, zz, xy, xz, yz
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
                            let q = (pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2] + 2.0 * (pi[3] * pi[3] + pi[4] * pi[4] + pi[5] * pi[5])).sqrt();
                            tau = 0.5 * (tau0 + (tau0 * tau0 + 18.0 * cs2 * q / dens).sqrt());
                        }
                        let omega_p = 1.0 / tau;
                        let omega_m = 1.0 / (lambda / (tau - 0.5) + 0.5);
                        let old = *pop;
                        for a in 0..Q {
                            let o = OPP[a];
                            let sym = 0.5 * (old[a] + old[o]) - 0.5 * (feq[a] + feq[o]);
                            let anti = 0.5 * (old[a] - old[o]) - 0.5 * (feq[a] - feq[o]);
                            let (cx, cy, cz) = (C[a][0] as f32, C[a][1] as f32, C[a][2] as f32);
                            let cu = cx * vel[0] + cy * vel[1] + cz * vel[2];
                            // Guo forcing term
                            let fx = (cx - vel[0]) * 3.0 + cu * 3.0 * 3.0 * cx;
                            let fy = (cy - vel[1]) * 3.0 + cu * 3.0 * 3.0 * cy;
                            let fz = (cz - vel[2]) * 3.0 + cu * 3.0 * 3.0 * cz;
                            let forcing = (1.0 - 0.5 * omega_p) * W[a] * (fx * force[0] + fy * force[1] + fz * force[2]);
                            pop[a] = old[a] - omega_p * sym - omega_m * anti + forcing;
                        }
                        if let Some(o) = uc.get_mut(r) {
                            *o = vel;
                        }
                        if let Some(o) = rc.get_mut(r) {
                            *o = dens;
                        }
                    }
                }
            });
        }
        // streaming (pull) into f2
        {
            let src = &f;
            let sources = &lattice.sources;
            let u_now = &u;
            par_chunks_mut(&mut f2, chunk, |ci, out| {
                let base = ci * chunk;
                for (r, dst) in out.iter_mut().enumerate() {
                    let cell = base + r;
                    let (Some(s), Some(mine)) = (sources.get(cell), src.get(cell)) else { continue };
                    let open_eq = u_now.get(cell).map(|v| equilibrium(1.0, *v));
                    for a in 0..Q {
                        dst[a] = match s[a] {
                            Source::Air(c) => src.get(c as usize).map_or(mine[a], |p| p[a]),
                            Source::Bounce => mine[OPP[a]],
                            Source::Open => open_eq.map_or(mine[a], |e| e[a]),
                        };
                    }
                }
            });
        }
        std::mem::swap(&mut f, &mut f2);
        steps = step;
        if step % 50 == 0 || step == settings.max_steps {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            let mut fastest = 0.0f32;
            for (a, b) in u.iter().zip(&u_prev) {
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                num += f64::from(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                let s = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
                den += f64::from(s);
                fastest = fastest.max(s);
            }
            max_speed = f64::from(fastest.sqrt());
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
            u_prev.copy_from_slice(&u);
            if !progress("flow", step, settings.max_steps) {
                return Err(SimError::Cancelled);
            }
            if step >= 400 && change < settings.tolerance {
                converged = true;
                break;
            }
        }
    }
    // back to physical units, on the full grid
    let n = grid.len();
    let mut velocity = vec![[0.0f32; 3]; n];
    let mut pressure = vec![0.0f32; n];
    let p_scale = (problem.air.density * (dx / dt) * (dx / dt) / 3.0) as f32;
    for (c, &idx) in lattice.cells.iter().enumerate() {
        if let (Some(v), Some(r), Some(ov), Some(op)) = (u.get(c), rho.get(c), velocity.get_mut(idx as usize), pressure.get_mut(idx as usize)) {
            *ov = [v[0] / u_scale, v[1] / u_scale, v[2] / u_scale];
            *op = (r - 1.0) * p_scale;
        }
    }
    Ok(FlowReport { velocity, pressure, steps, converged, change, dt, max_lattice_speed: max_speed, air_cells: m, effective_viscosity })
}
