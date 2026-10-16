//! What a solver backend computes. The grid, the voxels, the units and the boundary conditions
//! are prepared here in plain arrays; a backend (the GPU one in `solidcraft-sim-gpu`, or the
//! CPU reference below) steps the lattice and solves the linear system.

use crate::lattice::{C, OPP, Q, collide, equilibrium};
use crate::linsolve::{Stencil, bicgstab, conjugate_gradient};
use crate::{Progress, Result, SimError, par_chunks_mut};

/// Cell kinds in [`LatticeSetup::flags`] (low byte); a fan cell carries its fan index in the
/// bits above.
pub const CELL_SOLID: u32 = 0;
pub const CELL_AIR: u32 = 1;
pub const CELL_FAN: u32 = 2;
pub const FAN_SHIFT: u32 = 8;

/// A lattice-Boltzmann run, in lattice units, on the full grid.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeSetup {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Per grid cell (x fastest): [`CELL_SOLID`], [`CELL_AIR`] or [`CELL_FAN`] | fan << [`FAN_SHIFT`].
    pub flags: Vec<u32>,
    /// Bit `i` set: wall `i` (`-x, +x, -y, +y, -z, +z`) is open (ambient equilibrium flows in and
    /// out); clear: the wall bounces back.
    pub open_walls: u32,
    /// Fan velocities, lattice units.
    pub fans: Vec<[f32; 3]>,
    pub tau0: f32,
    /// Smagorinsky constant squared; 0 turns the eddy viscosity off.
    pub smagorinsky2: f32,
    /// TRT magic parameter.
    pub lambda: f32,
    /// Buoyant acceleration per kelvin above ambient, lattice units, along +z.
    pub g_lat: f32,
    /// Fan velocities ramp from 0 to full over this many steps.
    pub ramp_steps: u32,
    /// Kelvin above ambient per grid cell (buoyancy); `None` for no buoyancy.
    pub dtemp: Option<Vec<f32>>,
    /// Extra relaxation time per grid cell (a sponge layer beside open walls: the viscosity
    /// rises there so vortices are damped before they reach the boundary); `None` for none.
    pub sponge: Option<Vec<f32>>,
    pub max_steps: usize,
    /// Relative change of the velocity field between checks (every 50 steps) that counts as steady.
    pub tolerance: f64,
}

impl LatticeSetup {
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn air_cells(&self) -> usize {
        self.flags.iter().filter(|f| (**f & 0xff) != CELL_SOLID).count()
    }
    pub fn validate(&self) -> Result<()> {
        if self.flags.len() != self.len() || self.is_empty() {
            return Err(SimError::Invalid("lattice flags do not match the grid".into()));
        }
        if let Some(t) = &self.sponge
            && (t.len() != self.len() || t.iter().any(|v| !v.is_finite() || *v < 0.0))
        {
            return Err(SimError::Invalid("lattice sponge does not match the grid".into()));
        }
        if let Some(t) = &self.dtemp
            && t.len() != self.len()
        {
            return Err(SimError::Invalid("lattice temperatures do not match the grid".into()));
        }
        if !self.tau0.is_finite() || self.tau0 <= 0.5 {
            return Err(SimError::Invalid("lattice relaxation time must be above 1/2".into()));
        }
        Ok(())
    }
    /// The fan velocity of a cell, if it is a fan cell.
    #[inline]
    pub fn fan(&self, flag: u32) -> Option<[f32; 3]> {
        if flag & 0xff == CELL_FAN { self.fans.get((flag >> FAN_SHIFT) as usize).copied() } else { None }
    }
}

/// The velocity and density fields a lattice run ends with, lattice units, on the full grid.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowFields {
    pub u: Vec<[f32; 3]>,
    pub rho: Vec<f32>,
    pub steps: usize,
    pub converged: bool,
    pub change: f64,
    pub max_lattice_speed: f64,
}

/// The linear system of the temperature field: `A x = b` on the 7-point stencil.
pub struct ThermalSystem {
    pub stencil: Stencil,
    pub rhs: Vec<f64>,
    /// `A` is symmetric positive definite (conjugate gradients apply); else one-sided (BiCGSTAB).
    pub symmetric: bool,
    pub tolerance: f64,
    pub max_iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThermalSolution {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
    pub method: String,
}

/// A solver backend. Implementations must never panic on a hostile setup: validate and return
/// errors.
pub trait Accelerator: Send + Sync {
    /// What runs the solves, for reports (`gpu: NVIDIA ...`, `cpu`).
    fn name(&self) -> String;
    /// Step the lattice to a steady state.
    fn flow(&self, setup: &LatticeSetup, progress: Progress) -> Result<FlowFields>;
    /// Solve the temperature system.
    fn thermal(&self, system: &ThermalSystem, progress: Progress) -> Result<ThermalSolution>;
}

/// The reference implementation on the CPU: the oracle the GPU kernels are checked against in
/// tests, and what a build without a GPU backend can still run.
pub struct CpuAccelerator;

/// Where a pulled population comes from (CPU streaming).
#[derive(Clone, Copy)]
enum Source {
    Air(u32),
    Bounce,
    Open,
}

impl Accelerator for CpuAccelerator {
    fn name(&self) -> String {
        "cpu".into()
    }

    fn flow(&self, setup: &LatticeSetup, progress: Progress) -> Result<FlowFields> {
        setup.validate()?;
        let (nx, ny, nz) = (setup.nx, setup.ny, setup.nz);
        let n = setup.len();
        let mut compact = vec![u32::MAX; n];
        let mut cells: Vec<u32> = Vec::new();
        for (idx, f) in setup.flags.iter().enumerate() {
            if f & 0xff != CELL_SOLID {
                if cells.len() >= u32::MAX as usize - 1 {
                    return Err(SimError::TooLarge("too many air cells".into()));
                }
                compact[idx] = cells.len() as u32;
                cells.push(idx as u32);
            }
        }
        if cells.is_empty() {
            return Err(SimError::Invalid("no air in the domain: nothing to flow".into()));
        }
        let index = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let mut sources: Vec<[Source; Q]> = Vec::with_capacity(cells.len());
        let mut fan: Vec<Option<[f32; 3]>> = Vec::with_capacity(cells.len());
        for &idx in &cells {
            let idx = idx as usize;
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            let mut s = [Source::Bounce; Q];
            for a in 0..Q {
                let (ni, nj, nk) = (i as i64 - C[a][0] as i64, j as i64 - C[a][1] as i64, k as i64 - C[a][2] as i64);
                let outside = |v: i64, n: usize| v < 0 || v >= n as i64;
                if outside(ni, nx) || outside(nj, ny) || outside(nk, nz) {
                    let mut open = true;
                    let crossed = [ni < 0, ni >= nx as i64, nj < 0, nj >= ny as i64, nk < 0, nk >= nz as i64];
                    for (w, hit) in crossed.iter().enumerate() {
                        if *hit && setup.open_walls & (1 << w) == 0 {
                            open = false;
                        }
                    }
                    s[a] = if open { Source::Open } else { Source::Bounce };
                    continue;
                }
                let nidx = index(ni as usize, nj as usize, nk as usize);
                s[a] = match compact.get(nidx) {
                    Some(&c) if c != u32::MAX => Source::Air(c),
                    _ => Source::Bounce,
                };
            }
            sources.push(s);
            fan.push(setup.flags.get(idx).and_then(|f| setup.fan(*f)));
        }
        let m = cells.len();
        let mut f: Vec<[f32; Q]> = vec![equilibrium(1.0, [0.0; 3]); m];
        let mut f2 = f.clone();
        let mut u: Vec<[f32; 3]> = vec![[0.0; 3]; m];
        let mut rho: Vec<f32> = vec![1.0; m];
        let mut u_prev = u.clone();
        let dtemp: Vec<f32> = match &setup.dtemp {
            Some(t) => cells.iter().map(|&idx| t.get(idx as usize).copied().unwrap_or(0.0)).collect(),
            None => vec![0.0; m],
        };
        let sponge: Vec<f32> = match &setup.sponge {
            Some(t) => cells.iter().map(|&idx| t.get(idx as usize).copied().unwrap_or(0.0)).collect(),
            None => vec![0.0; m],
        };
        let (tau0, cs2, lambda, g_lat) = (setup.tau0, setup.smagorinsky2, setup.lambda, setup.g_lat);
        let mut steps = 0;
        let mut converged = false;
        let mut change = f64::INFINITY;
        let mut max_speed = 0.0f64;
        let chunk = (m / 64).max(4096);
        for step in 1..=setup.max_steps {
            let ramp = if setup.ramp_steps > 0 { (step as f32 / setup.ramp_steps as f32).min(1.0) } else { 1.0 };
            {
                let dtemp = &dtemp;
                let sponge = &sponge;
                let fan = &fan;
                let mut work: Vec<(&mut [[f32; Q]], &mut [[f32; 3]], &mut [f32])> =
                    f.chunks_mut(chunk).zip(u.chunks_mut(chunk)).zip(rho.chunks_mut(chunk)).map(|((a, b), c)| (a, b, c)).collect();
                par_chunks_mut(&mut work, 1, |ci, items| {
                    let base = ci * chunk;
                    for (fc, uc, rc) in items.iter_mut() {
                        for (r, pop) in fc.iter_mut().enumerate() {
                            let cell = base + r;
                            let force = [0.0, 0.0, g_lat * dtemp.get(cell).copied().unwrap_or(0.0)];
                            let fv = fan.get(cell).copied().flatten().map(|v| [v[0] * ramp, v[1] * ramp, v[2] * ramp]);
                            let tau = tau0 + sponge.get(cell).copied().unwrap_or(0.0);
                            let (vel, dens) = collide(pop, force, tau, cs2, lambda, fv);
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
            {
                let src = &f;
                let sources = &sources;
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
            if step % 50 == 0 || step == setup.max_steps {
                let (mut num, mut den, mut fastest) = (0.0f64, 0.0f64, 0.0f32);
                for (a, b) in u.iter().zip(&u_prev) {
                    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                    num += f64::from(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    let s = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
                    den += f64::from(s);
                    fastest = fastest.max(s);
                }
                max_speed = f64::from(fastest.sqrt());
                if std::env::var_os("SOLIDCRAFT_SIM_DEBUG").is_some() {
                    let (mut best, mut at) = (0.0f32, 0usize);
                    for (c, v) in u.iter().enumerate() {
                        let s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                        if s > best {
                            best = s;
                            at = c;
                        }
                    }
                    let idx = cells.get(at).copied().unwrap_or(0) as usize;
                    let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
                    let r = rho.get(at).copied().unwrap_or(0.0);
                    let rmin = rho.iter().copied().fold(f32::INFINITY, f32::min);
                    let rmax = rho.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!(
                        "step {step}: max |u| {:.4} at ({i},{j},{k}) rho {r:.4}; rho {rmin:.4}..{rmax:.4}; change {:.2e}",
                        best.sqrt(),
                        (num / den.max(1e-30)).sqrt()
                    );
                }
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
                if !progress("flow", step, setup.max_steps) {
                    return Err(SimError::Cancelled);
                }
                if step >= 400 && change < setup.tolerance {
                    converged = true;
                    break;
                }
            }
        }
        let mut uf = vec![[0.0f32; 3]; n];
        let mut rf = vec![1.0f32; n];
        for (c, &idx) in cells.iter().enumerate() {
            if let (Some(v), Some(r), Some(ov), Some(or)) = (u.get(c), rho.get(c), uf.get_mut(idx as usize), rf.get_mut(idx as usize)) {
                *ov = *v;
                *or = *r;
            }
        }
        Ok(FlowFields { u: uf, rho: rf, steps, converged, change, max_lattice_speed: max_speed })
    }

    fn thermal(&self, system: &ThermalSystem, progress: Progress) -> Result<ThermalSolution> {
        let n = system.stencil.len();
        if system.rhs.len() != n || n == 0 {
            return Err(SimError::Invalid("thermal system does not match its stencil".into()));
        }
        let mut x = vec![0.0; n];
        let total = system.max_iterations;
        let mut cancelled = false;
        let mut watch = |it: usize, _res: f64| -> bool {
            let go = progress("thermal", it, total);
            if !go {
                cancelled = true;
            }
            go
        };
        let report = if system.symmetric {
            conjugate_gradient(&system.stencil, &system.rhs, &mut x, system.tolerance, total, &mut watch)
        } else {
            bicgstab(&system.stencil, &system.rhs, &mut x, system.tolerance, total, &mut watch)
        };
        if cancelled {
            return Err(SimError::Cancelled);
        }
        Ok(ThermalSolution { x, iterations: report.iterations, residual: report.residual, converged: report.converged, method: report.method.into() })
    }
}
