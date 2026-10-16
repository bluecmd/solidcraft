//! Airflow by the lattice-Boltzmann method (D3Q19, two-relaxation-time collision with a
//! Smagorinsky eddy viscosity so coarse grids at realistic air viscosities stay stable).
//!
//! Air cells carry 19 populations; solid cells and closed walls bounce back; open walls supply
//! equilibrium populations at ambient density with the boundary cell's velocity (a pressure
//! outlet that also lets air in). Fans hold the air cells in their region at a velocity.
//! Buoyancy (Boussinesq, gravity along -z) acts through Guo forcing from a temperature field.
//! Time is stepped until the velocity field stops changing or a step cap is reached: the result
//! is a steady (or quasi-steady, time-averaged) flow, which is what a thermal study needs.
//!
//! This module prepares the lattice in lattice units ([`LatticeSetup`]) and converts the fields
//! back to physical units; a backend ([`Accelerator`]) does the stepping.

use serde::{Deserialize, Serialize};

use crate::backend::{Accelerator, CELL_AIR, CELL_FAN, CELL_SOLID, FAN_SHIFT, LatticeSetup};
use crate::grid::{FLUID, Grid};
use crate::problem::{Problem, Wall};
use crate::voxel::Voxels;
use crate::{Progress, Result, SimError, finite};

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
    /// What stepped the lattice.
    pub backend: String,
}

/// Lattice ↔ physical unit conversion of one run.
#[derive(Clone, Copy, Debug)]
pub struct Units {
    /// Physical m/s → lattice.
    pub u_scale: f64,
    pub dt: f64,
    pub effective_viscosity: f64,
    /// Lattice density excess → Pa.
    pub p_scale: f64,
}

/// The lattice of a problem, in lattice units, with the unit conversion.
pub fn setup(grid: &Grid, voxels: &Voxels, problem: &Problem, settings: &FlowSettings, temperature: Option<&[f64]>) -> Result<(LatticeSetup, Units)> {
    settings.validate()?;
    let dx = grid.dx_m();
    // Fastest physical speed to expect: fans, or a buoyant plume scale.
    let fan_max = problem.fans.iter().map(|f| f.velocity.len()).fold(0.0, f64::max);
    let height = (grid.nz as f64) * dx;
    let plume = (9.81 * problem.air.thermal_expansion * 30.0 * height.max(0.01)).sqrt();
    let u_ref = if fan_max > 0.0 { fan_max.max(plume * 0.5) } else { plume.max(0.05) };
    let dt = settings.lattice_velocity * dx / u_ref;
    let u_scale = dt / dx;
    let nu_lat = problem.air.kinematic_viscosity * dt / (dx * dx);
    // Eddy-viscosity floor: the grid Reynolds number u·dx/ν stays below the limit.
    let nu_floor = settings.lattice_velocity / settings.max_cell_reynolds;
    let nu_used = nu_lat.max(nu_floor);
    let effective_viscosity = nu_used * dx * dx / dt;
    let tau0 = (3.0 * nu_used + 0.5) as f32;
    let g_lat = (9.81 * problem.air.thermal_expansion * dt * dt / dx) as f32;
    let p_scale = problem.air.density * (dx / dt) * (dx / dt) / 3.0;

    let n = grid.len();
    if voxels.label.len() != n {
        return Err(SimError::Invalid("voxel labels do not match the grid".into()));
    }
    let mut flags = vec![CELL_SOLID; n];
    let mut fans: Vec<[f32; 3]> = Vec::with_capacity(problem.fans.len());
    for f in &problem.fans {
        let v = f.velocity * u_scale;
        fans.push([v.x as f32, v.y as f32, v.z as f32]);
    }
    let mut air = 0usize;
    for (idx, flag) in flags.iter_mut().enumerate() {
        if voxels.label.get(idx) != Some(&FLUID) {
            continue;
        }
        air += 1;
        let centre = grid.centre(grid.coords(idx));
        *flag = CELL_AIR;
        for (fi, f) in problem.fans.iter().enumerate() {
            let r = &f.region;
            if centre.x >= r.min.x && centre.x <= r.max.x && centre.y >= r.min.y && centre.y <= r.max.y && centre.z >= r.min.z && centre.z <= r.max.z
            {
                *flag = CELL_FAN | ((fi as u32) << FAN_SHIFT);
                break;
            }
        }
    }
    if air == 0 {
        return Err(SimError::Invalid("no air in the domain: nothing to flow".into()));
    }
    let mut open_walls = 0u32;
    for (i, w) in problem.walls.iter().enumerate() {
        if *w == Wall::Open {
            open_walls |= 1 << i;
        }
    }
    let dtemp = if settings.buoyancy {
        temperature.map(|t| (0..n).map(|idx| t.get(idx).map_or(0.0, |t| (*t - problem.thermal.ambient_c) as f32)).collect())
    } else {
        None
    };
    let lattice = LatticeSetup {
        nx: grid.nx,
        ny: grid.ny,
        nz: grid.nz,
        flags,
        open_walls,
        fans,
        tau0,
        smagorinsky2: (settings.smagorinsky * settings.smagorinsky) as f32,
        lambda: settings.trt_lambda as f32,
        g_lat,
        ramp_steps: 200,
        dtemp,
        max_steps: settings.max_steps,
        tolerance: settings.tolerance,
    };
    Ok((lattice, Units { u_scale, dt, effective_viscosity, p_scale }))
}

pub(crate) fn solve(
    grid: &Grid,
    voxels: &Voxels,
    problem: &Problem,
    settings: &FlowSettings,
    temperature: Option<&[f64]>,
    backend: &dyn Accelerator,
    progress: Progress,
) -> Result<FlowReport> {
    let (lattice, units) = setup(grid, voxels, problem, settings, temperature)?;
    let air_cells = lattice.air_cells();
    let fields = backend.flow(&lattice, progress)?;
    let n = grid.len();
    if fields.u.len() != n || fields.rho.len() != n {
        return Err(SimError::Internal(format!("{}: flow fields do not match the grid", backend.name())));
    }
    let inv = (1.0 / units.u_scale) as f32;
    let p_scale = units.p_scale as f32;
    let velocity = fields.u.iter().map(|v| [v[0] * inv, v[1] * inv, v[2] * inv]).collect();
    let pressure = fields.rho.iter().map(|r| (r - 1.0) * p_scale).collect();
    Ok(FlowReport {
        velocity,
        pressure,
        steps: fields.steps,
        converged: fields.converged,
        change: fields.change,
        dt: units.dt,
        max_lattice_speed: fields.max_lattice_speed,
        air_cells,
        effective_viscosity: units.effective_viscosity,
        backend: backend.name(),
    })
}
