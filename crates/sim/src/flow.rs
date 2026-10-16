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
use crate::grid::Grid;
use crate::problem::{Problem, Wall};
use crate::zones::{ZONE_AIR, ZONE_SOLID, fluid_of, zone_name};
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

/// One fluid zone's run.
#[derive(Clone, Debug, PartialEq)]
pub struct ZoneFlow {
    pub zone: u16,
    pub name: String,
    pub cells: usize,
    pub steps: usize,
    pub converged: bool,
    /// Last relative change of the velocity field.
    pub change: f64,
    /// Physical time step, s.
    pub dt: f64,
    /// Fastest lattice speed seen (stability needs it well below 0.3).
    pub max_lattice_speed: f64,
    /// The kinematic viscosity the zone was solved with, m²/s (molecular, or raised for the grid).
    pub effective_viscosity: f64,
    /// Million lattice updates per second.
    pub mlups: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowReport {
    /// m/s per grid cell (zero in solids).
    pub velocity: Vec<[f32; 3]>,
    /// Pa relative to ambient (or to the zone's mean), per grid cell.
    pub pressure: Vec<f32>,
    /// The air's run (zone 0); the fields below repeat it for convenience.
    pub steps: usize,
    pub converged: bool,
    pub change: f64,
    pub dt: f64,
    pub max_lattice_speed: f64,
    pub air_cells: usize,
    pub effective_viscosity: f64,
    /// What stepped the lattices.
    pub backend: String,
    /// Million lattice (air-cell) updates per second over the air's run.
    pub mlups: f64,
    /// Every zone that had cells, air first.
    pub zones: Vec<ZoneFlow>,
}

impl FlowReport {
    /// The viscosity a zone was solved with, if it was.
    pub fn zone_viscosity(&self, zone: u16) -> Option<f64> {
        self.zones.iter().find(|z| z.zone == zone).map(|z| z.effective_viscosity)
    }
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

/// The lattice of one fluid zone, in lattice units, with the unit conversion; `None` when the
/// zone has no cells.
pub fn setup(
    grid: &Grid,
    zones: &[u16],
    problem: &Problem,
    settings: &FlowSettings,
    temperature: Option<&[f64]>,
    zone: u16,
) -> Result<Option<(LatticeSetup, Units)>> {
    settings.validate()?;
    let n = grid.len();
    if zones.len() != n {
        return Err(SimError::Invalid("zones do not match the grid".into()));
    }
    let Some(fluid) = fluid_of(problem, zone) else { return Ok(None) };
    let dx = grid.dx_m();
    // Which fans drive cells of this zone, and the zone's extent (for the plume scale).
    let mut fan_hit = vec![false; problem.fans.len()];
    let mut kmin = usize::MAX;
    let mut kmax = 0usize;
    let mut cells = 0usize;
    let mut fan_of = vec![u32::MAX; n];
    for (idx, slot) in fan_of.iter_mut().enumerate() {
        if zones.get(idx) != Some(&zone) {
            continue;
        }
        cells += 1;
        let c = grid.coords(idx);
        kmin = kmin.min(c[2]);
        kmax = kmax.max(c[2]);
        let centre = grid.centre(c);
        for (fi, f) in problem.fans.iter().enumerate() {
            let r = &f.region;
            if centre.x >= r.min.x && centre.x <= r.max.x && centre.y >= r.min.y && centre.y <= r.max.y && centre.z >= r.min.z && centre.z <= r.max.z
            {
                *slot = fi as u32;
                if let Some(h) = fan_hit.get_mut(fi) {
                    *h = true;
                }
                break;
            }
        }
    }
    if cells == 0 {
        return Ok(None);
    }
    let fan_max = problem.fans.iter().zip(&fan_hit).filter(|(_, h)| **h).map(|(f, _)| f.velocity.len()).fold(0.0, f64::max);
    let height = ((kmax + 1 - kmin) as f64) * dx;
    let plume = (9.81 * fluid.thermal_expansion * 30.0 * height.max(0.01)).sqrt();
    let u_ref = if fan_max > 0.0 { fan_max.max(plume * 0.5) } else { plume.max(0.05) };
    let dt = settings.lattice_velocity * dx / u_ref;
    let u_scale = dt / dx;
    let nu_lat = fluid.kinematic_viscosity * dt / (dx * dx);
    // Eddy-viscosity floor: the grid Reynolds number u·dx/ν stays below the limit.
    let nu_floor = settings.lattice_velocity / settings.max_cell_reynolds;
    let nu_used = nu_lat.max(nu_floor);
    let effective_viscosity = nu_used * dx * dx / dt;
    let tau0 = (3.0 * nu_used + 0.5) as f32;
    let g_lat = (9.81 * fluid.thermal_expansion * dt * dt / dx) as f32;
    let p_scale = fluid.density * (dx / dt) * (dx / dt) / 3.0;
    let fans: Vec<[f32; 3]> = problem
        .fans
        .iter()
        .map(|f| {
            let v = f.velocity * u_scale;
            [v.x as f32, v.y as f32, v.z as f32]
        })
        .collect();
    let flags: Vec<u32> = (0..n)
        .map(|idx| {
            if zones.get(idx) != Some(&zone) {
                CELL_SOLID
            } else if let Some(&fi) = fan_of.get(idx).filter(|f| **f != u32::MAX) {
                CELL_FAN | (fi << FAN_SHIFT)
            } else {
                CELL_AIR
            }
        })
        .collect();
    // Only the air meets the domain walls; a coolant is held in its body.
    let mut open_walls = 0u32;
    if zone == ZONE_AIR {
        for (i, w) in problem.walls.iter().enumerate() {
            if *w == Wall::Open {
                open_walls |= 1 << i;
            }
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
    Ok(Some((lattice, Units { u_scale, dt, effective_viscosity, p_scale })))
}

pub(crate) fn solve(
    grid: &Grid,
    zones: &[u16],
    problem: &Problem,
    settings: &FlowSettings,
    temperature: Option<&[f64]>,
    backend: &dyn Accelerator,
    progress: Progress,
) -> Result<FlowReport> {
    let n = grid.len();
    let mut velocity = vec![[0.0f32; 3]; n];
    let mut pressure = vec![0.0f32; n];
    let mut runs = Vec::new();
    let zone_count = problem.fluids.len() + 1;
    for zone in 0..zone_count {
        let zone = zone as u16;
        let Some((lattice, units)) = setup(grid, zones, problem, settings, temperature, zone)? else { continue };
        let cells = lattice.air_cells();
        let started = web_time::Instant::now();
        let fields = backend.flow(&lattice, progress)?;
        let seconds = started.elapsed().as_secs_f64().max(1e-9);
        if fields.u.len() != n || fields.rho.len() != n {
            return Err(SimError::Internal(format!("{}: flow fields do not match the grid", backend.name())));
        }
        let inv = (1.0 / units.u_scale) as f32;
        let p_scale = units.p_scale as f32;
        for idx in 0..n {
            if zones.get(idx) != Some(&zone) {
                continue;
            }
            if let (Some(v), Some(r), Some(ov), Some(op)) = (fields.u.get(idx), fields.rho.get(idx), velocity.get_mut(idx), pressure.get_mut(idx)) {
                *ov = [v[0] * inv, v[1] * inv, v[2] * inv];
                *op = (r - 1.0) * p_scale;
            }
        }
        runs.push(ZoneFlow {
            zone,
            name: zone_name(zone),
            cells,
            steps: fields.steps,
            converged: fields.converged,
            change: fields.change,
            dt: units.dt,
            max_lattice_speed: fields.max_lattice_speed,
            effective_viscosity: units.effective_viscosity,
            mlups: (cells as f64) * (fields.steps as f64) / seconds / 1e6,
        });
    }
    if runs.is_empty() {
        return Err(SimError::Invalid("no fluid in the domain: nothing to flow".into()));
    }
    let air = runs.iter().find(|z| z.zone == ZONE_AIR).cloned().unwrap_or_else(|| ZoneFlow {
        zone: ZONE_SOLID,
        name: "none".into(),
        cells: 0,
        steps: 0,
        converged: true,
        change: 0.0,
        dt: 0.0,
        max_lattice_speed: 0.0,
        effective_viscosity: problem.air.kinematic_viscosity,
        mlups: 0.0,
    });
    Ok(FlowReport {
        velocity,
        pressure,
        steps: air.steps,
        converged: runs.iter().all(|z| z.converged),
        change: air.change,
        dt: air.dt,
        max_lattice_speed: runs.iter().map(|z| z.max_lattice_speed).fold(0.0, f64::max),
        air_cells: air.cells,
        effective_viscosity: air.effective_viscosity,
        backend: backend.name(),
        mlups: air.mlups,
        zones: runs,
    })
}
