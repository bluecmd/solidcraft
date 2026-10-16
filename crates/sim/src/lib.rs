//! Simulation for enclosures and electronics: how hot things get and where the air goes.
//!
//! Everything works on one uniform voxel grid over a domain box. Solid bodies (triangle meshes)
//! are voxelised; the rest of the grid is air. The thermal model is conjugate conduction on the
//! grid (a 7-point finite-volume stencil): solids conduct with their material's conductivity, air
//! conducts with an effective "mixing" conductivity, and a film coefficient stands in for the
//! boundary layer at every solid/air face. Heat sources are powers per body. Open domain walls sit
//! at ambient temperature. The optional airflow model is a D3Q19 lattice-Boltzmann solver on the
//! air cells with fans as velocity regions; its velocity field advects heat in the air (upwind),
//! which is how forced convection reaches the thermal result.
//!
//! Geometry is in millimetres (the rest of SolidCraft); physics is in SI inside the solvers.
//! The crate is wasm-clean: no threads are required, and `rayon` only joins in on native builds.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

pub mod backend;
mod flow;
mod grid;
pub mod lattice;
mod linsolve;
mod problem;
mod results;
mod sample;
mod thermal;
mod voxel;
mod vtk;
pub mod zones;

pub use backend::{Accelerator, CpuAccelerator, FlowFields, LatticeSetup, ThermalSolution, ThermalSystem};
pub use flow::{FlowReport, FlowSettings, Units, ZoneFlow};
pub use grid::{FLUID, Grid, Ijk};
pub use linsolve::Stencil;
pub use problem::{Air, Fan, Fluid, Limits, Problem, SolidBody, Thermal, Wall};
pub use results::{BodyReport, Results, Summary};
pub use sample::{MAX_SEEDS, MAX_STREAMLINE_STEPS, Streamline, StreamlineOptions, fan_seeds};
pub use thermal::ThermalReport;
pub use voxel::{Voxels, voxelize};
pub use vtk::write_vtk;

/// Everything that can go wrong. Never a panic: hostile numbers come back as `Invalid`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SimError {
    #[error("invalid simulation setup: {0}")]
    Invalid(String),
    #[error("simulation too large: {0}")]
    TooLarge(String),
    #[error("solver did not converge: {0}")]
    NotConverged(String),
    #[error("simulation cancelled")]
    Cancelled,
    #[error("simulation backend failed: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, SimError>;

/// Progress callback: `(phase, done, total)`; return `false` to cancel.
pub type Progress<'a> = &'a mut dyn FnMut(&str, usize, usize) -> bool;

/// Runs the whole study on the CPU reference backend: voxelise, solve airflow if asked for, then
/// the temperature field. Tests and oracles use it; the application requires a GPU backend
/// through [`solve_with`].
pub fn solve(problem: &Problem, progress: Progress) -> Result<Results> {
    solve_with(problem, &CpuAccelerator, progress)
}

/// Runs the whole study on `backend`. With buoyancy the flow and thermal solves alternate (the
/// flow needs the temperatures; the temperatures need the flow).
pub fn solve_with(problem: &Problem, backend: &dyn Accelerator, progress: Progress) -> Result<Results> {
    problem.validate()?;
    let grid = Grid::new(problem.domain, problem.cell_size, problem.limits.max_cells)?;
    if !progress("voxelize", 0, 1) {
        return Err(SimError::Cancelled);
    }
    let voxels = voxelize(&grid, &problem.bodies)?;
    let zones = zones::zones(&voxels, problem);
    let (thermal, flow) = match &problem.flow {
        None => (thermal::solve(&grid, &voxels, &zones, problem, None, backend, progress)?, None),
        Some(settings) if !settings.buoyancy => {
            let flow = flow::solve(&grid, &zones, problem, settings, None, backend, progress)?;
            (thermal::solve(&grid, &voxels, &zones, problem, Some(&flow), backend, progress)?, Some(flow))
        }
        Some(settings) => {
            let mut thermal = thermal::solve(&grid, &voxels, &zones, problem, None, backend, progress)?;
            let mut flow = None;
            for _ in 0..settings.coupling_iterations {
                let f = flow::solve(&grid, &zones, problem, settings, Some(&thermal.temperature), backend, progress)?;
                thermal = thermal::solve(&grid, &voxels, &zones, problem, Some(&f), backend, progress)?;
                flow = Some(f);
            }
            (thermal, flow)
        }
    };
    Ok(Results::new(grid, voxels, zones, problem, thermal, flow))
}

/// Shared helper: a finite, in-range number from hostile input.
pub(crate) fn finite(value: f64, low: f64, high: f64, what: &str) -> Result<f64> {
    if !value.is_finite() || value < low || value > high {
        return Err(SimError::Invalid(format!("{what}: expected a finite value in [{low}, {high}], got {value}")));
    }
    Ok(value)
}

/// `for_each` over chunks of a slice, in parallel on native builds and in order on wasm.
pub(crate) fn par_chunks_mut<T: Send, F: Fn(usize, &mut [T]) + Sync>(data: &mut [T], chunk: usize, f: F) {
    let chunk = chunk.max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
    }
    #[cfg(target_arch = "wasm32")]
    {
        data.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
    }
}

#[cfg(test)]
mod tests;
