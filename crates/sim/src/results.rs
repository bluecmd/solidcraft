//! What a solved study hands back: the fields on the grid, per-body numbers, and point queries.

use serde::{Deserialize, Serialize};
use solidcraft_geom::Vec3;

use crate::flow::FlowReport;
use crate::grid::{FLUID, Grid};
use crate::problem::Problem;
use crate::thermal::ThermalReport;
use crate::voxel::Voxels;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyReport {
    pub name: String,
    pub cells: usize,
    pub power_w: f64,
    pub min_c: f64,
    pub max_c: f64,
    pub mean_c: f64,
}

/// The numbers a person (or an agent) reads first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub cells: [usize; 3],
    pub cell_size_mm: f64,
    pub air_cells: usize,
    pub ambient_c: f64,
    pub bodies: Vec<BodyReport>,
    pub air_max_c: f64,
    pub air_mean_c: f64,
    pub heat_in_w: f64,
    pub heat_out_w: f64,
    pub thermal_iterations: usize,
    pub thermal_residual: f64,
    pub thermal_converged: bool,
    pub thermal_method: String,
    pub flow: Option<FlowSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlowSummary {
    pub steps: usize,
    pub converged: bool,
    pub change: f64,
    pub dt_s: f64,
    pub max_lattice_speed: f64,
    pub max_speed_m_s: f64,
    pub mean_speed_m_s: f64,
    pub effective_viscosity_m2_s: f64,
    pub backend: String,
    /// Million lattice updates per second the airflow solve achieved.
    pub mlups: f64,
}

#[derive(Clone, Debug)]
pub struct Results {
    pub grid: Grid,
    pub voxels: Voxels,
    pub thermal: ThermalReport,
    pub flow: Option<FlowReport>,
    pub summary: Summary,
}

impl Results {
    pub(crate) fn new(grid: Grid, voxels: Voxels, problem: &Problem, thermal: ThermalReport, flow: Option<FlowReport>) -> Results {
        let mut bodies = Vec::new();
        for (b, body) in problem.bodies.iter().enumerate() {
            let (mut min, mut max, mut sum, mut count) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
            for (idx, l) in voxels.label.iter().enumerate() {
                if *l as usize == b
                    && let Some(&t) = thermal.temperature.get(idx)
                {
                    min = min.min(t);
                    max = max.max(t);
                    sum += t;
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { f64::NAN };
            bodies.push(BodyReport {
                name: body.name.clone(),
                cells: count,
                power_w: body.power_w,
                min_c: if count > 0 { min } else { f64::NAN },
                max_c: if count > 0 { max } else { f64::NAN },
                mean_c: mean,
            });
        }
        let (mut air_max, mut air_sum, mut air_count) = (f64::NEG_INFINITY, 0.0, 0usize);
        for (idx, l) in voxels.label.iter().enumerate() {
            if *l == FLUID
                && let Some(&t) = thermal.temperature.get(idx)
            {
                air_max = air_max.max(t);
                air_sum += t;
                air_count += 1;
            }
        }
        let flow_summary = flow.as_ref().map(|f| {
            let (mut max, mut sum, mut n) = (0.0f64, 0.0f64, 0usize);
            for (idx, v) in f.velocity.iter().enumerate() {
                if voxels.label.get(idx) == Some(&FLUID) {
                    let s = f64::from(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    max = max.max(s);
                    sum += s;
                    n += 1;
                }
            }
            FlowSummary {
                steps: f.steps,
                converged: f.converged,
                change: f.change,
                dt_s: f.dt,
                max_lattice_speed: f.max_lattice_speed,
                max_speed_m_s: max,
                mean_speed_m_s: if n > 0 { sum / n as f64 } else { 0.0 },
                effective_viscosity_m2_s: f.effective_viscosity,
                backend: f.backend.clone(),
                mlups: f.mlups,
            }
        });
        let summary = Summary {
            cells: [grid.nx, grid.ny, grid.nz],
            cell_size_mm: grid.cell,
            air_cells: air_count,
            ambient_c: problem.thermal.ambient_c,
            bodies,
            air_max_c: if air_count > 0 { air_max } else { f64::NAN },
            air_mean_c: if air_count > 0 { air_sum / air_count as f64 } else { f64::NAN },
            heat_in_w: thermal.heat_in,
            heat_out_w: thermal.heat_out,
            thermal_iterations: thermal.iterations,
            thermal_residual: thermal.residual,
            thermal_converged: thermal.converged,
            thermal_method: thermal.method.clone(),
            flow: flow_summary,
        };
        Results { grid, voxels, thermal, flow, summary }
    }

    /// Temperature (°C) of the cell containing a world point.
    pub fn temperature_at(&self, p: Vec3) -> Option<f64> {
        let c = self.grid.cell_of(p)?;
        self.thermal.temperature.get(self.grid.index(c)).copied()
    }

    /// Air velocity (m/s) of the cell containing a world point; zero in solids.
    pub fn velocity_at(&self, p: Vec3) -> Option<Vec3> {
        let c = self.grid.cell_of(p)?;
        let v = self.flow.as_ref()?.velocity.get(self.grid.index(c))?;
        Some(Vec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])))
    }

    /// Temperature at a point on the surface of body `body`: the mean of that body's cells among
    /// the eight cells around the point (surface points sit on cell boundaries, so the containing
    /// cell is often air). Falls back to the nearest cell of the body within two cells.
    pub fn surface_temperature(&self, body: u32, p: Vec3) -> Option<f64> {
        let g = self.grid.grid_coords(p);
        if !g.is_finite() {
            return None;
        }
        let around = |radius: i64| -> Option<f64> {
            let (mut sum, mut n) = (0.0, 0usize);
            let (cx, cy, cz) = ((g.x - 0.5).floor() as i64, (g.y - 0.5).floor() as i64, (g.z - 0.5).floor() as i64);
            for dz in -radius + 1..=radius {
                for dy in -radius + 1..=radius {
                    for dx in -radius + 1..=radius {
                        let (i, j, k) = (cx + dx, cy + dy, cz + dz);
                        if i < 0 || j < 0 || k < 0 || i >= self.grid.nx as i64 || j >= self.grid.ny as i64 || k >= self.grid.nz as i64 {
                            continue;
                        }
                        let idx = self.grid.index([i as usize, j as usize, k as usize]);
                        if self.voxels.label.get(idx) == Some(&body)
                            && let Some(&t) = self.thermal.temperature.get(idx)
                        {
                            sum += t;
                            n += 1;
                        }
                    }
                }
            }
            (n > 0).then(|| sum / n as f64)
        };
        around(1).or_else(|| around(2))
    }

    /// Temperature range over the whole grid, °C.
    pub fn temperature_range(&self) -> (f64, f64) {
        self.thermal.temperature.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &t| (lo.min(t), hi.max(t)))
    }
}
