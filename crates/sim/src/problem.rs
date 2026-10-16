//! The study as the solver sees it: meshes, material numbers, sources and boundary conditions.
//! Geometry in mm; everything else in SI (W, m, K, kg, s).

use serde::{Deserialize, Serialize};
use solidcraft_geom::{Aabb3, Vec3};

use crate::{Result, SimError, finite};

/// One solid body in the study.
#[derive(Clone, Debug, PartialEq)]
pub struct SolidBody {
    pub name: String,
    /// Closed triangle mesh in world mm (counter-clockwise seen from outside).
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// W/(m K).
    pub conductivity: f64,
    /// Heat generated inside the body, W, spread evenly over its volume.
    pub power_w: f64,
    /// Held at this temperature (°C): a heat sink, a cold plate, a boundary the study does not model.
    pub fixed_temperature_c: Option<f64>,
}

impl SolidBody {
    pub fn bounds(&self) -> Aabb3 {
        let mut b = Aabb3::EMPTY;
        for p in &self.positions {
            b.add(*p);
        }
        b
    }
}

/// What a domain wall does. Order: `-x, +x, -y, +y, -z, +z`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Wall {
    /// Ambient air continues past the wall: temperature is the ambient, air may flow in or out.
    #[default]
    Open,
    /// An insulated solid wall: no heat and no air crosses it.
    Closed,
}

/// Air (or another fluid) filling everything that is not a body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Air {
    /// W/(m K); 0.026 for air at room temperature.
    pub conductivity: f64,
    /// kg/m³.
    pub density: f64,
    /// J/(kg K).
    pub specific_heat: f64,
    /// m²/s; 1.5e-5 for air.
    pub kinematic_viscosity: f64,
    /// Multiplies the conductivity of air cells to stand in for the mixing that natural convection
    /// provides on a grid too coarse to resolve it: 1 is still air, 5–10 a sealed enclosure, 100 or
    /// more a part in free air. Ignored (set to 1) where airflow is solved.
    pub mixing_factor: f64,
    /// 1/K; 3.4e-3 for air. Buoyancy for natural convection when airflow is solved.
    pub thermal_expansion: f64,
}

impl Default for Air {
    fn default() -> Self {
        Air { conductivity: 0.026, density: 1.2, specific_heat: 1005.0, kinematic_viscosity: 1.5e-5, mixing_factor: 10.0, thermal_expansion: 3.4e-3 }
    }
}

/// Thermal boundary conditions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Thermal {
    /// °C of the surrounding air and of open walls.
    pub ambient_c: f64,
    /// W/(m² K) between a solid surface and the air cell next to it: the boundary layer the grid
    /// cannot resolve. Natural convection 5–10, forced 20–100; 0 for pure conduction.
    pub film_coefficient: f64,
}

impl Default for Thermal {
    fn default() -> Self {
        Thermal { ambient_c: 20.0, film_coefficient: 8.0 }
    }
}

/// A fan: air inside `region` (world mm, axis-aligned) is driven at `velocity` (m/s).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fan {
    pub name: String,
    pub region: Aabb3,
    pub velocity: Vec3,
}

/// Caps that keep a hostile or careless setup from eating the machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_cells: usize,
    /// Linear solver iterations for the temperature field.
    pub max_iterations: usize,
    /// Relative residual at which the temperature solve stops.
    pub tolerance: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_cells: 4_000_000, max_iterations: 5000, tolerance: 1e-7 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// World mm; everything outside is ambient.
    pub domain: Aabb3,
    /// mm.
    pub cell_size: f64,
    pub bodies: Vec<SolidBody>,
    pub walls: [Wall; 6],
    pub air: Air,
    pub thermal: Thermal,
    pub fans: Vec<Fan>,
    /// Solve airflow (forced by fans, buoyant from heat) and let it carry heat; `None` keeps air still.
    pub flow: Option<crate::FlowSettings>,
    pub limits: Limits,
}

impl Problem {
    pub fn validate(&self) -> Result<()> {
        if self.domain.is_empty() || !self.domain.min.is_finite() || !self.domain.max.is_finite() {
            return Err(SimError::Invalid("domain box is empty".into()));
        }
        finite(self.cell_size, 1e-3, 1e4, "cell size (mm)")?;
        if self.bodies.is_empty() {
            return Err(SimError::Invalid("no bodies in the study".into()));
        }
        if self.bodies.len() > 10_000 {
            return Err(SimError::TooLarge("more than 10000 bodies".into()));
        }
        for b in &self.bodies {
            let what = format!("body '{}'", b.name);
            finite(b.conductivity, 1e-6, 1e6, &format!("{what} conductivity"))?;
            finite(b.power_w, 0.0, 1e9, &format!("{what} power"))?;
            if let Some(t) = b.fixed_temperature_c {
                finite(t, -273.15, 1e5, &format!("{what} fixed temperature"))?;
            }
            if b.triangles.is_empty() {
                return Err(SimError::Invalid(format!("{what} has no triangles")));
            }
            if b.triangles.len() > 50_000_000 {
                return Err(SimError::TooLarge(format!("{what} has too many triangles")));
            }
        }
        finite(self.air.conductivity, 0.0, 1e6, "air conductivity")?;
        finite(self.air.density, 1e-6, 1e6, "air density")?;
        finite(self.air.specific_heat, 1e-6, 1e9, "air specific heat")?;
        finite(self.air.kinematic_viscosity, 1e-12, 1.0, "air kinematic viscosity")?;
        finite(self.air.mixing_factor, 0.0, 1e9, "air mixing factor")?;
        finite(self.air.thermal_expansion, 0.0, 1.0, "air thermal expansion")?;
        finite(self.thermal.ambient_c, -273.15, 1e5, "ambient temperature")?;
        finite(self.thermal.film_coefficient, 0.0, 1e6, "film coefficient")?;
        if self.fans.len() > 1000 {
            return Err(SimError::TooLarge("more than 1000 fans".into()));
        }
        for f in &self.fans {
            if f.region.is_empty() || !f.region.min.is_finite() || !f.region.max.is_finite() {
                return Err(SimError::Invalid(format!("fan '{}': empty region", f.name)));
            }
            if !f.velocity.is_finite() || f.velocity.len() > 1e3 {
                return Err(SimError::Invalid(format!("fan '{}': velocity must be finite and below 1000 m/s", f.name)));
            }
        }
        if let Some(flow) = &self.flow {
            flow.validate()?;
        }
        if self.limits.max_cells == 0 || self.limits.max_cells > 64_000_000 {
            return Err(SimError::Invalid("max_cells must be in 1..=64000000".into()));
        }
        if self.limits.max_iterations == 0 || self.limits.max_iterations > 1_000_000 {
            return Err(SimError::Invalid("max_iterations must be in 1..=1000000".into()));
        }
        finite(self.limits.tolerance, 1e-15, 1.0, "tolerance")?;
        Ok(())
    }
}
