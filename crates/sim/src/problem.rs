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
    /// The body is a volume of fluid (index into [`Problem::fluids`]): water in tubes and
    /// blocks, modelled as the coolant's own shape. It flows (driven by fans/pumps whose region
    /// covers it), carries heat, and bounces off every other cell.
    pub fluid: Option<usize>,
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

/// A fluid: the air filling everything that is not a body, or a coolant in tagged bodies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fluid {
    /// W/(m K); 0.026 for air at room temperature, 0.6 for water.
    pub conductivity: f64,
    /// kg/m³.
    pub density: f64,
    /// J/(kg K).
    pub specific_heat: f64,
    /// m²/s; 1.5e-5 for air, 1e-6 for water.
    pub kinematic_viscosity: f64,
    /// Multiplies the conductivity of still cells to stand in for the mixing that natural
    /// convection provides on a grid too coarse to resolve it: 1 is still fluid, 5–10 a sealed
    /// enclosure, 100 or more a part in free air. Ignored (set to 1) where the flow is solved.
    pub mixing_factor: f64,
    /// 1/K; 3.4e-3 for air, 2e-4 for water. Buoyancy when the flow is solved.
    pub thermal_expansion: f64,
    /// W/(m² K) between a solid surface and this fluid's cell next to it, standing in for the
    /// boundary layer; `None` uses the study's [`Thermal::film_coefficient`] (meant for air).
    /// Water against a wall runs in the thousands.
    pub film_coefficient: Option<f64>,
}

/// The air the study is bathed in.
pub type Air = Fluid;

impl Default for Fluid {
    fn default() -> Self {
        Fluid::air()
    }
}

impl Fluid {
    pub fn air() -> Fluid {
        Fluid {
            conductivity: 0.026,
            density: 1.2,
            specific_heat: 1005.0,
            kinematic_viscosity: 1.5e-5,
            mixing_factor: 10.0,
            thermal_expansion: 3.4e-3,
            film_coefficient: None,
        }
    }
    /// Water at room temperature.
    pub fn water() -> Fluid {
        Fluid {
            conductivity: 0.6,
            density: 998.0,
            specific_heat: 4180.0,
            kinematic_viscosity: 1.0e-6,
            mixing_factor: 1.0,
            thermal_expansion: 2.1e-4,
            film_coefficient: Some(3000.0),
        }
    }
    pub fn validate(&self, what: &str) -> Result<()> {
        finite(self.conductivity, 0.0, 1e6, &format!("{what} conductivity"))?;
        finite(self.density, 1e-6, 1e6, &format!("{what} density"))?;
        finite(self.specific_heat, 1e-6, 1e9, &format!("{what} specific heat"))?;
        finite(self.kinematic_viscosity, 1e-12, 1.0, &format!("{what} kinematic viscosity"))?;
        finite(self.mixing_factor, 0.0, 1e9, &format!("{what} mixing factor"))?;
        finite(self.thermal_expansion, 0.0, 1.0, &format!("{what} thermal expansion"))?;
        if let Some(h) = self.film_coefficient {
            finite(h, 0.0, 1e7, &format!("{what} film coefficient"))?;
        }
        Ok(())
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
    pub air: Fluid,
    /// Other fluids, in the bodies that name them ([`SolidBody::fluid`]): zone `k + 1`.
    pub fluids: Vec<Fluid>,
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
            if let Some(f) = b.fluid
                && f >= self.fluids.len()
            {
                return Err(SimError::Invalid(format!("{what} names fluid {f}, but the study has {}", self.fluids.len())));
            }
        }
        self.air.validate("air")?;
        if self.fluids.len() > 64 {
            return Err(SimError::TooLarge("more than 64 fluids".into()));
        }
        for (k, f) in self.fluids.iter().enumerate() {
            f.validate(&format!("fluid {k}"))?;
        }
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
