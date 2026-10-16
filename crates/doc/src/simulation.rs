//! Saved simulation studies: what to simulate and with which boundary conditions. Results are
//! runtime state (`solidcraft_engine`), never part of the design.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use solidcraft_geom::Aabb3;
pub use solidcraft_sim::{Air, Fan, FlowSettings, Fluid, Limits, Thermal, Wall};

use crate::{DocError, Result};

pub const MAX_STUDIES: usize = 32;
pub const MAX_STUDY_NAME_BYTES: usize = 128;
pub const MAX_STUDY_BODIES: usize = 10_000;
pub const MAX_STUDY_FANS: usize = 100;

/// Per-body settings of a study. A body without an entry takes part with its material's
/// conductivity and no heat.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BodySetup {
    /// Heat generated in the body, W.
    pub power_w: f64,
    /// W/(m K); `None` takes the body's physical material.
    pub conductivity: Option<f64>,
    /// Hold the body at this temperature (°C): a heat sink the study does not model, a cold plate.
    pub fixed_temperature_c: Option<f64>,
    /// Leave the body out of the study (it becomes air).
    pub exclude: bool,
    /// The body is a volume of this fluid (a name in [`Study::fluids`], or the built-in
    /// `water`): coolant in tubes and blocks, modelled as the coolant's own shape.
    pub fluid: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Study {
    /// Cell edge in mm; 0 chooses one from the domain size (about 60 cells along its longest side).
    pub cell_size_mm: f64,
    /// The box to simulate, world mm; `None` is the bodies' bounds grown by `padding_mm`.
    pub domain: Option<Aabb3>,
    pub padding_mm: f64,
    /// Settings by body name (the body's definition name for instances).
    pub bodies: BTreeMap<String, BodySetup>,
    /// `-x, +x, -y, +y, -z, +z`.
    pub walls: [Wall; 6],
    pub air: Air,
    /// Coolants by name (`water` is built in and may be omitted); bodies name them in
    /// [`BodySetup::fluid`].
    pub fluids: BTreeMap<String, Fluid>,
    pub thermal: Thermal,
    pub fans: Vec<Fan>,
    /// Solve airflow too; `None` keeps the air still (conduction with a mixing factor).
    pub flow: Option<FlowSettings>,
    pub limits: Limits,
}

impl Default for Study {
    fn default() -> Self {
        Study {
            cell_size_mm: 0.0,
            domain: None,
            padding_mm: 20.0,
            bodies: BTreeMap::new(),
            walls: [Wall::Open; 6],
            air: Air::default(),
            fluids: BTreeMap::new(),
            thermal: Thermal::default(),
            fans: Vec::new(),
            flow: None,
            limits: Limits::default(),
        }
    }
}

/// The name of the built-in coolant.
pub const WATER: &str = "water";
pub const MAX_STUDY_FLUIDS: usize = 16;

fn number(value: f64, low: f64, high: f64, what: &str) -> Result<()> {
    if !value.is_finite() || value < low || value > high {
        return Err(DocError::Invalid(format!("{what}: expected a finite value in [{low}, {high}]")));
    }
    Ok(())
}

impl Study {
    /// The coolants the study's bodies can name, in a fixed order: the defined ones by name, then
    /// the built-in `water` when some body uses it without defining it.
    pub fn fluid_list(&self) -> Vec<(String, Fluid)> {
        let mut list: Vec<(String, Fluid)> = self.fluids.iter().map(|(n, f)| (n.clone(), f.clone())).collect();
        if !self.fluids.contains_key(WATER) && self.bodies.values().any(|b| b.fluid.as_deref() == Some(WATER)) {
            list.push((WATER.to_string(), Fluid::water()));
        }
        list
    }

    pub fn validate(&self) -> Result<()> {
        number(self.cell_size_mm, 0.0, 1e4, "cell_size_mm")?;
        number(self.padding_mm, 0.0, 1e6, "padding_mm")?;
        if let Some(d) = &self.domain
            && (d.is_empty() || !d.min.is_finite() || !d.max.is_finite() || d.diagonal() > 1e7)
        {
            return Err(DocError::Invalid("domain: expected a finite, non-empty box".into()));
        }
        if self.bodies.len() > MAX_STUDY_BODIES {
            return Err(DocError::Invalid(format!("at most {MAX_STUDY_BODIES} body settings")));
        }
        for (name, b) in &self.bodies {
            if name.is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
                return Err(DocError::Invalid("body name: expected nonempty text of at most 256 bytes".into()));
            }
            number(b.power_w, 0.0, 1e9, &format!("{name}: power_w"))?;
            if let Some(k) = b.conductivity {
                number(k, 1e-6, 1e6, &format!("{name}: conductivity"))?;
            }
            if let Some(t) = b.fixed_temperature_c {
                number(t, -273.15, 1e5, &format!("{name}: fixed_temperature_c"))?;
            }
            if let Some(f) = &b.fluid
                && f != WATER
                && !self.fluids.contains_key(f)
            {
                return Err(DocError::Invalid(format!("{name}: fluid '{f}' is not defined in the study (fluids) and is not `{WATER}`")));
            }
        }
        if self.fluids.len() > MAX_STUDY_FLUIDS {
            return Err(DocError::Invalid(format!("at most {MAX_STUDY_FLUIDS} fluids")));
        }
        for (name, f) in &self.fluids {
            if name.is_empty() || name.len() > 64 || name.chars().any(char::is_control) {
                return Err(DocError::Invalid("fluid name: expected nonempty text of at most 64 bytes".into()));
            }
            f.validate(&format!("fluid '{name}'")).map_err(|e| DocError::Invalid(e.to_string()))?;
        }
        if self.fans.len() > MAX_STUDY_FANS {
            return Err(DocError::Invalid(format!("at most {MAX_STUDY_FANS} fans")));
        }
        // The solver's own checks cover air, thermal, fans, flow and limits.
        let probe = solidcraft_sim::Problem {
            domain: Aabb3 { min: solidcraft_geom::Vec3::ZERO, max: solidcraft_geom::Vec3::new(1.0, 1.0, 1.0) },
            cell_size: 1.0,
            bodies: vec![solidcraft_sim::SolidBody {
                name: "probe".into(),
                positions: vec![solidcraft_geom::Vec3::ZERO; 3],
                triangles: vec![[0, 1, 2]],
                conductivity: 1.0,
                power_w: 0.0,
                fixed_temperature_c: None,
                fluid: None,
            }],
            walls: self.walls,
            air: self.air.clone(),
            fluids: Vec::new(),
            thermal: self.thermal.clone(),
            fans: self.fans.clone(),
            flow: self.flow.clone(),
            limits: self.limits.clone(),
        };
        probe.validate().map_err(|e| DocError::Invalid(e.to_string()))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "UncheckedStudies")]
pub struct SimulationSettings {
    pub studies: BTreeMap<String, Study>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct UncheckedStudies {
    studies: BTreeMap<String, Study>,
}

impl TryFrom<UncheckedStudies> for SimulationSettings {
    type Error = DocError;
    fn try_from(value: UncheckedStudies) -> Result<Self> {
        let settings = SimulationSettings { studies: value.studies };
        settings.validate()?;
        Ok(settings)
    }
}

impl SimulationSettings {
    pub fn is_empty(&self) -> bool {
        self.studies.is_empty()
    }

    pub fn validate_name(name: &str) -> Result<()> {
        if name.trim().is_empty() || name.len() > MAX_STUDY_NAME_BYTES || name.chars().any(char::is_control) {
            return Err(DocError::Invalid(format!("study name: expected nonempty text of at most {MAX_STUDY_NAME_BYTES} bytes without controls")));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.studies.len() > MAX_STUDIES {
            return Err(DocError::Invalid(format!("at most {MAX_STUDIES} simulation studies")));
        }
        for (name, study) in &self.studies {
            Self::validate_name(name)?;
            study.validate().map_err(|e| DocError::Invalid(format!("study '{name}': {e}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate_and_round_trip() {
        let mut s = SimulationSettings::default();
        s.studies.insert("Thermal".into(), Study::default());
        s.validate().unwrap();
        let json = serde_json::to_value(&s).unwrap();
        let back: SimulationSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn fluids_are_named_and_water_is_built_in() {
        let mut s = Study::default();
        s.bodies.insert("Tube".into(), BodySetup { fluid: Some("water".into()), ..Default::default() });
        s.validate().unwrap();
        assert_eq!(s.fluid_list().len(), 1);
        assert_eq!(s.fluid_list()[0].0, "water");
        s.bodies.insert("Oil".into(), BodySetup { fluid: Some("oil".into()), ..Default::default() });
        assert!(s.validate().is_err(), "an undefined fluid");
        s.fluids.insert("oil".into(), Fluid { kinematic_viscosity: 5e-5, ..Fluid::water() });
        s.validate().unwrap();
        assert_eq!(s.fluid_list().iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), ["oil", "water"]);
    }

    #[test]
    fn loading_rejects_bad_numbers_and_unknown_fields() {
        let bad = serde_json::json!({"studies": {"x": {"cell_size_mm": -1.0}}});
        assert!(serde_json::from_value::<SimulationSettings>(bad).is_err());
        let bad = serde_json::json!({"studies": {"x": {"nonsense": 1}}});
        assert!(serde_json::from_value::<SimulationSettings>(bad).is_err());
        let bad = serde_json::json!({"studies": {"x": {"thermal": {"film_coefficient": f64::MAX}}}});
        assert!(serde_json::from_value::<SimulationSettings>(bad).is_err());
        let bad = serde_json::json!({"studies": {"": {}}});
        assert!(serde_json::from_value::<SimulationSettings>(bad).is_err());
    }
}
