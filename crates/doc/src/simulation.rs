//! Saved simulation studies: what to simulate and with which boundary conditions. Results are
//! runtime state (`solidcraft_engine`), never part of the design.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use solidcraft_geom::Aabb3;
pub use solidcraft_sim::{Air, Fan, FlowSettings, Limits, Thermal, Wall};

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
            thermal: Thermal::default(),
            fans: Vec::new(),
            flow: None,
            limits: Limits::default(),
        }
    }
}

fn number(value: f64, low: f64, high: f64, what: &str) -> Result<()> {
    if !value.is_finite() || value < low || value > high {
        return Err(DocError::Invalid(format!("{what}: expected a finite value in [{low}, {high}]")));
    }
    Ok(())
}

impl Study {
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
