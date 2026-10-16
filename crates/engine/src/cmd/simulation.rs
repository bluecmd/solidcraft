//! Simulation studies: thermal (conduction, film convection) and airflow (lattice Boltzmann) on
//! a voxel grid over the visible bodies. Setup lives in the design (`doc.simulation`); a solve's
//! results are session state for `sim.results`, `sim.probe`, `sim.export` and the viewport.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};
use solidcraft_doc::simulation::{BodySetup, SimulationSettings, Study};
use solidcraft_geom::{Aabb3, Vec3};
use solidcraft_sim::{Fan, Problem, SolidBody};

use super::CommandSpec;
use crate::params::bad;
use crate::{Result, Session, SimulationDisplay, SimulationRun};

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("sim.study.set", "Simulation Study", study_set).doc_only().at("SIMULATION", "SETUP").icon("box")
        .params("name: study name, set?: object of fields to replace (cell_size_mm, domain {min,max}, padding_mm, walls [6 of open|closed], air {conductivity, density, specific_heat, kinematic_viscosity, mixing_factor, thermal_expansion}, thermal {ambient_c, film_coefficient}, flow {max_steps, tolerance, smagorinsky, buoyancy, coupling_iterations, lattice_velocity} or null, limits {max_cells, max_iterations, tolerance}). Creates the study from defaults; validates atomically; one undo step; identical edits are no-ops. See docs/simulation.md"),
    CommandSpec::new("sim.study.get", "Simulation Studies", study_get).noundo().doc_only()
        .params("name?: study. Without name returns {studies, defaults}; with name returns {name, study}."),
    CommandSpec::new("sim.study.delete", "Delete Simulation Study", study_delete).doc_only().params("name: study to remove."),
    CommandSpec::new("sim.body.set", "Heat Source", body_set).doc_only().at("SIMULATION", "SETUP").icon("box")
        .params("study: study name (created from defaults if missing), body: body name (an instance's definition name applies to every instance), power_w?: W generated in the body, conductivity?: W/(m K) overriding the material (null to clear), fixed_temperature_c?: hold the body at this °C (null to clear), exclude?: leave the body out. Merges into the body's settings; one undo step."),
    CommandSpec::new("sim.fan.set", "Fan", fan_set).doc_only().at("SIMULATION", "SETUP").icon("box")
        .params("study: study name (created if missing), name: fan name, region: {min:[x,y,z], max:[x,y,z]} world mm box of air the fan drives, velocity: [x,y,z] m/s. Replaces a fan of that name; one undo step. Airflow is solved when the study has `flow` settings."),
    CommandSpec::new("sim.fan.delete", "Delete Fan", fan_delete).doc_only().params("study: study name, name: fan to remove."),
    CommandSpec::new("sim.solve", "Solve Simulation", solve).noundo().at("SIMULATION", "SOLVE").icon("box")
        .params("study: study name, flow?: true to solve airflow with default flow settings, false to keep the air still (overrides the study for this solve). Voxelises the visible bodies, solves airflow (if asked) and the temperature field; keeps the results for sim.results, sim.probe, sim.export and sim.display. Returns the summary: per-body min/max/mean °C, air temperatures, heat balance, convergence. Does not change the design. See docs/simulation.md"),
    CommandSpec::new("sim.results", "Simulation Results", results).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("No parameters. The summary of the last solve, with `stale: true` when the geometry changed since."),
    CommandSpec::new("sim.probe", "Probe Simulation", probe).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("point: [x, y, z] world mm. The temperature (°C) and, when airflow was solved, the air velocity (m/s) of the cell there."),
    CommandSpec::new("sim.export", "Export Simulation (VTK)", export).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("path: destination .vtk file (legacy VTK structured points: body, temperature, velocity, pressure per cell; opens in ParaView), overwrite?: false."),
    CommandSpec::new("sim.display", "Show Simulation", display).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("mode: off | temperature (colour map on the bodies) | velocity (air arrows and the colour map)."),
];

fn settings_mut(s: &mut Session) -> SimulationSettings {
    (*s.doc.simulation).clone()
}

fn store(s: &mut Session, cmd: &str, settings: SimulationSettings) -> Result<()> {
    settings.validate().map_err(|e| bad(cmd, e.to_string()))?;
    if *s.doc.simulation != settings {
        s.doc_mut().simulation = Arc::new(settings);
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StudySet {
    name: String,
    #[serde(default)]
    set: Option<Value>,
}

fn study_set(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.study.set";
    let request: StudySet = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    SimulationSettings::validate_name(&request.name).map_err(|e| bad(cmd, e.to_string()))?;
    let current = s.doc.simulation.studies.get(&request.name).cloned().unwrap_or_default();
    let mut value = serde_json::to_value(&current).map_err(|e| bad(cmd, e.to_string()))?;
    if let Some(set) = &request.set {
        let patch = set.as_object().filter(|p| p.len() <= 12).ok_or_else(|| bad(cmd, "`set` must be an object with at most 12 fields"))?;
        let object = value.as_object_mut().ok_or_else(|| bad(cmd, "invalid study"))?;
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }
    let study: Study = serde_json::from_value(value).map_err(|e| bad(cmd, e.to_string()))?;
    study.validate().map_err(|e| bad(cmd, e.to_string()))?;
    let reply = json!({"name": request.name, "study": study});
    let mut settings = settings_mut(s);
    settings.studies.insert(request.name, study);
    store(s, cmd, settings)?;
    Ok(reply)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Named {
    #[serde(default)]
    name: Option<String>,
}

fn study_get(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.study.get";
    let request: Named = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    match request.name {
        Some(name) => {
            SimulationSettings::validate_name(&name).map_err(|e| bad(cmd, e.to_string()))?;
            let study = s.doc.simulation.studies.get(&name).ok_or_else(|| bad(cmd, "no simulation study with that name"))?;
            Ok(json!({"name": name, "study": study}))
        }
        None => Ok(json!({"studies": s.doc.simulation.studies, "defaults": Study::default()})),
    }
}

fn study_delete(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.study.delete";
    let request: Named = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    let name = request.name.ok_or_else(|| bad(cmd, "`name` is required"))?;
    let mut settings = settings_mut(s);
    if settings.studies.remove(&name).is_none() {
        return Err(bad(cmd, "no simulation study with that name"));
    }
    store(s, cmd, settings)?;
    Ok(json!({"name": name, "deleted": true}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BodySet {
    study: String,
    body: String,
    #[serde(default)]
    power_w: Option<f64>,
    /// Absent: keep; `null`: clear; number: set.
    #[serde(default, deserialize_with = "present")]
    conductivity: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    fixed_temperature_c: Option<Value>,
    #[serde(default)]
    exclude: Option<bool>,
}

/// Keeps an explicit `null` apart from an absent field.
fn present<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

fn optional_number(cmd: &str, v: &Option<Value>, what: &str) -> Result<Option<Option<f64>>> {
    match v {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::Number(n)) => {
            let x = n.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(cmd, format!("`{what}` must be a finite number or null")))?;
            Ok(Some(Some(x)))
        }
        Some(_) => Err(bad(cmd, format!("`{what}` must be a finite number or null"))),
    }
}

fn body_set(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.body.set";
    let request: BodySet = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    SimulationSettings::validate_name(&request.study).map_err(|e| bad(cmd, e.to_string()))?;
    if request.body.is_empty() || request.body.len() > 256 {
        return Err(bad(cmd, "`body` must be a body name"));
    }
    let mut settings = settings_mut(s);
    let study = settings.studies.entry(request.study.clone()).or_default();
    let setup: &mut BodySetup = study.bodies.entry(request.body.clone()).or_default();
    if let Some(p) = request.power_w {
        setup.power_w = p;
    }
    if let Some(k) = optional_number(cmd, &request.conductivity, "conductivity")? {
        setup.conductivity = k;
    }
    if let Some(t) = optional_number(cmd, &request.fixed_temperature_c, "fixed_temperature_c")? {
        setup.fixed_temperature_c = t;
    }
    if let Some(e) = request.exclude {
        setup.exclude = e;
    }
    let reply = json!({"study": request.study, "body": request.body, "setup": setup});
    store(s, cmd, settings)?;
    Ok(reply)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FanSet {
    study: String,
    name: String,
    region: Aabb3,
    velocity: Vec3,
}

fn fan_set(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.fan.set";
    let request: FanSet = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    SimulationSettings::validate_name(&request.study).map_err(|e| bad(cmd, e.to_string()))?;
    SimulationSettings::validate_name(&request.name).map_err(|e| bad(cmd, format!("fan {e}")))?;
    let fan = Fan { name: request.name.clone(), region: request.region, velocity: request.velocity };
    let mut settings = settings_mut(s);
    let study = settings.studies.entry(request.study.clone()).or_default();
    match study.fans.iter_mut().find(|f| f.name == fan.name) {
        Some(existing) => *existing = fan.clone(),
        None => study.fans.push(fan.clone()),
    }
    store(s, cmd, settings)?;
    Ok(json!({"study": request.study, "fan": fan}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FanDelete {
    study: String,
    name: String,
}

fn fan_delete(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.fan.delete";
    let request: FanDelete = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    let mut settings = settings_mut(s);
    let study = settings.studies.get_mut(&request.study).ok_or_else(|| bad(cmd, "no simulation study with that name"))?;
    let before = study.fans.len();
    study.fans.retain(|f| f.name != request.name);
    if study.fans.len() == before {
        return Err(bad(cmd, "no fan with that name"));
    }
    store(s, cmd, settings)?;
    Ok(json!({"study": request.study, "name": request.name, "deleted": true}))
}

/// The study as a solver problem over the visible bodies, and the world body name per index.
pub(crate) fn problem(s: &Session, name: &str, study: &Study, flow_override: Option<bool>) -> Result<(Problem, Vec<String>, Vec<String>)> {
    let cmd = "sim.solve";
    let st = s.world_state();
    let mut bodies = Vec::new();
    let mut names = Vec::new();
    let mut used: Vec<&str> = Vec::new();
    let mut bounds = Aabb3::EMPTY;
    for b in &st.bodies {
        if !s.body_visible(&b.name) {
            continue;
        }
        let def = b.definition_name();
        let (key, setup) = match study.bodies.get_key_value(b.name.as_str()).or_else(|| study.bodies.get_key_value(def)) {
            Some((k, v)) => (Some(k.as_str()), v.clone()),
            None => (None, BodySetup::default()),
        };
        if let Some(k) = key {
            used.push(k);
        }
        if setup.exclude {
            continue;
        }
        let mesh = b.mesh();
        if mesh.triangles.is_empty() {
            continue;
        }
        let conductivity = setup.conductivity.unwrap_or_else(|| s.doc.thermal_conductivity(def));
        bounds = bounds.union(&mesh.bounds());
        bodies.push(SolidBody {
            name: b.name.clone(),
            positions: mesh.positions.clone(),
            triangles: mesh.triangles.clone(),
            conductivity,
            power_w: setup.power_w,
            fixed_temperature_c: setup.fixed_temperature_c,
        });
        names.push(b.name.clone());
    }
    if bodies.is_empty() {
        return Err(bad(cmd, format!("study '{name}': no visible bodies to simulate")));
    }
    let unknown: Vec<String> = study.bodies.keys().filter(|k| !used.contains(&k.as_str())).cloned().collect();
    let domain = match study.domain {
        Some(d) => d,
        None => {
            let pad = Vec3::new(study.padding_mm, study.padding_mm, study.padding_mm);
            Aabb3 { min: bounds.min - pad, max: bounds.max + pad }
        }
    };
    let size = domain.size();
    let cell = if study.cell_size_mm > 0.0 { study.cell_size_mm } else { (size.x.max(size.y).max(size.z) / 60.0).max(1e-3) };
    let flow = match flow_override {
        Some(true) => Some(study.flow.clone().unwrap_or_default()),
        Some(false) => None,
        None => study.flow.clone(),
    };
    let problem = Problem {
        domain,
        cell_size: cell,
        bodies,
        walls: study.walls,
        air: study.air.clone(),
        thermal: study.thermal.clone(),
        fans: study.fans.clone(),
        flow,
        limits: study.limits.clone(),
    };
    Ok((problem, names, unknown))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Solve {
    study: String,
    #[serde(default)]
    flow: Option<bool>,
}

fn summary_json(run: &SimulationRun, stale: bool) -> Value {
    let mut v = serde_json::to_value(&run.results.summary).unwrap_or(Value::Null);
    if let Some(o) = v.as_object_mut() {
        o.insert("study".into(), json!(run.study));
        o.insert("revision".into(), json!(run.revision));
        o.insert("stale".into(), json!(stale));
        let (lo, hi) = run.results.temperature_range();
        o.insert("min_c".into(), json!(lo));
        o.insert("max_c".into(), json!(hi));
    }
    v
}

fn solve(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.solve";
    let request: Solve = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    let study = s.doc.simulation.studies.get(&request.study).cloned().ok_or_else(|| bad(cmd, "no simulation study with that name"))?;
    let (problem, bodies, unknown) = problem(s, &request.study, &study, request.flow)?;
    let mut go = |_: &str, _: usize, _: usize| true;
    let results = solidcraft_sim::solve(&problem, &mut go).map_err(|e| bad(cmd, e.to_string()))?;
    let run = Arc::new(SimulationRun { study: request.study, revision: s.revision, results, bodies });
    let mut reply = summary_json(&run, false);
    if let Some(o) = reply.as_object_mut() {
        o.insert("unknown_bodies".into(), json!(unknown));
        o.insert("domain".into(), json!({"min": problem.domain.min, "max": problem.domain.max}));
    }
    s.simulation = Some(run);
    if s.simulation_display == SimulationDisplay::Off {
        s.simulation_display = SimulationDisplay::Temperature;
    }
    Ok(reply)
}

fn results(s: &mut Session, params: &Value) -> Result<Value> {
    if !params.as_object().is_some_and(|p| p.is_empty()) {
        return Err(bad("sim.results", "expected an empty parameter object"));
    }
    let run = s.simulation.as_ref().ok_or_else(|| bad("sim.results", "no simulation has been solved; run sim.solve first"))?;
    Ok(summary_json(run, run.revision != s.revision))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    point: Vec3,
}

fn probe(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.probe";
    let request: Probe = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    if !request.point.is_finite() {
        return Err(bad(cmd, "`point` must be finite"));
    }
    let run = s.simulation.as_ref().ok_or_else(|| bad(cmd, "no simulation has been solved; run sim.solve first"))?;
    let r = &run.results;
    let cell = r.grid.cell_of(request.point).ok_or_else(|| bad(cmd, "the point is outside the simulated domain"))?;
    let idx = r.grid.index(cell);
    let label = r.voxels.label.get(idx).copied().unwrap_or(solidcraft_sim::FLUID);
    let body = if label == solidcraft_sim::FLUID { Value::Null } else { json!(run.bodies.get(label as usize)) };
    Ok(json!({
        "point": request.point,
        "cell": cell,
        "body": body,
        "temperature_c": r.temperature_at(request.point),
        "velocity_m_s": r.velocity_at(request.point),
        "stale": run.revision != s.revision,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Export {
    path: String,
    #[serde(default)]
    overwrite: bool,
}

fn export(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.export";
    let request: Export = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    let path = std::path::Path::new(&request.path);
    if request.path.len() > 4096 || request.path.chars().any(char::is_control) || !path.extension().is_some_and(|x| x.eq_ignore_ascii_case("vtk")) {
        return Err(bad(cmd, "a bounded .vtk destination path is required"));
    }
    let run = s.simulation.as_ref().ok_or_else(|| bad(cmd, "no simulation has been solved; run sim.solve first"))?;
    solidcraft_io::vfs::check(&request.path).map_err(|e| bad(cmd, e.to_string()))?;
    let text = solidcraft_sim::write_vtk(&run.results);
    let result = if request.overwrite {
        solidcraft_io::write_atomic(path, text.as_bytes(), false)
    } else {
        solidcraft_io::safe_write::create_atomic(path, text.as_bytes())
    };
    result.map_err(|e| bad(cmd, format!("could not write VTK: {e}")))?;
    Ok(json!({"path": request.path, "bytes": text.len(), "cells": run.results.grid.len()}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Display {
    mode: SimulationDisplay,
}

fn display(s: &mut Session, params: &Value) -> Result<Value> {
    let request: Display = serde_json::from_value(params.clone()).map_err(|e| bad("sim.display", e.to_string()))?;
    s.simulation_display = request.mode;
    Ok(json!({"mode": request.mode, "solved": s.simulation.is_some()}))
}

#[cfg(test)]
#[path = "simulation_tests.rs"]
mod tests;
