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

/// A study prepared for solving: everything `sim.solve` needs, detached from the session so a
/// UI can solve it off the window thread and hand the run back with `Session::set_simulation`.
pub struct Prepared {
    pub study: String,
    pub revision: u64,
    pub problem: Problem,
    /// World body name per body index of the problem.
    pub bodies: Vec<String>,
    /// Study body settings that matched no visible body.
    pub unknown: Vec<String>,
}

impl Prepared {
    pub fn solve(
        self,
        backend: &dyn solidcraft_sim::Accelerator,
        progress: solidcraft_sim::Progress,
    ) -> std::result::Result<SimulationRun, solidcraft_sim::SimError> {
        let results = solidcraft_sim::solve_with(&self.problem, backend, progress)?;
        Ok(SimulationRun { study: self.study, revision: self.revision, results, bodies: self.bodies })
    }
}

/// The refusal when no backend is installed.
pub const NO_BACKEND: &str = "no GPU simulation backend is installed: simulation requires a GPU";

type Factory = Box<dyn Fn() -> Option<Arc<dyn solidcraft_sim::Accelerator>> + Send + Sync>;
static FACTORY: std::sync::OnceLock<Factory> = std::sync::OnceLock::new();
static SHARED: std::sync::OnceLock<Option<Arc<dyn solidcraft_sim::Accelerator>>> = std::sync::OnceLock::new();

/// Register, once per process, how to open the simulation backend (the application's GPU). It
/// is opened lazily by the first solve and shared by every session without one of its own.
pub fn register_backend_factory(factory: impl Fn() -> Option<Arc<dyn solidcraft_sim::Accelerator>> + Send + Sync + 'static) {
    let _ = FACTORY.set(Box::new(factory));
}

/// The process-wide backend, opened on first use by the registered factory.
pub fn shared_backend() -> Option<Arc<dyn solidcraft_sim::Accelerator>> {
    SHARED.get_or_init(|| FACTORY.get().and_then(|f| f())).clone()
}

/// Prepare a study of the design for solving. `flow` overrides the study's airflow setting.
pub fn prepare(s: &Session, study: &str, flow: Option<bool>) -> Result<Prepared> {
    let setup = s.doc.simulation.studies.get(study).cloned().ok_or_else(|| bad("sim.solve", "no simulation study with that name"))?;
    let (problem, bodies, unknown) = problem(s, study, &setup, flow)?;
    Ok(Prepared { study: study.to_string(), revision: s.revision, problem, bodies, unknown })
}

impl Session {
    /// Install what runs simulation solves (the application's GPU backend). `None` uninstalls.
    pub fn set_simulation_accelerator(&mut self, backend: Option<Arc<dyn solidcraft_sim::Accelerator>>) {
        self.simulation_accelerator = backend;
    }

    /// The installed simulation backend, else the process-wide one (`register_backend_factory`).
    pub fn simulation_accelerator(&self) -> Option<Arc<dyn solidcraft_sim::Accelerator>> {
        self.simulation_accelerator.clone().or_else(shared_backend)
    }

    /// Keep a solved study as the session's simulation (what the viewport shows and `sim.results`
    /// reports); the display switches on if it was off.
    pub fn set_simulation(&mut self, run: SimulationRun) {
        self.simulation = Some(Arc::new(run));
        if self.simulation_display == SimulationDisplay::Off {
            self.simulation_display = SimulationDisplay::Temperature;
        }
    }
}

/// Colour for a value in 0..=1: dark blue through cyan, green and yellow to red.
pub fn heat_color(t: f64) -> [u8; 3] {
    const STOPS: [[f32; 3]; 5] = [[30.0, 20.0, 110.0], [40.0, 120.0, 230.0], [50.0, 210.0, 150.0], [245.0, 215.0, 40.0], [210.0, 30.0, 25.0]];
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 } as f32;
    let x = t * (STOPS.len() - 1) as f32;
    let i = (x.floor() as usize).min(STOPS.len() - 2);
    let f = x - i as f32;
    let (a, b) = (STOPS[i], STOPS[i + 1]);
    [(a[0] + (b[0] - a[0]) * f) as u8, (a[1] + (b[1] - a[1]) * f) as u8, (a[2] + (b[2] - a[2]) * f) as u8]
}

/// The temperature range the colour map spans: the bodies' surfaces.
pub fn body_range(run: &SimulationRun) -> (f64, f64) {
    let s = &run.results.summary;
    let lo = s.bodies.iter().map(|b| b.min_c).filter(|t| t.is_finite()).fold(f64::INFINITY, f64::min);
    let hi = s.bodies.iter().map(|b| b.max_c).filter(|t| t.is_finite()).fold(f64::NEG_INFINITY, f64::max);
    if !lo.is_finite() || !hi.is_finite() {
        return (s.ambient_c, s.ambient_c + 1.0);
    }
    if hi - lo < 0.1 { (lo, lo + 0.1) } else { (lo, hi) }
}

/// Per-vertex colours of a body's mesh from the solved surface temperatures.
pub fn vertex_colors(run: &SimulationRun, body: u32, mesh: &solidcraft_geom::Mesh, lo: f64, hi: f64) -> Vec<[u8; 3]> {
    let span = (hi - lo).max(1e-9);
    mesh.positions
        .iter()
        .map(|p| match run.results.surface_temperature(body, *p) {
            Some(t) => heat_color((t - lo) / span),
            None => [120, 120, 120],
        })
        .collect()
}

/// One air-velocity arrow piece: a dim shaft or a bright tip, as a line segment.
pub struct ArrowSegment {
    pub a: Vec3,
    pub b: Vec3,
    pub color: [u8; 3],
    pub width: f64,
}

/// Air velocity arrows: a thinned sample of the air cells, each a line along its velocity,
/// coloured and lengthened by speed.
pub fn velocity_arrows(run: &SimulationRun) -> Vec<ArrowSegment> {
    let mut out = Vec::new();
    let r = &run.results;
    let Some(flow) = &r.flow else { return out };
    let max = r.summary.flow.as_ref().map_or(0.0, |f| f.max_speed_m_s);
    if !(max > 0.0) {
        return out;
    }
    let cells = r.grid.len() as f64;
    let stride = ((cells / 20_000.0).cbrt().ceil() as usize).max(1);
    let scale = r.grid.cell * stride as f64;
    for k in (0..r.grid.nz).step_by(stride) {
        for j in (0..r.grid.ny).step_by(stride) {
            for i in (0..r.grid.nx).step_by(stride) {
                let idx = r.grid.index([i, j, k]);
                if r.voxels.label.get(idx) != Some(&solidcraft_sim::FLUID) {
                    continue;
                }
                let Some(v) = flow.velocity.get(idx) else { continue };
                let v = Vec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
                let speed = v.len();
                if speed < 0.03 * max {
                    continue;
                }
                let f = speed / max;
                let p0 = r.grid.centre([i, j, k]);
                let dir = v * (1.0 / speed);
                let len = scale * (0.25 + 0.7 * f);
                let mid = p0 + dir * (len * 0.65);
                let c = heat_color(f);
                out.push(ArrowSegment { a: p0, b: mid, color: [c[0] / 2 + 40, c[1] / 2 + 40, c[2] / 2 + 40], width: 1.2 });
                out.push(ArrowSegment { a: mid, b: p0 + dir * len, color: c, width: 2.4 });
            }
        }
    }
    out
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
    let backend = s.simulation_accelerator().ok_or_else(|| bad(cmd, NO_BACKEND))?;
    let prepared = prepare(s, &request.study, request.flow)?;
    let unknown = prepared.unknown.clone();
    let domain = prepared.problem.domain;
    let mut go = |_: &str, _: usize, _: usize| true;
    let run = prepared.solve(&*backend, &mut go).map_err(|e| bad(cmd, e.to_string()))?;
    let mut reply = summary_json(&run, false);
    if let Some(o) = reply.as_object_mut() {
        o.insert("unknown_bodies".into(), json!(unknown));
        o.insert("domain".into(), json!({"min": domain.min, "max": domain.max}));
        o.insert("backend".into(), json!(backend.name()));
    }
    s.set_simulation(run);
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
