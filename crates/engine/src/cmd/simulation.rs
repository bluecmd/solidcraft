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
        .params("study: study name (created from defaults if missing), body: body name (an instance's definition name applies to every instance), power_w?: W generated in the body, conductivity?: W/(m K) overriding the material (null to clear), fixed_temperature_c?: hold the body at this °C (null to clear), exclude?: leave the body out, fluid?: the body is a volume of this coolant (`water`, or a name from sim.fluid.set; null to clear). Merges into the body's settings; one undo step."),
    CommandSpec::new("sim.fan.set", "Fan", fan_set).doc_only().at("SIMULATION", "SETUP").icon("box")
        .params("study: study name (created if missing), name: fan name, region: {min:[x,y,z], max:[x,y,z]} world mm box of air the fan drives, velocity: [x,y,z] m/s. Replaces a fan of that name; one undo step. Airflow is solved when the study has `flow` settings."),
    CommandSpec::new("sim.fan.delete", "Delete Fan", fan_delete).doc_only().params("study: study name, name: fan to remove."),
    CommandSpec::new("sim.fluid.set", "Coolant", fluid_set).doc_only().at("SIMULATION", "SETUP").icon("box")
        .params("study: study name (created if missing), name: fluid name (`water` is built in), set?: object of fields to replace (conductivity, density, specific_heat, kinematic_viscosity, mixing_factor, thermal_expansion, film_coefficient); starts from water's values. Bodies become volumes of it with sim.body.set {fluid}; a fan inside is its pump. One undo step."),
    CommandSpec::new("sim.fluid.delete", "Delete Coolant", fluid_delete).doc_only().params("study: study name, name: fluid to remove (bodies naming it are untagged)."),
    CommandSpec::new("sim.solve", "Solve Simulation", solve).noundo().at("SIMULATION", "SOLVE").icon("box")
        .params("study: study name, flow?: true to solve airflow with default flow settings, false to keep the air still (overrides the study for this solve). Voxelises the visible bodies, solves airflow (if asked) and the temperature field; keeps the results for sim.results, sim.probe, sim.export and sim.display. Returns the summary: per-body min/max/mean °C, air temperatures, heat balance, convergence. Does not change the design. See docs/simulation.md"),
    CommandSpec::new("sim.results", "Simulation Results", results).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("No parameters. The summary of the last solve, with `stale: true` when the geometry changed since."),
    CommandSpec::new("sim.probe", "Probe Simulation", probe).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("point: [x, y, z] world mm. The temperature (°C) and, when airflow was solved, the air velocity (m/s) of the cell there."),
    CommandSpec::new("sim.export", "Export Simulation (VTK)", export).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("path: destination .vtk file (legacy VTK structured points: body, temperature, velocity, pressure per cell; opens in ParaView), overwrite?: false."),
    CommandSpec::new("sim.display", "Show Simulation", display).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("mode?: off | temperature (colour map on the bodies) | velocity (air arrows and the colour map) | streamlines (lines through the air from the fans and inflow openings, coloured by speed, and the colour map), seeds?: streamline seeds 1..=512 (default 96), setup?: draw the study's setup glyphs (fans, heat sources, fixed temperatures, the domain) in pictures (default true)."),
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
    /// Absent: keep; `null`: clear; name: set.
    #[serde(default, deserialize_with = "present")]
    fluid: Option<Value>,
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
    match &request.fluid {
        None => {}
        Some(Value::Null) => setup.fluid = None,
        Some(Value::String(name)) if !name.is_empty() && name.len() <= 64 => setup.fluid = Some(name.clone()),
        Some(_) => return Err(bad(cmd, "`fluid` must be a fluid name or null")),
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
struct FluidSet {
    study: String,
    name: String,
    #[serde(default)]
    set: Option<Value>,
}

fn fluid_set(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.fluid.set";
    let request: FluidSet = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    SimulationSettings::validate_name(&request.study).map_err(|e| bad(cmd, e.to_string()))?;
    if request.name.is_empty() || request.name.len() > 64 || request.name.chars().any(char::is_control) {
        return Err(bad(cmd, "`name` must be a fluid name of at most 64 bytes"));
    }
    let mut settings = settings_mut(s);
    let study = settings.studies.entry(request.study.clone()).or_default();
    let current = study.fluids.get(&request.name).cloned().unwrap_or_else(solidcraft_sim::Fluid::water);
    let mut value = serde_json::to_value(&current).map_err(|e| bad(cmd, e.to_string()))?;
    if let Some(set) = &request.set {
        let patch = set.as_object().filter(|p| p.len() <= 8).ok_or_else(|| bad(cmd, "`set` must be an object with at most 8 fields"))?;
        let object = value.as_object_mut().ok_or_else(|| bad(cmd, "invalid fluid"))?;
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }
    let fluid: solidcraft_sim::Fluid = serde_json::from_value(value).map_err(|e| bad(cmd, e.to_string()))?;
    study.fluids.insert(request.name.clone(), fluid.clone());
    store(s, cmd, settings)?;
    Ok(json!({"study": request.study, "name": request.name, "fluid": fluid}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FluidDelete {
    study: String,
    name: String,
}

fn fluid_delete(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.fluid.delete";
    let request: FluidDelete = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    let mut settings = settings_mut(s);
    let study = settings.studies.get_mut(&request.study).ok_or_else(|| bad(cmd, "no simulation study with that name"))?;
    if study.fluids.remove(&request.name).is_none() {
        return Err(bad(cmd, "no fluid with that name"));
    }
    for b in study.bodies.values_mut() {
        if b.fluid.as_deref() == Some(request.name.as_str()) {
            b.fluid = None;
        }
    }
    store(s, cmd, settings)?;
    Ok(json!({"study": request.study, "name": request.name, "deleted": true}))
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
        Ok(SimulationRun { study: self.study, revision: self.revision, results, bodies: self.bodies, heat_cache: Default::default() })
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

/// A body's display mesh refined to about the simulation's cell size, with the solved surface
/// temperature at every vertex (NaN where no cell of the body is near).
#[derive(Debug)]
pub struct HeatMesh {
    pub mesh: Arc<solidcraft_geom::Mesh>,
    pub temperatures: Vec<f64>,
}

/// Most triangles a refined body may have; the edge length grows past the cell size to stay under.
const MAX_HEAT_TRIANGLES: usize = 400_000;

/// `mesh` with every triangle split uniformly until its edges are at most `target` long (each
/// triangle into k² by its longest edge), normals interpolated; B-rep edges and faces kept.
pub fn refine(mesh: &solidcraft_geom::Mesh, target: f64, max_triangles: usize) -> solidcraft_geom::Mesh {
    let tri = |t: &[u32; 3]| t.map(|k| mesh.positions.get(k as usize).copied().unwrap_or_default());
    let splits = |t: &[u32; 3], len: f64| -> usize {
        let [a, b, c] = tri(t);
        let longest = a.dist(b).max(b.dist(c)).max(c.dist(a));
        if !(longest.is_finite() && len > 0.0) { 1 } else { ((longest / len).ceil() as usize).clamp(1, 256) }
    };
    let mut len = target.max(1e-6);
    for _ in 0..32 {
        let total: usize = mesh.triangles.iter().map(|t| splits(t, len).pow(2)).sum();
        if total <= max_triangles.max(mesh.triangles.len()) {
            break;
        }
        len *= ((total as f64) / (max_triangles as f64)).sqrt().max(1.1);
    }
    let mut out = solidcraft_geom::Mesh {
        edges: mesh.edges.clone(),
        seams: mesh.seams.clone(),
        smooth_edges: mesh.smooth_edges.clone(),
        sharp_edges: mesh.sharp_edges.clone(),
        edge_faces: mesh.edge_faces.clone(),
        ..Default::default()
    };
    for (ti, t) in mesh.triangles.iter().enumerate() {
        let k = splits(t, len);
        let [pa, pb, pc] = tri(t);
        let [na, nb, nc] = t.map(|i| mesh.normals.get(i as usize).copied().unwrap_or(Vec3::Z));
        let face = mesh.tri_face.get(ti).copied().unwrap_or(0);
        let base = out.positions.len() as u32;
        // vertex (i, j) at barycentric ((k - i - j)/k, i/k, j/k), rows of decreasing length
        let mut index = Vec::with_capacity((k + 1) * (k + 2) / 2);
        for i in 0..=k {
            for j in 0..=(k - i) {
                let (u, v) = (i as f64 / k as f64, j as f64 / k as f64);
                let w = 1.0 - u - v;
                index.push(out.positions.len() as u32 - base);
                out.positions.push(pa * w + pb * u + pc * v);
                out.normals.push((na * w + nb * u + nc * v).normalized().unwrap_or(na));
            }
        }
        let at = |i: usize, j: usize| -> u32 {
            // row i starts after rows 0..i, which hold (k+1) + k + … + (k-i+2) vertices
            let start = i * (k + 1) - i * (i.saturating_sub(1)) / 2;
            base + index.get(start + j).copied().unwrap_or(0)
        };
        for i in 0..k {
            for j in 0..(k - i) {
                out.triangles.push([at(i, j), at(i + 1, j), at(i, j + 1)]);
                out.tri_face.push(face);
                if j + 1 < k - i {
                    out.triangles.push([at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
                    out.tri_face.push(face);
                }
            }
        }
    }
    out
}

impl SimulationRun {
    /// `mesh` (a body's display mesh) refined to the grid with its surface temperatures, made once
    /// per body and mesh.
    pub fn heat_mesh(&self, body: u32, mesh: &Arc<solidcraft_geom::Mesh>) -> Arc<HeatMesh> {
        let key = (body, Arc::as_ptr(mesh) as usize);
        if let Ok(cache) = self.heat_cache.lock()
            && let Some((_, h)) = cache.get(&key)
        {
            return h.clone();
        }
        let refined = Arc::new(refine(mesh, self.results.grid.cell, MAX_HEAT_TRIANGLES));
        let temperatures = refined.positions.iter().map(|p| self.results.surface_temperature(body, *p).unwrap_or(f64::NAN)).collect();
        let h = Arc::new(HeatMesh { mesh: refined, temperatures });
        if let Ok(mut cache) = self.heat_cache.lock() {
            cache.insert(key, (mesh.clone(), h.clone()));
        }
        h
    }
}

/// A solved body as the display draws it: the refined mesh and a colour per vertex for the
/// temperature scale `lo..hi`.
pub fn heat_display(
    run: &SimulationRun,
    body: u32,
    mesh: &Arc<solidcraft_geom::Mesh>,
    lo: f64,
    hi: f64,
) -> (Arc<solidcraft_geom::Mesh>, Vec<[u8; 3]>) {
    let h = run.heat_mesh(body, mesh);
    let span = (hi - lo).max(1e-9);
    let colors = h.temperatures.iter().map(|t| if t.is_finite() { heat_color((t - lo) / span) } else { [120, 120, 120] }).collect();
    (h.mesh.clone(), colors)
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
    let fluid_list = study.fluid_list();
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
        let fluid = match &setup.fluid {
            Some(f) => Some(fluid_list.iter().position(|(n, _)| n == f).ok_or_else(|| bad(cmd, format!("body '{}': unknown fluid '{f}'", b.name)))?),
            None => None,
        };
        let conductivity = match fluid.and_then(|i| fluid_list.get(i)) {
            Some((_, f)) => f.conductivity,
            None => setup.conductivity.unwrap_or_else(|| s.doc.thermal_conductivity(def)),
        };
        bounds = bounds.union(&mesh.bounds());
        bodies.push(SolidBody {
            name: b.name.clone(),
            positions: mesh.positions.clone(),
            triangles: mesh.triangles.clone(),
            conductivity,
            power_w: setup.power_w,
            fixed_temperature_c: setup.fixed_temperature_c,
            fluid,
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
        fluids: fluid_list.into_iter().map(|(_, f)| f).collect(),
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
    #[serde(default)]
    mode: Option<SimulationDisplay>,
    /// Streamline seeds.
    #[serde(default)]
    seeds: Option<usize>,
    /// Setup glyphs in pictures.
    #[serde(default)]
    setup: Option<bool>,
}

fn display(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.display";
    let request: Display = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    if let Some(n) = request.seeds {
        let max = super::simulation_view::MAX_STREAMLINE_SEEDS;
        if n == 0 || n > max {
            return Err(bad(cmd, format!("`seeds` must be in 1..={max}")));
        }
        s.simulation_view.seeds = n;
    }
    if let Some(on) = request.setup {
        s.simulation_view.setup = on;
    }
    if let Some(mode) = request.mode {
        s.simulation_display = mode;
    }
    Ok(json!({"mode": s.simulation_display, "seeds": s.simulation_view.seeds, "setup": s.simulation_view.setup, "solved": s.simulation.is_some()}))
}

#[cfg(test)]
#[path = "simulation_tests.rs"]
mod tests;
