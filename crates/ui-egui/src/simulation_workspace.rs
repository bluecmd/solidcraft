//! The Simulation workspace: a right-hand panel that edits a study of the design (what heats,
//! what blows, where the box ends), solves it off the window thread, and shows the result as a
//! temperature colour map and air arrows in the ordinary viewport.

use std::sync::{Arc, Mutex};

use serde_json::json;
use solidcraft_engine::SimulationDisplay;
use solidcraft_engine::doc::simulation::{BodySetup, Study, Wall};
use solidcraft_engine::geom::{Aabb3, Vec3};
use solidcraft_engine::sim::{Fan, FlowSettings};

use crate::{SolveApp, theme::Tokens};

#[cfg(test)]
#[path = "simulation_workspace_tests.rs"]
mod tests;

/// Per-document editor state.
#[derive(Clone, Debug)]
pub struct ViewState {
    pub name: String,
    pub draft: Study,
    pub dirty: bool,
    /// The session revision the draft was loaded at.
    revision: u64,
}

impl Default for ViewState {
    fn default() -> Self {
        ViewState { name: "Thermal".into(), draft: Study::default(), dirty: false, revision: u64::MAX }
    }
}

/// A solve running off the window thread.
pub struct Worker {
    #[cfg(not(target_arch = "wasm32"))]
    rx: std::sync::mpsc::Receiver<Result<solidcraft_engine::SimulationRun, String>>,
    progress: Arc<Mutex<(String, usize, usize)>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    pub started: f64,
}

#[derive(Default)]
pub struct State {
    pub view: ViewState,
    pub worker: Option<Worker>,
    /// The fan picked in the model view (its label), opened in the panel.
    pub selected_fan: Option<usize>,
    /// The panel still has to bring the picked fan into view.
    pub reveal_fan: bool,
    /// The setup glyphs the view last built (`simulation_view::glyphs`).
    pub glyphs: Mutex<crate::simulation_view::Cache>,
}

pub fn active(app: &SolveApp) -> bool {
    app.ui.tab == "SIMULATION"
}

/// Enter (or leave) the Simulation workspace. The viewport stays; the toolbar shows the
/// SIMULATION tab and the study panel opens on the right.
pub fn enter(app: &mut SolveApp, simulation: bool) -> Result<(), String> {
    if simulation {
        if app.session.active_sketch.is_some() || app.tool.is_some() || app.dialog.is_some() || app.file_load.busy() {
            return Err("finish the current modelling operation before entering Simulation".into());
        }
        if crate::render_workspace::active(app) {
            crate::render_workspace::enter(app, false)?;
        }
        app.ui.tab = "SIMULATION".into();
        app.simulation_workspace.view.revision = u64::MAX;
        // Open on the study that was solved (or the design's only study) rather than an empty
        // default one, so the panel, the setup glyphs and the results agree.
        let studies = &app.session.doc.simulation.studies;
        let v = &mut app.simulation_workspace.view;
        if !v.dirty && !studies.contains_key(&v.name) {
            let solved = app.session.simulation.as_ref().map(|r| r.study.clone()).filter(|n| studies.contains_key(n));
            if let Some(name) = solved.or_else(|| studies.keys().next().cloned()) {
                v.name = name;
            }
        }
    } else if active(app) {
        app.ui.tab = "SOLID".into();
    }
    Ok(())
}

/// A `sim.*` command started from the toolbar, the palette or a shortcut: the workspace opens on
/// the matching part, and the commands that need no further input run.
pub fn start(app: &mut SolveApp, id: &str) {
    if let Err(e) = enter(app, true) {
        app.set_status(e, true);
        return;
    }
    match id {
        "sim.solve" => solve(app),
        "sim.display" => {
            let next = match app.session.simulation_display {
                SimulationDisplay::Off => SimulationDisplay::Temperature,
                SimulationDisplay::Temperature => SimulationDisplay::Velocity,
                SimulationDisplay::Velocity => SimulationDisplay::Streamlines,
                SimulationDisplay::Streamlines => SimulationDisplay::Off,
            };
            let _ = app.run("sim.display", json!({ "mode": next }));
        }
        // A slice through the middle of the domain, or off again.
        "sim.slice" => {
            let r = if app.session.simulation_view.slices.is_empty() {
                app.run("sim.slice", json!({ "axis": "z" }))
            } else {
                app.run("sim.slice", json!({ "off": true }))
            };
            if let Err(e) = r {
                app.set_status(e, true);
            }
        }
        "sim.export" => export(app),
        _ => {}
    }
}

pub fn reset_document(app: &mut SolveApp) {
    app.simulation_workspace.view = ViewState::default();
    app.simulation_workspace.worker = None;
}

/// Each frame: a finished solve lands in the session; a running one keeps the window awake.
pub fn tick(app: &mut SolveApp, ctx: &egui::Context) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(w) = &app.simulation_workspace.worker {
        match w.rx.try_recv() {
            Ok(Ok(run)) => {
                let summary = run.results.summary.clone();
                app.session.set_simulation(run);
                app.simulation_workspace.worker = None;
                let hottest = summary.bodies.iter().map(|b| b.max_c).fold(f64::NEG_INFINITY, f64::max);
                let text = format!("{}: {:.1} °C", crate::i18n::tr("sim.solved"), hottest);
                app.set_status(text, !summary.thermal_converged);
                ctx.request_repaint();
            }
            Ok(Err(e)) => {
                app.simulation_workspace.worker = None;
                app.set_status(e, true);
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.simulation_workspace.worker = None;
                app.set_status("the simulation worker stopped", true);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (app, ctx);
}

fn load_draft(app: &mut SolveApp) {
    let v = &mut app.simulation_workspace.view;
    if v.revision != app.session.revision && !v.dirty {
        v.draft = app.session.doc.simulation.studies.get(&v.name).cloned().unwrap_or_default();
        v.revision = app.session.revision;
    }
}

/// Save the draft as the named study (one undo step).
pub fn save(app: &mut SolveApp) -> Result<(), String> {
    let name = app.simulation_workspace.view.name.trim().to_string();
    if name.is_empty() {
        return Err("the study needs a name".into());
    }
    let mut draft = app.simulation_workspace.view.draft.clone();
    // Settings at their defaults need no entry.
    draft.bodies.retain(|_, b| *b != BodySetup::default());
    app.run("sim.study.set", json!({"name": name, "set": draft}))?;
    let v = &mut app.simulation_workspace.view;
    v.name = name;
    v.dirty = false;
    v.revision = app.session.revision;
    Ok(())
}

/// Solve the current study: saved first when edited, then computed off the window thread.
pub fn solve(app: &mut SolveApp) {
    if app.simulation_workspace.worker.is_some() {
        return;
    }
    let unsaved = app.simulation_workspace.view.dirty || !app.session.doc.simulation.studies.contains_key(&app.simulation_workspace.view.name);
    if unsaved && let Err(e) = save(app) {
        app.set_status(e, true);
        return;
    }
    let name = app.simulation_workspace.view.name.clone();
    let Some(backend) = app.session.simulation_accelerator() else {
        app.set_status(solidcraft_engine::cmd::simulation::NO_BACKEND, true);
        return;
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let prepared = match solidcraft_engine::cmd::simulation::prepare(&app.session, &name, None) {
            Ok(p) => p,
            Err(e) => {
                app.set_status(e.to_string(), true);
                return;
            }
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let progress = Arc::new(Mutex::new((String::from("voxelize"), 0usize, 1usize)));
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (p, c) = (progress.clone(), cancel.clone());
        let spawned = std::thread::Builder::new().name("solidcraft-sim".into()).spawn(move || {
            let mut report = |phase: &str, done: usize, total: usize| -> bool {
                if let Ok(mut g) = p.lock() {
                    *g = (phase.to_string(), done, total);
                }
                !c.load(std::sync::atomic::Ordering::Relaxed)
            };
            let _ = tx.send(prepared.solve(&*backend, &mut report).map_err(|e| e.to_string()));
        });
        match spawned {
            Ok(_) => app.simulation_workspace.worker = Some(Worker { rx, progress, cancel, started: crate::now_ms() }),
            Err(e) => app.set_status(format!("could not start the simulation: {e}"), true),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = backend;
        let _ = app.run("sim.solve", json!({ "study": name }));
    }
}

fn export(app: &mut SolveApp) {
    if app.session.simulation.is_none() {
        app.set_status(crate::i18n::tr("sim.none").to_string(), true);
        return;
    }
    app.pick_save_command("simulation.vtk", &["vtk"], "sim.export", json!({"overwrite": true}));
}

/// Colour for a value in 0..=1, with alpha.
pub fn heat_color(t: f64) -> [u8; 4] {
    let c = solidcraft_engine::cmd::simulation::heat_color(t);
    [c[0], c[1], c[2], 255]
}

fn drag(ui: &mut egui::Ui, v: &mut f64, speed: f64, range: std::ops::RangeInclusive<f64>, decimals: usize) -> bool {
    ui.add(egui::DragValue::new(v).speed(speed).range(range).max_decimals(decimals)).changed()
}

pub fn panel(app: &mut SolveApp, ui: &mut egui::Ui) {
    load_draft(app);
    egui::Panel::right("sc_simulation").exact_size(300.0).resizable(false).frame(egui::Frame::NONE.fill(Tokens::get().panel).inner_margin(10)).show(
        ui,
        |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading(app.tr("sim.title"));
                ui.small(app.tr("sim.help"));
                ui.separator();
                study_controls(app, ui);
                ui.separator();
                let mut changed = false;
                ui.collapsing(app.tr("sim.domain"), |ui| changed |= domain_controls(app, ui));
                ui.collapsing(app.tr("sim.air"), |ui| changed |= air_controls(app, ui));
                egui::CollapsingHeader::new(app.tr("sim.bodies"))
                    .default_open(app.session.simulation.is_none())
                    .show(ui, |ui| changed |= body_controls(app, ui));
                let fan_open = app.simulation_workspace.reveal_fan.then_some(true);
                egui::CollapsingHeader::new(app.tr("sim.fans")).open(fan_open).show(ui, |ui| changed |= fan_controls(app, ui));
                ui.collapsing(app.tr("sim.airflow"), |ui| changed |= flow_controls(app, ui));
                if changed {
                    app.simulation_workspace.view.dirty = true;
                }
                ui.separator();
                solve_controls(app, ui);
                ui.separator();
                results_controls(app, ui);
                ui.separator();
                if ui.button(app.tr("sim.return_design")).clicked()
                    && let Err(e) = enter(app, false)
                {
                    app.set_status(e, true);
                }
            });
        },
    );
}

fn study_controls(app: &mut SolveApp, ui: &mut egui::Ui) {
    let names: Vec<String> = app.session.doc.simulation.studies.keys().cloned().collect();
    let old = app.simulation_workspace.view.name.clone();
    ui.label(app.tr("sim.study"));
    egui::ComboBox::from_id_salt("sim_studies").selected_text(&old).show_ui(ui, |ui| {
        for name in names {
            ui.selectable_value(&mut app.simulation_workspace.view.name, name.clone(), name);
        }
    });
    ui.label(app.tr("sim.study_name"));
    ui.add(egui::TextEdit::singleline(&mut app.simulation_workspace.view.name).char_limit(128));
    if app.simulation_workspace.view.name != old {
        let v = &mut app.simulation_workspace.view;
        v.draft = app.session.doc.simulation.studies.get(&v.name).cloned().unwrap_or_default();
        v.dirty = false;
        v.revision = app.session.revision;
    }
}

fn domain_controls(app: &mut SolveApp, ui: &mut egui::Ui) -> bool {
    let d = &mut app.simulation_workspace.view.draft;
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(crate::i18n::tr("sim.cell_size"));
        changed |= drag(ui, &mut d.cell_size_mm, 0.1, 0.0..=1000.0, 2);
    });
    ui.horizontal(|ui| {
        ui.label(crate::i18n::tr("sim.padding"));
        changed |= drag(ui, &mut d.padding_mm, 1.0, 0.0..=100_000.0, 1);
    });
    let mut custom = d.domain.is_some();
    if ui.checkbox(&mut custom, crate::i18n::tr("sim.custom_domain")).changed() {
        d.domain = custom.then(|| Aabb3 { min: Vec3::new(-50.0, -50.0, -50.0), max: Vec3::new(50.0, 50.0, 50.0) });
        changed = true;
    }
    if let Some(b) = &mut d.domain {
        for (label, p) in [("sim.min", &mut b.min), ("sim.max", &mut b.max)] {
            ui.horizontal(|ui| {
                ui.label(crate::i18n::tr(label));
                changed |= drag(ui, &mut p.x, 1.0, -1e6..=1e6, 1);
                changed |= drag(ui, &mut p.y, 1.0, -1e6..=1e6, 1);
                changed |= drag(ui, &mut p.z, 1.0, -1e6..=1e6, 1);
            });
        }
    }
    changed
}

fn air_controls(app: &mut SolveApp, ui: &mut egui::Ui) -> bool {
    let d = &mut app.simulation_workspace.view.draft;
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(crate::i18n::tr("sim.ambient"));
        changed |= drag(ui, &mut d.thermal.ambient_c, 0.5, -100.0..=1000.0, 1);
    });
    ui.horizontal(|ui| {
        ui.label(crate::i18n::tr("sim.film"));
        changed |= drag(ui, &mut d.thermal.film_coefficient, 0.5, 0.0..=10_000.0, 1);
    });
    ui.horizontal(|ui| {
        ui.label(crate::i18n::tr("sim.mixing"));
        changed |= drag(ui, &mut d.air.mixing_factor, 1.0, 0.0..=1e6, 1);
    });
    ui.label(crate::i18n::tr("sim.walls"));
    ui.horizontal_wrapped(|ui| {
        for (i, label) in ["−X", "+X", "−Y", "+Y", "−Z", "+Z"].iter().enumerate() {
            if let Some(w) = d.walls.get_mut(i) {
                let mut open = *w == Wall::Open;
                if ui.checkbox(&mut open, *label).changed() {
                    *w = if open { Wall::Open } else { Wall::Closed };
                    changed = true;
                }
            }
        }
    });
    changed
}

fn body_controls(app: &mut SolveApp, ui: &mut egui::Ui) -> bool {
    let st = app.session.world_state();
    let bodies: Vec<(String, String, f64)> = st
        .bodies
        .iter()
        .filter(|b| app.session.body_visible(&b.name))
        .map(|b| (b.name.clone(), b.definition_name().to_string(), app.session.doc.thermal_conductivity(b.definition_name())))
        .collect();
    if bodies.is_empty() {
        ui.small(crate::i18n::tr("sim.no_bodies"));
        return false;
    }
    let d = &mut app.simulation_workspace.view.draft;
    let mut changed = false;
    for (name, def, material_k) in bodies {
        // An instance takes its definition's settings until it gets its own.
        if !d.bodies.contains_key(&name)
            && let Some(s) = d.bodies.get(&def).cloned()
        {
            d.bodies.insert(name.clone(), s);
        }
        let setup = d.bodies.entry(name.clone()).or_default();
        ui.label(egui::RichText::new(&name).strong());
        ui.horizontal(|ui| {
            ui.label(crate::i18n::tr("sim.power"));
            changed |= drag(ui, &mut setup.power_w, 0.1, 0.0..=1e6, 2);
            let mut exclude = setup.exclude;
            if ui.checkbox(&mut exclude, crate::i18n::tr("sim.exclude")).changed() {
                setup.exclude = exclude;
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            let mut over = setup.conductivity.is_some();
            if ui.checkbox(&mut over, crate::i18n::tr("sim.conductivity")).changed() {
                setup.conductivity = over.then_some(material_k);
                changed = true;
            }
            match &mut setup.conductivity {
                Some(k) => changed |= drag(ui, k, 1.0, 1e-3..=1e4, 2),
                None => {
                    ui.small(format!("{material_k:.2}"));
                }
            }
        });
        ui.horizontal(|ui| {
            let mut fixed = setup.fixed_temperature_c.is_some();
            if ui.checkbox(&mut fixed, crate::i18n::tr("sim.fixed")).changed() {
                setup.fixed_temperature_c = fixed.then_some(d.thermal.ambient_c);
                changed = true;
            }
            if let Some(t) = &mut setup.fixed_temperature_c {
                changed |= drag(ui, t, 0.5, -273.0..=10_000.0, 1);
            }
        });
    }
    changed
}

fn fan_controls(app: &mut SolveApp, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    let mut remove = None;
    let selected = app.simulation_workspace.selected_fan;
    let reveal = std::mem::take(&mut app.simulation_workspace.reveal_fan);
    let d = &mut app.simulation_workspace.view.draft;
    for (i, fan) in d.fans.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            let name = ui.add(egui::TextEdit::singleline(&mut fan.name).char_limit(64).desired_width(100.0));
            if selected == Some(i) {
                // picked in the model view: marked, and brought into view once
                ui.painter().rect_stroke(name.rect.expand(2.0), 3.0, egui::Stroke::new(1.5, Tokens::get().accent), egui::StrokeKind::Outside);
                if reveal {
                    name.scroll_to_me(Some(egui::Align::Center));
                }
            }
            if ui.button(crate::i18n::tr("sim.delete")).clicked() {
                remove = Some(i);
            }
        });
        ui.label(crate::i18n::tr("sim.region"));
        for (label, p) in [("sim.min", &mut fan.region.min), ("sim.max", &mut fan.region.max)] {
            ui.horizontal(|ui| {
                ui.label(crate::i18n::tr(label));
                changed |= drag(ui, &mut p.x, 1.0, -1e6..=1e6, 1);
                changed |= drag(ui, &mut p.y, 1.0, -1e6..=1e6, 1);
                changed |= drag(ui, &mut p.z, 1.0, -1e6..=1e6, 1);
            });
        }
        ui.horizontal(|ui| {
            ui.label(crate::i18n::tr("sim.velocity"));
            changed |= drag(ui, &mut fan.velocity.x, 0.1, -100.0..=100.0, 2);
            changed |= drag(ui, &mut fan.velocity.y, 0.1, -100.0..=100.0, 2);
            changed |= drag(ui, &mut fan.velocity.z, 0.1, -100.0..=100.0, 2);
        });
    }
    if let Some(i) = remove
        && i < d.fans.len()
    {
        d.fans.remove(i);
        changed = true;
    }
    if ui.button(crate::i18n::tr("sim.add_fan")).clicked() && d.fans.len() < 100 {
        let n = d.fans.len() + 1;
        d.fans.push(Fan {
            name: format!("Fan{n}"),
            region: Aabb3 { min: Vec3::new(-10.0, -10.0, -10.0), max: Vec3::new(0.0, 10.0, 10.0) },
            velocity: Vec3::new(1.0, 0.0, 0.0),
        });
        changed = true;
    }
    changed
}

fn flow_controls(app: &mut SolveApp, ui: &mut egui::Ui) -> bool {
    let d = &mut app.simulation_workspace.view.draft;
    let mut changed = false;
    let mut on = d.flow.is_some();
    if ui.checkbox(&mut on, crate::i18n::tr("sim.solve_flow")).changed() {
        d.flow = on.then(FlowSettings::default);
        changed = true;
    }
    if let Some(f) = &mut d.flow {
        ui.horizontal(|ui| {
            ui.label(crate::i18n::tr("sim.max_steps"));
            let mut steps = f.max_steps as f64;
            if drag(ui, &mut steps, 100.0, 100.0..=1e6, 0) {
                f.max_steps = steps as usize;
                changed = true;
            }
        });
        let mut buoyancy = f.buoyancy;
        if ui.checkbox(&mut buoyancy, crate::i18n::tr("sim.buoyancy")).changed() {
            f.buoyancy = buoyancy;
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.label(crate::i18n::tr("sim.cell_reynolds"));
            changed |= drag(ui, &mut f.max_cell_reynolds, 1.0, 1.0..=1000.0, 1);
        });
    }
    changed
}

fn solve_controls(app: &mut SolveApp, ui: &mut egui::Ui) {
    if app.simulation_workspace.view.dirty {
        ui.small(app.tr("sim.unsaved"));
    }
    ui.horizontal(|ui| {
        if ui.button(app.tr("sim.save")).clicked()
            && let Err(e) = save(app)
        {
            app.set_status(e, true);
        }
        let busy = app.simulation_workspace.worker.is_some();
        if ui.add_enabled(!busy, egui::Button::new(app.tr("sim.solve"))).clicked() {
            solve(app);
        }
    });
    if let Some(w) = &app.simulation_workspace.worker {
        let (phase, done, total) = w.progress.lock().map(|g| g.clone()).unwrap_or_default();
        let fraction = if total > 0 { done as f32 / total as f32 } else { 0.0 };
        ui.add(egui::ProgressBar::new(fraction).text(format!("{} {phase}", app.tr("sim.solving"))));
        if ui.button(app.tr("sim.cancel")).clicked() {
            w.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn results_controls(app: &mut SolveApp, ui: &mut egui::Ui) {
    ui.heading(app.tr("sim.results"));
    let Some(run) = app.session.simulation.clone() else {
        ui.small(app.tr("sim.none"));
        return;
    };
    let s = &run.results.summary;
    if run.revision != app.session.revision {
        ui.colored_label(Tokens::get().warning, app.tr("sim.stale"));
    }
    crate::simulation_view::body_table(app, ui, &run);
    ui.label(format!("{}: {:.1}", app.tr("sim.air_max"), s.air_max_c));
    ui.label(format!("{}: {:.2} / {:.2}", app.tr("sim.heat"), s.heat_in_w, s.heat_out_w));
    ui.small(if s.thermal_converged { app.tr("sim.converged") } else { app.tr("sim.not_converged") });
    if let Some(f) = &s.flow {
        ui.label(format!("{}: {} ({:.2} m/s)", app.tr("sim.flow_steps"), f.steps, f.max_speed_m_s));
    }
    ui.label(app.tr("sim.display"));
    let mut mode = app.session.simulation_display;
    ui.horizontal_wrapped(|ui| {
        ui.radio_value(&mut mode, SimulationDisplay::Off, app.tr("sim.display_off"));
        ui.radio_value(&mut mode, SimulationDisplay::Temperature, app.tr("sim.display_temperature"));
        ui.add_enabled_ui(run.results.flow.is_some(), |ui| {
            ui.radio_value(&mut mode, SimulationDisplay::Velocity, app.tr("sim.display_velocity"));
            ui.radio_value(&mut mode, SimulationDisplay::Streamlines, app.tr("sim.display_streamlines"));
        });
    });
    if mode != app.session.simulation_display {
        let _ = app.run("sim.display", json!({ "mode": mode }));
    }
    if app.session.simulation_display == SimulationDisplay::Streamlines {
        ui.horizontal(|ui| {
            ui.label(app.tr("sim.seeds"));
            let mut seeds = app.session.simulation_view.seeds as f64;
            let max = solidcraft_engine::cmd::simulation_view::MAX_STREAMLINE_SEEDS as f64;
            if drag(ui, &mut seeds, 1.0, 1.0..=max, 0) {
                let _ = app.run("sim.display", json!({ "seeds": seeds as usize }));
            }
        });
    }
    ui.label(app.tr("sim.slices"));
    crate::simulation_view::slice_controls(app, ui, &run);
    ui.label(app.tr("sim.probes"));
    crate::simulation_view::probe_controls(app, ui, &run);
    if ui.button(app.tr("sim.export")).clicked() {
        export(app);
    }
}

/// The colour legend drawn over the viewport while a simulation is shown.
pub fn legend(app: &SolveApp, painter: &egui::Painter, rect: egui::Rect) {
    use egui::{Align2, FontId, Rect, Stroke, pos2, vec2};
    let Some(run) = &app.session.simulation else { return };
    if app.session.simulation_display == SimulationDisplay::Off {
        return;
    }
    let t = Tokens::get();
    let (lo, hi) = crate::simulation_view::temperature_scale(run, app);
    let stale = run.revision != app.session.revision;
    let arrows =
        (app.session.simulation_display == SimulationDisplay::Velocity).then(|| run.results.summary.flow.as_ref()).flatten().map(|f| f.max_speed_m_s);
    let flow = arrows.or_else(|| crate::simulation_view::speed_legend(app, run));
    let slices = crate::simulation_view::slice_lines(app);
    let (pad, bar_h, row) = (8.0, 12.0, 18.0);
    let bars = 1 + usize::from(flow.is_some());
    let h = pad * 2.0 + 18.0 + bars as f32 * (row + bar_h + 18.0) + if stale { row } else { 0.0 } + slices.len() as f32 * 15.0;
    let w = 220.0;
    let r = Rect::from_min_size(pos2(rect.right() - 12.0 - w, rect.bottom() - 12.0 - h), vec2(w, h));
    painter.rect(r, 6.0, t.overlay, Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
    crate::scenario::publish_handle("simulation:legend", r.center());
    let mut y = r.top() + pad;
    painter.text(
        pos2(r.left() + pad, y),
        Align2::LEFT_TOP,
        format!("{} · {}", crate::i18n::tr("sim.title"), run.study),
        FontId::proportional(13.0),
        t.text,
    );
    y += 18.0;
    let bar = |title: String, low: String, high: String, y: &mut f32| {
        painter.text(pos2(r.left() + pad, *y), Align2::LEFT_TOP, title, FontId::proportional(11.5), t.text_dim);
        *y += row;
        let bar = Rect::from_min_size(pos2(r.left() + pad, *y), vec2(w - 2.0 * pad, bar_h));
        let n = 32;
        for i in 0..n {
            let c = heat_color(i as f64 / (n - 1) as f64);
            let x0 = bar.left() + bar.width() * i as f32 / n as f32;
            let x1 = bar.left() + bar.width() * (i + 1) as f32 / n as f32;
            painter.rect_filled(Rect::from_min_max(pos2(x0, bar.top()), pos2(x1, bar.bottom())), 0.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
        }
        *y += bar_h + 2.0;
        painter.text(pos2(bar.left(), *y), Align2::LEFT_TOP, low, FontId::proportional(10.5), t.text);
        painter.text(pos2(bar.right(), *y), Align2::RIGHT_TOP, high, FontId::proportional(10.5), t.text);
        *y += 16.0;
    };
    bar(crate::i18n::tr("sim.legend_temperature").to_string(), format!("{lo:.1} °C"), format!("{hi:.1} °C"), &mut y);
    if let Some(f) = flow {
        bar(crate::i18n::tr("sim.legend_velocity").to_string(), "0 m/s".into(), format!("{f:.2} m/s"), &mut y);
    }
    for line in slices {
        painter.text(pos2(r.left() + pad, y), Align2::LEFT_TOP, line, FontId::proportional(10.5), t.text_dim);
        y += 15.0;
    }
    crate::scenario::publish_count("simulation:legend_hi", hi);
    if stale {
        painter.text(pos2(r.left() + pad, y + 2.0), Align2::LEFT_TOP, crate::i18n::tr("sim.stale"), FontId::proportional(10.5), t.warning);
    }
}

pub use solidcraft_engine::cmd::simulation::body_range;

/// Per-vertex colours of a body's drawn mesh from the solved temperatures.
pub fn vertex_colors(run: &solidcraft_engine::SimulationRun, body: u32, mesh: &solidcraft_engine::geom::Mesh, lo: f64, hi: f64) -> Vec<[u8; 4]> {
    solidcraft_engine::cmd::simulation::vertex_colors(run, body, mesh, lo, hi).into_iter().map(|c| [c[0], c[1], c[2], 255]).collect()
}

/// Air velocity arrows through the air cells of the shown simulation.
pub fn velocity_arrows(sc: &mut crate::gpu::GpuScene, run: &solidcraft_engine::SimulationRun) {
    for seg in solidcraft_engine::cmd::simulation::velocity_arrows(run) {
        sc.line(seg.a.to_f32(), seg.b.to_f32(), [seg.color[0], seg.color[1], seg.color[2], 255], seg.width as f32, false);
    }
}
