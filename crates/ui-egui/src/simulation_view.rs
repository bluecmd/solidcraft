//! The model view's side of exploring a simulation: slices, streamlines, probe markers and the
//! study's setup glyphs in the GPU scene, and over it the glyph labels, the pinned probes'
//! readings and the readout under the pointer. Clicking a body or a slice in the Simulation
//! workspace pins a probe there (`sim.probe.pin`). The geometry comes from
//! `solidcraft_engine::cmd::simulation_view`, which headless pictures draw from too.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, vec2};
use serde_json::json;
use solidcraft_engine::cmd::simulation_view::{self as sv, GlyphKind, SetupGlyphs, Slice, SliceAxis, SliceField};
use solidcraft_engine::doc::simulation::Study;
use solidcraft_engine::geom::{Aabb3, Vec3};
use solidcraft_engine::{SimulationDisplay, SimulationRun};

use crate::SolveApp;
use crate::gpu::GpuScene;
use crate::theme::Tokens;
use crate::viewport::{Hit, Proj};

#[cfg(test)]
#[path = "simulation_view_tests.rs"]
mod tests;

/// Glyphs built for one scene key, reused by the labels every frame.
#[derive(Default)]
pub struct Cache {
    key: u64,
    glyphs: Option<Arc<SetupGlyphs>>,
}

/// The shown simulation run (display on).
fn shown(app: &SolveApp) -> Option<Arc<SimulationRun>> {
    app.session.simulation.clone().filter(|_| app.session.simulation_display != SimulationDisplay::Off)
}

/// The study whose setup the view draws: the workspace's draft while the Simulation workspace
/// is open (edits show as they are made), else the shown run's study when its glyphs are on.
fn glyph_study(app: &SolveApp) -> Option<Study> {
    if crate::simulation_workspace::active(app) {
        return Some(app.simulation_workspace.view.draft.clone());
    }
    if shown(app).is_some() && app.session.simulation_view.setup {
        return sv::shown_study(&app.session).map(|(_, s)| s.clone());
    }
    None
}

/// The run that solved the study the glyphs show (for its exact domain).
fn glyph_run(app: &SolveApp) -> Option<Arc<SimulationRun>> {
    let run = app.session.simulation.clone()?;
    if crate::simulation_workspace::active(app) {
        return (run.study == app.simulation_workspace.view.name && !app.simulation_workspace.view.dirty).then_some(run);
    }
    Some(run)
}

/// Everything here that changes the scene: the viewport's scene key.
pub fn hash_scene(app: &SolveApp, h: &mut impl Hasher) {
    app.session.simulation_view.hash_into(h);
    crate::simulation_workspace::active(app).hash(h);
    if let Some(study) = glyph_study(app) {
        // fans, body setups and the domain are what the glyphs draw
        format!("{:?}{:?}{:?}{}", study.fans, study.bodies, study.domain, study.padding_mm).hash(h);
        app.session.revision.hash(h);
    }
}

/// The setup glyphs for the current scene, built once per change.
pub fn glyphs(app: &SolveApp) -> Option<Arc<SetupGlyphs>> {
    let study = glyph_study(app)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_scene(app, &mut h);
    app.session.simulation.as_ref().map(|r| Arc::as_ptr(r) as usize).hash(&mut h);
    let key = h.finish();
    if let Ok(cache) = app.simulation_workspace.glyphs.lock()
        && cache.key == key
        && let Some(g) = &cache.glyphs
    {
        return Some(g.clone());
    }
    let run = glyph_run(app);
    let g = Arc::new(sv::setup_glyphs(&app.session, &study, run.as_deref()));
    if let Ok(mut cache) = app.simulation_workspace.glyphs.lock() {
        *cache = Cache { key, glyphs: Some(g.clone()) };
    }
    Some(g)
}

/// The temperature range the bodies and slices are coloured over.
pub fn temperature_scale(run: &SimulationRun, app: &SolveApp) -> (f64, f64) {
    sv::temperature_scale(run, &app.session.simulation_view)
}

fn c4(c: [u8; 3], a: u8) -> [u8; 4] {
    [c[0], c[1], c[2], a]
}

/// Adds the simulation's view geometry to the viewport's scene: airflow (arrows or
/// streamlines), slices as translucent unlit cell quads, probe markers and the setup glyphs.
pub fn add_to_scene(app: &SolveApp, sc: &mut GpuScene, t_range: Option<(f64, f64)>) {
    let s = &app.session;
    if let (Some(run), Some(range)) = (shown(app), t_range) {
        match s.simulation_display {
            SimulationDisplay::Velocity => crate::simulation_workspace::velocity_arrows(sc, &run),
            SimulationDisplay::Streamlines => {
                let study = s.doc.simulation.studies.get(&run.study);
                for path in sv::streamlines(&run, study, s.simulation_view.seeds) {
                    for (w, c) in path.points.windows(2).zip(&path.colors) {
                        if let (Some(a), Some(b)) = (w.first(), w.get(1)) {
                            sc.line(a.to_f32(), b.to_f32(), c4(*c, 255), 1.8, false);
                        }
                    }
                }
            }
            _ => {}
        }
        for slice in &s.simulation_view.slices {
            let quads = sv::slice_quads(&run, slice, range);
            for (q, c) in quads.corners.iter().zip(&quads.colors) {
                let col = c4(*c, 215);
                let [a, b, cc, d] = q.map(|p| p.to_f32());
                for p in [a, b, cc, a, cc, d] {
                    sc.trans_tri(p, [0.0; 3], col);
                }
            }
        }
        let size = run.results.grid.cell * 2.0;
        for p in &s.simulation_view.probes {
            for (a, b) in sv::probe_marker(p.point, size) {
                sc.line(a.to_f32(), b.to_f32(), [255, 255, 255, 255], 2.2, true);
            }
        }
    }
    if let Some(g) = glyphs(app) {
        for l in &g.lines {
            sc.line(l.a.to_f32(), l.b.to_f32(), c4(l.color, 255), l.width as f32, l.on_top);
        }
        for (b, c, _) in &g.boxes {
            box_faces(sc, b, c4(*c, 46));
        }
    }
}

/// The six faces of a box as unlit translucent triangles.
fn box_faces(sc: &mut GpuScene, b: &Aabb3, col: [u8; 4]) {
    let (lo, hi) = (b.min, b.max);
    let p = |x: bool, y: bool, z: bool| Vec3::new(if x { hi.x } else { lo.x }, if y { hi.y } else { lo.y }, if z { hi.z } else { lo.z }).to_f32();
    let faces = [
        [p(false, false, false), p(false, true, false), p(false, true, true), p(false, false, true)],
        [p(true, false, false), p(true, true, false), p(true, true, true), p(true, false, true)],
        [p(false, false, false), p(true, false, false), p(true, false, true), p(false, false, true)],
        [p(false, true, false), p(true, true, false), p(true, true, true), p(false, true, true)],
        [p(false, false, false), p(true, false, false), p(true, true, false), p(false, true, false)],
        [p(false, false, true), p(true, false, true), p(true, true, true), p(false, true, true)],
    ];
    for [a, b, c, d] in faces {
        for q in [a, b, c, a, c, d] {
            sc.trans_tri(q, [0.0; 3], col);
        }
    }
}

/// A small label in a rounded box: its rectangle.
fn pill(painter: &egui::Painter, at: Pos2, anchor: Align2, text: &str, fg: Color32, accent: Option<Color32>) -> Rect {
    let t = Tokens::get();
    let galley = painter.layout_no_wrap(text.to_string(), FontId::proportional(12.0), fg);
    let size = galley.size() + vec2(10.0, 4.0);
    let r = anchor.anchor_size(at, size);
    painter.rect(r, 4.0, t.overlay, Stroke::new(1.0, accent.unwrap_or(t.border)), egui::StrokeKind::Inside);
    painter.galley(r.min + vec2(5.0, 2.0), galley, fg);
    r
}

fn rgb(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

/// Where a click or the pointer probes: the solved body under the pointer (its surface point)
/// or, nearer or where no body is, the shown slice the pointer is over.
fn probe_target(app: &SolveApp, run: &SimulationRun, proj: &Proj, at: Pos2) -> Option<(Vec3, Option<String>)> {
    let (origin, dir) = proj.ray(at);
    let body = match &app.viewport.hover {
        Some(Hit::Face { body, .. } | Hit::Edge { body, .. } | Hit::Vertex { body, .. }) if run.body_index(body).is_some() => Some(body.clone()),
        _ => None,
    };
    let on_body = body.and_then(|name| {
        let st = app.session.world_state();
        let mesh = st.body(&name)?.mesh();
        let (t, _) = mesh.raycast(origin, dir)?;
        Some((origin + dir * t, t, name))
    });
    let on_slice = sv::slice_hit(run, &app.session.simulation_view, origin, dir);
    match (on_body, on_slice) {
        (Some((p, tb, name)), Some((q, ts, _))) => Some(if ts < tb { (q, None) } else { (p, Some(name)) }),
        (Some((p, _, name)), None) => Some((p, Some(name))),
        (None, Some((q, _, _))) => Some((q, None)),
        (None, None) => None,
    }
}

/// Probing is on: the Simulation workspace with a shown result and no tool running.
fn probing(app: &SolveApp) -> bool {
    crate::simulation_workspace::active(app) && shown(app).is_some() && app.tool.is_none() && app.dialog.is_none()
}

/// A click in the model view: in the Simulation workspace, a fan's label selects the fan in the
/// panel and a click on a solved body or a slice pins a probe there. `true` when it was used.
pub fn click(app: &mut SolveApp, proj: &Proj, at: Pos2) -> bool {
    if !crate::simulation_workspace::active(app) || app.tool.is_some() {
        return false;
    }
    if let Some(i) = fan_label_at(at) {
        app.simulation_workspace.selected_fan = Some(i);
        app.simulation_workspace.reveal_fan = true;
        return true;
    }
    if !probing(app) {
        return false;
    }
    let Some(run) = shown(app) else { return false };
    let Some((point, body)) = probe_target(app, &run, proj, at) else { return false };
    if let Err(e) = app.run("sim.probe.pin", json!({"point": point, "body": body})) {
        app.set_status(e, true);
    }
    true
}

thread_local! {
    /// The fan labels drawn last frame: (rectangle, fan index), for clicks.
    static FAN_LABELS: std::cell::RefCell<Vec<(Rect, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn fan_label_at(at: Pos2) -> Option<usize> {
    FAN_LABELS.with(|l| l.borrow().iter().find(|(r, _)| r.contains(at)).map(|(_, i)| *i))
}

/// Drawn over the model: glyph labels, pinned probes with their readings, and the reading under
/// the pointer.
pub fn overlay(app: &SolveApp, painter: &egui::Painter, proj: &Proj, hover: Option<Pos2>) {
    let t = Tokens::get();
    let mut fans = Vec::new();
    if let Some(g) = glyphs(app) {
        for l in &g.labels {
            let Some(p) = proj.to_screen(l.at) else { continue };
            if !proj.rect.contains(p) {
                continue;
            }
            let (accent, fg) = match l.kind {
                GlyphKind::Fan(i) => {
                    let on = app.simulation_workspace.selected_fan == Some(i);
                    (rgb(sv::FAN_COLOR), if on { rgb(sv::FAN_COLOR) } else { t.text })
                }
                GlyphKind::Heat => (rgb(sv::HEAT_COLOR), t.text),
                GlyphKind::Fixed => (rgb(sv::FIXED_COLOR), t.text),
            };
            let r = pill(painter, p + vec2(0.0, -6.0), Align2::CENTER_BOTTOM, &l.text, fg, Some(accent));
            if let GlyphKind::Fan(i) = l.kind {
                fans.push((r, i));
                crate::scenario::publish_handle(&format!("simulation:fan:{i}"), r.center());
            }
        }
    }
    FAN_LABELS.with(|l| *l.borrow_mut() = fans);
    let Some(run) = shown(app) else { return };
    let air = crate::i18n::tr("sim.air_label");
    for (i, p) in app.session.simulation_view.probes.iter().enumerate() {
        let Some(s) = proj.to_screen(p.point) else { continue };
        if !proj.rect.contains(s) {
            continue;
        }
        painter.circle(s, 4.0, Color32::WHITE, Stroke::new(1.5, Color32::from_gray(20)));
        if let Some(r) = sv::reading(&run, p.point, p.body.as_deref()) {
            let text = format!("{} · {}", i + 1, sv::reading_text(&r, air));
            pill(painter, s + vec2(8.0, -8.0), Align2::LEFT_BOTTOM, &text, t.text, None);
        }
        crate::scenario::publish_handle(&format!("simulation:probe:{i}"), s);
    }
    if !probing(app) {
        return;
    }
    let Some(at) = hover.filter(|p| proj.rect.contains(*p)) else { return };
    if fan_label_at(at).is_some() {
        return;
    }
    let Some((point, body)) = probe_target(app, &run, proj, at) else { return };
    let Some(r) = sv::reading(&run, point, body.as_deref()) else { return };
    let text = sv::reading_text(&r, air);
    let rect = pill(painter, at + vec2(14.0, 10.0), Align2::LEFT_TOP, &text, t.text, Some(t.accent));
    crate::scenario::publish_rect("simulation:readout", rect);
    crate::scenario::publish_count("simulation:readout_c", r.temperature_c);
}

/// The slice planes: one row per axis (on/off, field, position along the domain).
pub fn slice_controls(app: &mut SolveApp, ui: &mut egui::Ui, run: &SimulationRun) {
    let b = run.results.grid.bounds();
    let (lo, hi) = ([b.min.x, b.min.y, b.min.z], [b.max.x, b.max.y, b.max.z]);
    let flow = run.results.flow.is_some();
    for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
        let current = app.session.simulation_view.slice(axis).copied();
        let i = axis.index();
        let (l, h) = (lo.get(i).copied().unwrap_or(0.0), hi.get(i).copied().unwrap_or(0.0));
        let mut next: Option<Option<Slice>> = None;
        ui.horizontal(|ui| {
            let mut on = current.is_some();
            if ui.checkbox(&mut on, axis.label()).changed() {
                next = Some(on.then_some(Slice { axis, position: 0.5 * (l + h), field: SliceField::Temperature }));
            }
            if let Some(mut s) = current {
                let mut changed = ui
                    .add(egui::DragValue::new(&mut s.position).speed(run.results.grid.cell * 0.5).range(l..=h).max_decimals(1).suffix(" mm"))
                    .changed();
                egui::ComboBox::from_id_salt(("sim_slice_field", i))
                    .selected_text(if s.field == SliceField::Speed {
                        crate::i18n::tr("sim.legend_velocity")
                    } else {
                        crate::i18n::tr("sim.legend_temperature")
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui.selectable_value(&mut s.field, SliceField::Temperature, crate::i18n::tr("sim.legend_temperature")).changed();
                        if flow {
                            changed |= ui.selectable_value(&mut s.field, SliceField::Speed, crate::i18n::tr("sim.legend_velocity")).changed();
                        }
                    });
                if changed {
                    next = Some(Some(s));
                }
            }
        });
        match next {
            Some(Some(s)) => {
                let field = if s.field == SliceField::Speed { "speed" } else { "temperature" };
                if let Err(e) = app.run("sim.slice", json!({"axis": axis, "position": s.position, "field": field})) {
                    app.set_status(e, true);
                }
            }
            Some(None) => {
                let _ = app.run("sim.slice", json!({"axis": axis, "off": true}));
            }
            None => {}
        }
    }
}

/// The pinned probes with their readings, each with a delete button.
pub fn probe_controls(app: &mut SolveApp, ui: &mut egui::Ui, run: &SimulationRun) {
    let probes = app.session.simulation_view.probes.clone();
    if probes.is_empty() {
        ui.small(crate::i18n::tr("sim.probes_help"));
        return;
    }
    let air = crate::i18n::tr("sim.air_label");
    let mut remove = None;
    for (i, p) in probes.iter().enumerate() {
        ui.horizontal(|ui| {
            let text = match sv::reading(run, p.point, p.body.as_deref()) {
                Some(r) => format!("{} · {}", i + 1, sv::reading_text(&r, air)),
                None => format!("{} · —", i + 1),
            };
            ui.add(egui::Label::new(text).truncate());
            if ui.small_button(crate::i18n::tr("sim.delete")).clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        let _ = app.run("sim.probe.delete", json!({ "index": i }));
    }
    if ui.small_button(crate::i18n::tr("sim.clear_probes")).clicked() {
        let _ = app.run("sim.probe.delete", json!({ "all": true }));
    }
}

/// The bodies hottest first, each with a swatch of its maximum on the colour scale: clicking a
/// row frames the body and pins a probe at its hottest point.
pub fn body_table(app: &mut SolveApp, ui: &mut egui::Ui, run: &SimulationRun) {
    let (lo, hi) = temperature_scale(run, app);
    let mut rows: Vec<_> = run.results.summary.bodies.iter().filter(|b| b.max_c.is_finite()).collect();
    rows.sort_by(|a, b| b.max_c.partial_cmp(&a.max_c).unwrap_or(std::cmp::Ordering::Equal));
    let mut clicked = None;
    egui::Grid::new("sim_body_table").num_columns(4).spacing([8.0, 2.0]).striped(true).show(ui, |ui| {
        ui.label("");
        ui.small(crate::i18n::tr("sim.body"));
        ui.small(crate::i18n::tr("sim.max_c"));
        ui.small(crate::i18n::tr("sim.mean_c"));
        ui.end_row();
        for b in rows {
            let c = crate::simulation_workspace::heat_color((b.max_c - lo) / (hi - lo).max(1e-9));
            let (r, _) = ui.allocate_exact_size(vec2(12.0, 12.0), egui::Sense::hover());
            ui.painter().rect_filled(r, 2.0, Color32::from_rgb(c[0], c[1], c[2]));
            let row =
                ui.add(egui::Button::new(egui::RichText::new(&b.name)).frame(false).truncate()).on_hover_text(crate::i18n::tr("sim.body_row_help"));
            crate::scenario::publish_handle(&format!("simulation:body:{}", b.name), row.rect.center());
            if row.clicked() {
                clicked = Some(b.name.clone());
            }
            ui.label(format!("{:.1} °C", b.max_c));
            ui.label(format!("{:.1} °C", b.mean_c));
            ui.end_row();
        }
    });
    if let Some(name) = clicked {
        focus_body(app, run, &name);
    }
}

/// Frames a body and pins a probe at its hottest point.
pub fn focus_body(app: &mut SolveApp, run: &SimulationRun, name: &str) {
    let st = app.session.world_state();
    if let Some(b) = st.body(name) {
        app.animate_fit_bounds(b.mesh().bounds());
    }
    if let Some((p, _)) = sv::hottest_point(run, name) {
        let already = app.session.simulation_view.probes.iter().any(|x| x.point.dist(p) < 1e-9);
        if !already && let Err(e) = app.run("sim.probe.pin", json!({"point": p})) {
            app.set_status(e, true);
        }
    }
}

/// The colour scale for air speed in the legend, when a speed slice or streamlines are shown.
pub fn speed_legend(app: &SolveApp, run: &SimulationRun) -> Option<f64> {
    sv::shows_speed(run, &app.session.simulation_view, app.session.simulation_display).then(|| sv::speed_scale(run))
}

/// One line per slice for the legend: "Slice Z = 12.0 mm · temperature".
pub fn slice_lines(app: &SolveApp) -> Vec<String> {
    app.session
        .simulation_view
        .slices
        .iter()
        .map(|s| {
            let field = if s.field == SliceField::Speed { crate::i18n::tr("sim.legend_velocity") } else { crate::i18n::tr("sim.legend_temperature") };
            format!("{} {} = {:.1} mm · {}", crate::i18n::tr("sim.slice"), s.axis.label(), s.position, field)
        })
        .collect()
}
