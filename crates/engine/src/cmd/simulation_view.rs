//! Exploring a solved simulation in the model view: slice planes through the fields, streamlines
//! through the air, probes pinned to points, and glyphs for the study's setup (fans, heat
//! sources, fixed temperatures, the domain box).
//!
//! All of it is view state (`Session::simulation_view`): never part of the design and never an
//! undo step. The geometry is built here once for both the viewport (`solidcraft-ui-egui`) and
//! headless pictures (`view.rs`), so the two always agree.

use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use solidcraft_doc::simulation::Study;
use solidcraft_geom::{Aabb3, Vec3};
use solidcraft_sim::{FLUID, StreamlineOptions, Wall};

use super::CommandSpec;
use super::simulation::{body_range, heat_color};
use crate::params::bad;
use crate::{Result, Session, SimulationDisplay, SimulationRun};

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("sim.slice", "Slice Plane", slice).noundo().doc_only().at("SIMULATION", "RESULTS").icon("box")
        .params("axis?: x | y | z (null removes every slice), position?: mm along the axis (default: the slice's last position, else the domain's middle), field?: temperature | speed (default temperature), off?: true removes the slice on `axis` (or every slice without one). At most one slice per axis. Without parameters lists the slices. Colours the cells the plane cuts, air and solids alike. View state: no undo step. See docs/simulation.md"),
    CommandSpec::new("sim.probe.pin", "Pin Probe", probe_pin).noundo().doc_only()
        .params("point: [x, y, z] world mm inside the solved domain, body?: the body whose surface the point is on (reads that body's surface temperature, as the colour map does). Pins a probe the view labels with its reading (at most 64). Returns the probe's index and reading."),
    CommandSpec::new("sim.probe.delete", "Delete Probe", probe_delete).noundo().doc_only()
        .params("index?: probe to remove (from sim.probe.list), all?: true removes every probe."),
    CommandSpec::new("sim.probe.list", "Pinned Probes", probe_list).noundo().doc_only()
        .params("No parameters. The pinned probes with their readings from the last solve."),
];

/// Most pinned probes.
pub const MAX_PROBES: usize = 64;
/// Most quads a slice draws: a finer grid is sampled every few cells.
pub const MAX_SLICE_QUADS: usize = 65_536;
/// Most streamline seeds `sim.display` accepts.
pub const MAX_STREAMLINE_SEEDS: usize = 512;
/// Most points over all streamlines of a picture.
pub const MAX_STREAMLINE_POINTS: usize = 200_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

impl SliceAxis {
    pub fn index(self) -> usize {
        match self {
            SliceAxis::X => 0,
            SliceAxis::Y => 1,
            SliceAxis::Z => 2,
        }
    }
    pub fn unit(self) -> Vec3 {
        match self {
            SliceAxis::X => Vec3::X,
            SliceAxis::Y => Vec3::Y,
            SliceAxis::Z => Vec3::Z,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            SliceAxis::X => "X",
            SliceAxis::Y => "Y",
            SliceAxis::Z => "Z",
        }
    }
}

/// The field a slice shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceField {
    #[default]
    Temperature,
    /// Air speed, m/s (solids drawn grey).
    Speed,
}

/// A plane across the domain at `position` mm along `axis`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Slice {
    pub axis: SliceAxis,
    pub position: f64,
    pub field: SliceField,
}

/// A point the view labels with its reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PinnedProbe {
    pub point: Vec3,
    /// The body whose surface the point is on (its surface temperature is read).
    pub body: Option<String>,
}

/// How the model view explores the last simulation. View state, kept per document.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationView {
    /// At most one per axis, in axis order.
    pub slices: Vec<Slice>,
    pub probes: Vec<PinnedProbe>,
    /// Streamline seeds (`sim.display {mode: streamlines, seeds}`).
    pub seeds: usize,
    /// Draw the setup glyphs (fans, heat sources, fixed temperatures, the domain) in pictures.
    pub setup: bool,
}

impl Default for SimulationView {
    fn default() -> Self {
        SimulationView { slices: Vec::new(), probes: Vec::new(), seeds: 96, setup: true }
    }
}

impl SimulationView {
    /// Feeds everything that changes the drawn geometry into a hasher (the viewport's scene key).
    pub fn hash_into(&self, h: &mut impl Hasher) {
        for s in &self.slices {
            (s.axis, s.field, s.position.to_bits()).hash(h);
        }
        self.slices.len().hash(h);
        for p in &self.probes {
            (p.point.x.to_bits(), p.point.y.to_bits(), p.point.z.to_bits(), &p.body).hash(h);
        }
        self.probes.len().hash(h);
        (self.seeds, self.setup).hash(h);
    }
    pub fn slice(&self, axis: SliceAxis) -> Option<&Slice> {
        self.slices.iter().find(|s| s.axis == axis)
    }
    fn has_field(&self, field: SliceField) -> bool {
        self.slices.iter().any(|s| s.field == field)
    }
}

/// The temperature range the colour map spans: the bodies' surfaces, widened to the whole grid
/// (the air included) while a temperature slice is shown, so one legend reads the bodies and
/// the slice alike.
pub fn temperature_scale(run: &SimulationRun, view: &SimulationView) -> (f64, f64) {
    let (lo, hi) = body_range(run);
    if !view.has_field(SliceField::Temperature) {
        return (lo, hi);
    }
    let (glo, ghi) = run.results.temperature_range();
    let (lo, hi) = (if glo.is_finite() { lo.min(glo) } else { lo }, if ghi.is_finite() { hi.max(ghi) } else { hi });
    if hi - lo < 0.1 { (lo, lo + 0.1) } else { (lo, hi) }
}

/// The air speed range the speed colours span, m/s: 0 to the fastest cell.
pub fn speed_scale(run: &SimulationRun) -> f64 {
    let max = run.results.max_speed();
    if max > 1e-9 { max } else { 1.0 }
}

/// Whether the shown picture uses the air-speed colour scale (a speed slice or streamlines).
pub fn shows_speed(run: &SimulationRun, view: &SimulationView, display: SimulationDisplay) -> bool {
    run.results.flow.is_some() && (display == SimulationDisplay::Streamlines || view.has_field(SliceField::Speed))
}

fn scaled(v: f64, lo: f64, hi: f64) -> [u8; 3] {
    heat_color((v - lo) / (hi - lo).max(1e-12))
}

/// Grey for solid cells on a speed slice.
pub const SOLID_GREY: [u8; 3] = [96, 98, 104];

/// A slice as coloured quads: four corners each (counter-clockwise seen from +axis), one colour each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SliceQuads {
    pub corners: Vec<[Vec3; 4]>,
    pub colors: Vec<[u8; 3]>,
    /// Cells per quad side (1 unless the slice was too fine to draw every cell).
    pub stride: usize,
}

/// The cells a slice cuts, as quads in the slice plane coloured by the slice's field: per cell
/// (sampled every `stride` cells when the plane has more than [`MAX_SLICE_QUADS`] cells). Empty
/// when the plane misses the grid or the field was not solved.
pub fn slice_quads(run: &SimulationRun, slice: &Slice, t_range: (f64, f64)) -> SliceQuads {
    let r = &run.results;
    let g = &r.grid;
    let mut out = SliceQuads { stride: 1, ..Default::default() };
    if slice.field == SliceField::Speed && r.flow.is_none() {
        return out;
    }
    let a = slice.axis.index();
    let (u, w) = ((a + 1) % 3, (a + 2) % 3);
    let n = [g.nx, g.ny, g.nz];
    let origin = [g.origin.x, g.origin.y, g.origin.z];
    let (Some(&na), Some(&nu), Some(&nw), Some(&oa)) = (n.get(a), n.get(u), n.get(w), origin.get(a)) else { return out };
    if !slice.position.is_finite() {
        return out;
    }
    let f = (slice.position - oa) / g.cell;
    if f < 0.0 || f > na as f64 {
        return out;
    }
    let layer = (f.floor() as usize).min(na.saturating_sub(1));
    let cells = nu.saturating_mul(nw);
    let stride = if cells > MAX_SLICE_QUADS { ((cells as f64 / MAX_SLICE_QUADS as f64).sqrt().ceil() as usize).max(1) } else { 1 };
    out.stride = stride;
    let max_speed = speed_scale(run);
    let (lo, hi) = t_range;
    let corner = |cu: usize, cw: usize| -> Vec3 {
        let mut p = [0.0f64; 3];
        if let Some(x) = p.get_mut(a) {
            *x = slice.position;
        }
        if let Some(x) = p.get_mut(u) {
            *x = origin.get(u).copied().unwrap_or(0.0) + cu.min(nu) as f64 * g.cell;
        }
        if let Some(x) = p.get_mut(w) {
            *x = origin.get(w).copied().unwrap_or(0.0) + cw.min(nw) as f64 * g.cell;
        }
        Vec3::from(p)
    };
    for bw in (0..nw).step_by(stride) {
        for bu in (0..nu).step_by(stride) {
            // the block's middle cell stands for the block
            let (su, sw) = ((bu + stride / 2).min(nu - 1), (bw + stride / 2).min(nw - 1));
            let mut ijk = [0usize; 3];
            if let Some(x) = ijk.get_mut(a) {
                *x = layer;
            }
            if let Some(x) = ijk.get_mut(u) {
                *x = su;
            }
            if let Some(x) = ijk.get_mut(w) {
                *x = sw;
            }
            let idx = g.index(ijk);
            let color = match slice.field {
                SliceField::Temperature => match r.thermal.temperature.get(idx) {
                    Some(t) => scaled(*t, lo, hi),
                    None => continue,
                },
                SliceField::Speed => {
                    if r.voxels.label.get(idx).is_some_and(|l| *l != FLUID) {
                        SOLID_GREY
                    } else {
                        scaled(r.cell_speed(idx), 0.0, max_speed)
                    }
                }
            };
            let (u1, w1) = (bu + stride, bw + stride);
            out.corners.push([corner(bu, bw), corner(u1, bw), corner(u1, w1), corner(bu, w1)]);
            out.colors.push(color);
        }
    }
    out
}

/// Where a ray first meets a shown slice inside the grid: (point, distance along the ray, slice).
pub fn slice_hit(run: &SimulationRun, view: &SimulationView, origin: Vec3, dir: Vec3) -> Option<(Vec3, f64, Slice)> {
    let b = run.results.grid.bounds();
    let mut best: Option<(Vec3, f64, Slice)> = None;
    for s in &view.slices {
        if s.field == SliceField::Speed && run.results.flow.is_none() {
            continue;
        }
        let n = s.axis.unit();
        let d = dir.dot(n);
        if d.abs() < 1e-12 {
            continue;
        }
        let t = (s.position - origin.dot(n)) / d;
        if !t.is_finite() || t < 0.0 {
            continue;
        }
        let p = origin + dir * t;
        let eps = run.results.grid.cell * 1e-6;
        let inside = p.x >= b.min.x - eps
            && p.y >= b.min.y - eps
            && p.z >= b.min.z - eps
            && p.x <= b.max.x + eps
            && p.y <= b.max.y + eps
            && p.z <= b.max.z + eps;
        if inside && best.as_ref().is_none_or(|(_, bt, _)| t < *bt) {
            best = Some((p, t, *s));
        }
    }
    best
}

/// What a probe reads at a point.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reading {
    pub point: Vec3,
    /// The body the point is in or on; `None` in the air.
    pub body: Option<String>,
    pub temperature_c: f64,
    /// Air velocity, m/s, when airflow was solved.
    pub velocity_m_s: Option<Vec3>,
    pub speed_m_s: Option<f64>,
}

/// The reading at `point`: a body's surface temperature when `body` names a solved body (as its
/// colour map shows it), else the temperature of the cell there; the air velocity of the cell.
/// `None` outside the solved domain.
pub fn reading(run: &SimulationRun, point: Vec3, body: Option<&str>) -> Option<Reading> {
    let r = &run.results;
    let cell = r.grid.cell_of(point)?;
    let idx = r.grid.index(cell);
    let surface = body.and_then(|b| run.body_index(b).map(|i| (b, i)));
    let (body, temperature) = match surface {
        Some((name, i)) => (Some(name.to_string()), r.surface_temperature(i, point).or_else(|| r.temperature_at(point))?),
        None => {
            let label = r.voxels.label.get(idx).copied().unwrap_or(FLUID);
            let name = (label != FLUID).then(|| run.bodies.get(label as usize).cloned()).flatten();
            (name, r.temperature_at(point)?)
        }
    };
    let velocity = r.velocity_at(point);
    Some(Reading { point, body, temperature_c: temperature, velocity_m_s: velocity, speed_m_s: velocity.map(|v| v.len()) })
}

/// The hottest cell of a solved body: its centre and temperature.
pub fn hottest_point(run: &SimulationRun, body: &str) -> Option<(Vec3, f64)> {
    let label = run.body_index(body)?;
    let r = &run.results;
    let mut best: Option<(usize, f64)> = None;
    for (idx, l) in r.voxels.label.iter().enumerate() {
        if *l == label
            && let Some(&t) = r.thermal.temperature.get(idx)
            && t.is_finite()
            && best.is_none_or(|(_, bt)| t > bt)
        {
            best = Some((idx, t));
        }
    }
    best.map(|(idx, t)| (r.grid.centre(r.grid.coords(idx)), t))
}

/// One traced streamline with a colour per point (air speed on the speed scale).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamlinePath {
    pub points: Vec<Vec3>,
    pub colors: Vec<[u8; 3]>,
}

/// Streamlines through the solved air: seeded across each fan (traced both ways, so the line
/// shows where the fan draws air from and where it blows it) and where air enters through open
/// walls (traced downstream). `seeds` (capped at [`MAX_STREAMLINE_SEEDS`]) is shared between the
/// two; all lines together hold at most [`MAX_STREAMLINE_POINTS`] points.
pub fn streamlines(run: &SimulationRun, study: Option<&Study>, seeds: usize) -> Vec<StreamlinePath> {
    let r = &run.results;
    let mut out = Vec::new();
    if r.flow.is_none() {
        return out;
    }
    let seeds = seeds.clamp(1, MAX_STREAMLINE_SEEDS);
    let max_speed = speed_scale(run);
    let fans = study.map(|s| s.fans.as_slice()).unwrap_or_default();
    let walls = study.map_or([Wall::Open; 6], |s| s.walls);
    let mut starts: Vec<(Vec3, bool)> = Vec::new();
    let fan_budget = if fans.is_empty() { 0 } else { seeds / 2 };
    if let Some(per_fan) = fan_budget.checked_div(fans.len()) {
        for fan in fans {
            starts.extend(solidcraft_sim::fan_seeds(fan, per_fan.max(1)).into_iter().map(|p| (p, true)));
        }
    }
    let rest = seeds.saturating_sub(starts.len()).max(1);
    starts.extend(r.inflow_seeds(&walls, rest, 0.05 * max_speed).into_iter().map(|p| (p, false)));
    let n = (r.grid.nx + r.grid.ny + r.grid.nz) as f64;
    let options =
        StreamlineOptions { step_cells: 0.5, max_steps: (n * 6.0) as usize, max_length_mm: 0.0, min_speed: 0.01 * max_speed, both_ways: false };
    let mut total = 0usize;
    for (seed, both) in starts {
        if total >= MAX_STREAMLINE_POINTS {
            break;
        }
        let line = r.streamline(seed, &StreamlineOptions { both_ways: both, ..options.clone() });
        if line.points.len() < 2 {
            continue;
        }
        let keep = line.points.len().min(MAX_STREAMLINE_POINTS - total);
        total += keep;
        out.push(StreamlinePath {
            points: line.points.into_iter().take(keep).collect(),
            colors: line.speeds.into_iter().take(keep).map(|s| scaled(s, 0.0, max_speed)).collect(),
        });
    }
    out
}

/// A label the viewport writes beside a setup glyph or a probe.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphLabel {
    pub at: Vec3,
    pub text: String,
    pub kind: GlyphKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlyphKind {
    /// A fan (its index in the study).
    Fan(usize),
    /// A heat source body.
    Heat,
    /// A body held at a fixed temperature.
    Fixed,
}

/// A line of a glyph: ends, colour and width (px).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphLine {
    pub a: Vec3,
    pub b: Vec3,
    pub color: [u8; 3],
    pub width: f64,
    /// Drawn over the model (a fan's arrow stays readable through bodies and slices).
    pub on_top: bool,
}

/// The study's setup in the model view: fan boxes with an arrow along their air, badges on heat
/// sources and fixed-temperature bodies, and the domain box as a dashed outline.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SetupGlyphs {
    pub lines: Vec<GlyphLine>,
    /// Fan regions to fill translucently: (box, colour, fan index).
    pub boxes: Vec<(Aabb3, [u8; 3], usize)>,
    pub labels: Vec<GlyphLabel>,
    pub domain: Option<Aabb3>,
}

pub const FAN_COLOR: [u8; 3] = [40, 170, 235];
pub const FAN_ARROW: [u8; 3] = [120, 220, 255];
pub const DOMAIN_COLOR: [u8; 3] = [150, 152, 165];
pub const HEAT_COLOR: [u8; 3] = [235, 120, 40];
pub const FIXED_COLOR: [u8; 3] = [90, 170, 240];

/// The twelve edges of a box.
pub fn box_edges(b: &Aabb3) -> [(Vec3, Vec3); 12] {
    let (lo, hi) = (b.min, b.max);
    let p = |x: bool, y: bool, z: bool| Vec3::new(if x { hi.x } else { lo.x }, if y { hi.y } else { lo.y }, if z { hi.z } else { lo.z });
    [
        (p(false, false, false), p(true, false, false)),
        (p(false, true, false), p(true, true, false)),
        (p(false, false, true), p(true, false, true)),
        (p(false, true, true), p(true, true, true)),
        (p(false, false, false), p(false, true, false)),
        (p(true, false, false), p(true, true, false)),
        (p(false, false, true), p(false, true, true)),
        (p(true, false, true), p(true, true, true)),
        (p(false, false, false), p(false, false, true)),
        (p(true, false, false), p(true, false, true)),
        (p(false, true, false), p(false, true, true)),
        (p(true, true, false), p(true, true, true)),
    ]
}

/// An arrow from `from` along `dir` (unit) of length `len`: a shaft and four head strokes.
fn arrow(from: Vec3, dir: Vec3, len: f64, color: [u8; 3], width: f64, out: &mut Vec<GlyphLine>) {
    let tip = from + dir * len;
    out.push(GlyphLine { a: from, b: tip, color, width, on_top: true });
    let side = dir.any_perp();
    let other = dir.cross(side);
    let head = len * 0.28;
    for s in [side, side * -1.0, other, other * -1.0] {
        out.push(GlyphLine { a: tip, b: tip - dir * head + s * (head * 0.5), color, width, on_top: true });
    }
}

/// The domain a study simulates: its own box, or the solved grid's when `run` solved it, or the
/// visible bodies' bounds grown by the padding (as `sim.solve` takes it).
pub fn study_domain(s: &Session, study: &Study, run: Option<&SimulationRun>) -> Option<Aabb3> {
    if let Some(d) = study.domain {
        return Some(d);
    }
    if let Some(run) = run {
        return Some(run.results.grid.bounds());
    }
    let st = s.world_state();
    let mut bounds = Aabb3::EMPTY;
    for b in st.bodies.iter().filter(|b| s.body_visible(&b.name)) {
        let excluded = study.bodies.get(b.name.as_str()).or_else(|| study.bodies.get(b.definition_name())).is_some_and(|x| x.exclude);
        if !excluded {
            bounds = bounds.union(&b.mesh().bounds());
        }
    }
    if bounds.is_empty() {
        return None;
    }
    let pad = Vec3::new(study.padding_mm, study.padding_mm, study.padding_mm);
    Some(Aabb3 { min: bounds.min - pad, max: bounds.max + pad })
}

fn short(x: f64) -> String {
    let t = format!("{x:.1}");
    t.strip_suffix(".0").map(str::to_string).unwrap_or(t)
}

/// Glyphs for `study`'s setup over the visible bodies; `run` (when it solved this study) gives
/// the exact domain.
pub fn setup_glyphs(s: &Session, study: &Study, run: Option<&SimulationRun>) -> SetupGlyphs {
    let mut g = SetupGlyphs::default();
    for (i, fan) in study.fans.iter().enumerate() {
        let r = fan.region;
        if r.is_empty() || !r.min.is_finite() || !r.max.is_finite() || r.diagonal() > 1e7 {
            continue;
        }
        for (a, b) in box_edges(&r) {
            g.lines.push(GlyphLine { a, b, color: FAN_COLOR, width: 1.6, on_top: false });
        }
        g.boxes.push((r, FAN_COLOR, i));
        let size = r.size();
        let len = (size.x.max(size.y).max(size.z) * 0.3).max(4.0);
        if let Some(dir) = fan.velocity.normalized() {
            arrow(r.center(), dir, len, FAN_ARROW, 3.0, &mut g.lines);
        }
        let speed = fan.velocity.len();
        g.labels.push(GlyphLabel { at: r.center(), text: format!("{} · {} m/s", fan.name, short(speed)), kind: GlyphKind::Fan(i) });
    }
    let st = s.world_state();
    for b in st.bodies.iter().filter(|b| s.body_visible(&b.name)) {
        let Some(setup) = study.bodies.get(b.name.as_str()).or_else(|| study.bodies.get(b.definition_name())) else { continue };
        if setup.exclude {
            continue;
        }
        let bounds = b.mesh().bounds();
        if bounds.is_empty() {
            continue;
        }
        let at = Vec3::new(bounds.center().x, bounds.center().y, bounds.max.z);
        if let Some(t) = setup.fixed_temperature_c {
            g.labels.push(GlyphLabel { at, text: format!("❄ {} °C", short(t)), kind: GlyphKind::Fixed });
        } else if setup.power_w > 0.0 {
            g.labels.push(GlyphLabel { at, text: format!("⚡ {} W", short(setup.power_w)), kind: GlyphKind::Heat });
        }
    }
    if let Some(d) = study_domain(s, study, run) {
        g.domain = Some(d);
        let dash = (d.diagonal() / 120.0).max(0.5);
        for (a, b) in box_edges(&d) {
            let len = a.dist(b);
            let Some(dir) = (b - a).normalized() else { continue };
            let n = ((len / (2.0 * dash)).floor() as usize).clamp(1, 400);
            let step = len / n as f64;
            for k in 0..n {
                let p = a + dir * (k as f64 * step);
                g.lines.push(GlyphLine { a: p, b: p + dir * (step * 0.55), color: DOMAIN_COLOR, width: 1.0, on_top: false });
            }
        }
    }
    g
}

/// The study a picture shows the setup of: the solved one, else the only one.
pub fn shown_study(s: &Session) -> Option<(&str, &Study)> {
    let studies = &s.doc.simulation.studies;
    if let Some(run) = &s.simulation
        && let Some((k, v)) = studies.get_key_value(run.study.as_str())
    {
        return Some((k.as_str(), v));
    }
    if studies.len() == 1 {
        return studies.iter().next().map(|(k, v)| (k.as_str(), v));
    }
    None
}

/// A small 3D cross marking a pinned probe, `size` mm across.
pub fn probe_marker(p: Vec3, size: f64) -> [(Vec3, Vec3); 3] {
    let h = size * 0.5;
    [(p - Vec3::X * h, p + Vec3::X * h), (p - Vec3::Y * h, p + Vec3::Y * h), (p - Vec3::Z * h, p + Vec3::Z * h)]
}

/// The text a label shows for a reading: "Chip · 63.4 °C" (with the air speed in the air).
pub fn reading_text(r: &Reading, air: &str) -> String {
    let name = r.body.clone().unwrap_or_else(|| air.to_string());
    match (r.body.is_none(), r.speed_m_s) {
        (true, Some(v)) => format!("{name} · {:.1} °C · {:.2} m/s", r.temperature_c, v),
        _ => format!("{name} · {:.1} °C", r.temperature_c),
    }
}

// ---- commands ----

fn axis_of(cmd: &str, v: &Value) -> Result<Option<SliceAxis>> {
    match v {
        Value::Null => Ok(None),
        Value::String(t) => match t.to_ascii_lowercase().as_str() {
            "x" => Ok(Some(SliceAxis::X)),
            "y" => Ok(Some(SliceAxis::Y)),
            "z" => Ok(Some(SliceAxis::Z)),
            _ => Err(bad(cmd, "`axis` must be x, y, z or null")),
        },
        _ => Err(bad(cmd, "`axis` must be x, y, z or null")),
    }
}

fn slices_json(s: &Session) -> Value {
    json!({"slices": s.simulation_view.slices, "solved": s.simulation.is_some()})
}

fn slice(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.slice";
    let p = params.as_object().ok_or_else(|| bad(cmd, "expected an object"))?;
    if let Some(k) = p.keys().find(|k| !matches!(k.as_str(), "axis" | "position" | "field" | "off")) {
        return Err(bad(cmd, format!("unknown field `{k}`")));
    }
    if p.is_empty() {
        return Ok(slices_json(s));
    }
    let off = match p.get("off") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(cmd, "`off` must be true or false")),
    };
    let axis = match p.get("axis") {
        Some(v) => axis_of(cmd, v)?,
        None if off => None,
        None => return Err(bad(cmd, "`axis` is required (x, y, z, or null to remove every slice)")),
    };
    let Some(axis) = axis else {
        s.simulation_view.slices.clear();
        return Ok(slices_json(s));
    };
    if off {
        s.simulation_view.slices.retain(|x| x.axis != axis);
        return Ok(slices_json(s));
    }
    let existing = s.simulation_view.slice(axis).copied();
    let field = match p.get("field") {
        None | Some(Value::Null) => existing.map_or(SliceField::Temperature, |e| e.field),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| bad(cmd, "`field` must be temperature or speed"))?,
    };
    let position = match p.get("position") {
        None | Some(Value::Null) => match (existing, &s.simulation) {
            (Some(e), _) => e.position,
            (None, Some(run)) => {
                let c = run.results.grid.bounds().center();
                [c.x, c.y, c.z].get(axis.index()).copied().unwrap_or(0.0)
            }
            (None, None) => return Err(bad(cmd, "`position` is required before a simulation is solved")),
        },
        Some(v) => v.as_f64().filter(|x| x.is_finite() && x.abs() <= 1e7).ok_or_else(|| bad(cmd, "`position` must be a finite number of mm"))?,
    };
    if field == SliceField::Speed && s.simulation.as_ref().is_some_and(|r| r.results.flow.is_none()) {
        return Err(bad(cmd, "the last solve has no airflow to slice: solve with airflow or use field temperature"));
    }
    let new = Slice { axis, position, field };
    s.simulation_view.slices.retain(|x| x.axis != axis);
    s.simulation_view.slices.push(new);
    s.simulation_view.slices.sort_by_key(|x| x.axis);
    if s.simulation.is_some() && s.simulation_display == SimulationDisplay::Off {
        s.simulation_display = SimulationDisplay::Temperature;
    }
    let mut reply = slices_json(s);
    if let (Some(o), Some(run)) = (reply.as_object_mut(), &s.simulation) {
        let b = run.results.grid.bounds();
        let (lo, hi) = ([b.min.x, b.min.y, b.min.z], [b.max.x, b.max.y, b.max.z]);
        let i = axis.index();
        let (l, h) = (lo.get(i).copied().unwrap_or(0.0), hi.get(i).copied().unwrap_or(0.0));
        o.insert("range".into(), json!([l, h]));
        o.insert("inside".into(), json!(position >= l && position <= h));
    }
    Ok(reply)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    point: Vec3,
    #[serde(default)]
    body: Option<String>,
}

fn probe_pin(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.probe.pin";
    let request: Pin = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    if !request.point.is_finite() {
        return Err(bad(cmd, "`point` must be finite"));
    }
    if request.body.as_ref().is_some_and(|b| b.is_empty() || b.len() > 256) {
        return Err(bad(cmd, "`body` must be a body name"));
    }
    let run = s.simulation.as_ref().ok_or_else(|| bad(cmd, "no simulation has been solved; run sim.solve first"))?;
    if s.simulation_view.probes.len() >= MAX_PROBES {
        return Err(bad(cmd, format!("at most {MAX_PROBES} probes; delete one first")));
    }
    let body = request.body.filter(|b| run.body_index(b).is_some());
    let r = reading(run, request.point, body.as_deref()).ok_or_else(|| bad(cmd, "the point is outside the simulated domain"))?;
    s.simulation_view.probes.push(PinnedProbe { point: request.point, body });
    Ok(json!({"index": s.simulation_view.probes.len() - 1, "reading": r}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Unpin {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    all: bool,
}

fn probe_delete(s: &mut Session, params: &Value) -> Result<Value> {
    let cmd = "sim.probe.delete";
    let request: Unpin = serde_json::from_value(params.clone()).map_err(|e| bad(cmd, e.to_string()))?;
    match (request.index, request.all) {
        (_, true) => s.simulation_view.probes.clear(),
        (Some(i), false) => {
            if i >= s.simulation_view.probes.len() {
                return Err(bad(cmd, "no probe with that index"));
            }
            s.simulation_view.probes.remove(i);
        }
        (None, false) => return Err(bad(cmd, "`index` or `all: true` is required")),
    }
    Ok(json!({"probes": s.simulation_view.probes.len()}))
}

fn probe_list(s: &mut Session, params: &Value) -> Result<Value> {
    if !params.as_object().is_some_and(|p| p.is_empty()) {
        return Err(bad("sim.probe.list", "expected an empty parameter object"));
    }
    let run = s.simulation.as_ref();
    let probes: Vec<Value> = s
        .simulation_view
        .probes
        .iter()
        .enumerate()
        .map(|(i, p)| json!({"index": i, "point": p.point, "body": p.body, "reading": run.and_then(|r| reading(r, p.point, p.body.as_deref()))}))
        .collect();
    Ok(json!({"probes": probes, "stale": run.is_some_and(|r| r.revision != s.revision)}))
}

#[cfg(test)]
#[path = "simulation_view_tests.rs"]
mod tests;
