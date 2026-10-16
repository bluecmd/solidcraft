//! Read-only sampling of a solved study for display: smooth (trilinear) field values between cell
//! centres, and streamlines traced through the velocity field.
//!
//! Nothing here changes a result. Every loop is bounded by caps on the caller's numbers, so a
//! hostile seed count, step count or length cannot run away.

use solidcraft_geom::Vec3;

use crate::grid::{FLUID, Grid};
use crate::problem::{Fan, Wall};
use crate::results::Results;

/// Most steps one streamline may take, whatever the caller asks for.
pub const MAX_STREAMLINE_STEPS: usize = 20_000;
/// Most seeds any seeding helper returns, whatever the caller asks for.
pub const MAX_SEEDS: usize = 4096;

/// The eight cell indices and weights around a world point for trilinear interpolation between
/// cell centres. Outside the grid box: `None`. Near the walls the stencil is clamped to the last
/// cell (constant extrapolation over the outer half cell).
fn stencil(grid: &Grid, p: Vec3) -> Option<[(usize, f64); 8]> {
    if !p.is_finite() || grid.is_empty() {
        return None;
    }
    let b = grid.bounds();
    let eps = grid.cell * 1e-9;
    if p.x < b.min.x - eps || p.y < b.min.y - eps || p.z < b.min.z - eps || p.x > b.max.x + eps || p.y > b.max.y + eps || p.z > b.max.z + eps {
        return None;
    }
    let g = grid.grid_coords(p) - Vec3::new(0.5, 0.5, 0.5);
    let axis = |v: f64, n: usize| -> (usize, usize, f64) {
        let top = n.saturating_sub(1) as f64;
        let v = v.clamp(0.0, top);
        let i0 = v.floor() as usize;
        let i0 = i0.min(n.saturating_sub(1));
        let i1 = (i0 + 1).min(n.saturating_sub(1));
        (i0, i1, v - i0 as f64)
    };
    let (i0, i1, tx) = axis(g.x, grid.nx);
    let (j0, j1, ty) = axis(g.y, grid.ny);
    let (k0, k1, tz) = axis(g.z, grid.nz);
    let w = |a: bool, t: f64| if a { t } else { 1.0 - t };
    let mut out = [(0usize, 0.0f64); 8];
    for (n, slot) in out.iter_mut().enumerate() {
        let (hx, hy, hz) = (n & 1 == 1, n & 2 == 2, n & 4 == 4);
        let idx = grid.index([if hx { i1 } else { i0 }, if hy { j1 } else { j0 }, if hz { k1 } else { k0 }]);
        *slot = (idx, w(hx, tx) * w(hy, ty) * w(hz, tz));
    }
    Some(out)
}

/// One traced streamline: points in world mm and the air speed (m/s) at each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Streamline {
    pub points: Vec<Vec3>,
    pub speeds: Vec<f64>,
}

/// How far and how finely to trace a streamline.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamlineOptions {
    /// Step length in cells (clamped to 0.05..=2).
    pub step_cells: f64,
    /// Steps per direction (capped at [`MAX_STREAMLINE_STEPS`]).
    pub max_steps: usize,
    /// Length per direction in mm (non-finite or ≤ 0: the domain diagonal times four).
    pub max_length_mm: f64,
    /// Stop where the air is slower than this (m/s): a stagnation point or a dead corner.
    pub min_speed: f64,
    /// Trace upstream as well as downstream (a seed in the middle of a fan shows both sides).
    pub both_ways: bool,
}

impl Default for StreamlineOptions {
    fn default() -> Self {
        StreamlineOptions { step_cells: 0.5, max_steps: 2000, max_length_mm: 0.0, min_speed: 1e-3, both_ways: false }
    }
}

impl Results {
    /// Temperature (°C) trilinearly interpolated between cell centres. Smooth across air and
    /// solids alike; outside the grid `None`.
    pub fn temperature_smooth(&self, p: Vec3) -> Option<f64> {
        let s = stencil(&self.grid, p)?;
        let mut sum = 0.0;
        for (idx, w) in s {
            sum += self.thermal.temperature.get(idx)? * w;
        }
        sum.is_finite().then_some(sum)
    }

    /// Air velocity (m/s) trilinearly interpolated between cell centres; solid cells count as
    /// still air, so the speed falls off towards a wall. `None` outside the grid or without a
    /// flow solution.
    pub fn velocity_smooth(&self, p: Vec3) -> Option<Vec3> {
        let flow = self.flow.as_ref()?;
        let s = stencil(&self.grid, p)?;
        let mut sum = Vec3::ZERO;
        for (idx, w) in s {
            let v = flow.velocity.get(idx)?;
            sum += Vec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])) * w;
        }
        sum.is_finite().then_some(sum)
    }

    /// Air speed of one cell, m/s (0 in solids or without a flow solution).
    pub fn cell_speed(&self, idx: usize) -> f64 {
        let Some(v) = self.flow.as_ref().and_then(|f| f.velocity.get(idx)) else { return 0.0 };
        let s = f64::from(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if s.is_finite() { s } else { 0.0 }
    }

    /// Fastest air speed anywhere in the grid, m/s (0 without a flow solution).
    pub fn max_speed(&self) -> f64 {
        let n = self.flow.as_ref().map_or(0, |f| f.velocity.len());
        (0..n).map(|i| self.cell_speed(i)).fold(0.0, f64::max)
    }

    /// Traces a streamline from `seed` with fourth-order Runge–Kutta steps of fixed length along
    /// the interpolated velocity direction. Stops at the domain wall, on leaving the fluid zone
    /// the seed is in (a solid, or another fluid: air and water never mix), where the fluid is
    /// slower than `min_speed`, or at the step and length caps. Empty without a flow solution or
    /// when the seed is outside the grid or in a solid.
    pub fn streamline(&self, seed: Vec3, options: &StreamlineOptions) -> Streamline {
        let mut out = Streamline::default();
        let Some(zone) = self.fluid_zone(seed).filter(|_| self.flow.is_some()) else {
            return out;
        };
        let forward = self.trace(seed, options, 1.0, zone);
        if options.both_ways {
            let back = self.trace(seed, options, -1.0, zone);
            // back runs seed → upstream; reverse it and drop its copy of the seed
            for (p, s) in back.points.iter().rev().zip(back.speeds.iter().rev()) {
                out.points.push(*p);
                out.speeds.push(*s);
            }
            out.points.pop();
            out.speeds.pop();
        }
        out.points.extend(forward.points);
        out.speeds.extend(forward.speeds);
        out
    }

    /// The fluid zone at `p` (air or a coolant), `None` in a solid or outside the grid.
    fn fluid_zone(&self, p: Vec3) -> Option<u16> {
        let c = self.grid.cell_of(p)?;
        self.zones.get(self.grid.index(c)).copied().filter(|z| *z != crate::zones::ZONE_SOLID)
    }

    /// One direction of a streamline (`sign` = 1 downstream, -1 upstream), seed first.
    fn trace(&self, seed: Vec3, o: &StreamlineOptions, sign: f64, zone: u16) -> Streamline {
        let mut out = Streamline::default();
        let step_cells = if o.step_cells.is_finite() { o.step_cells.clamp(0.05, 2.0) } else { 0.5 };
        let h = step_cells * self.grid.cell;
        let max_steps = o.max_steps.min(MAX_STREAMLINE_STEPS);
        let diag = self.grid.bounds().diagonal();
        let max_len = if o.max_length_mm.is_finite() && o.max_length_mm > 0.0 { o.max_length_mm.min(diag * 16.0) } else { diag * 4.0 };
        let min_speed = if o.min_speed.is_finite() { o.min_speed.max(0.0) } else { 1e-3 };
        // unit direction (times sign) and the speed at a point; None where tracing must stop
        let dir = |p: Vec3| -> Option<(Vec3, f64)> {
            let v = self.velocity_smooth(p)?;
            let s = v.len();
            if !(s > min_speed) || s <= 0.0 {
                return None;
            }
            Some((v * (sign / s), s))
        };
        let Some((_, s0)) = dir(seed) else { return out };
        out.points.push(seed);
        out.speeds.push(s0);
        let (mut p, mut length) = (seed, 0.0);
        for _ in 0..max_steps {
            let Some((k1, _)) = dir(p) else { break };
            let Some((k2, _)) = dir(p + k1 * (h * 0.5)) else { break };
            let Some((k3, _)) = dir(p + k2 * (h * 0.5)) else { break };
            let Some((k4, _)) = dir(p + k3 * h) else { break };
            let d = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (1.0 / 6.0);
            let next = p + d * h;
            if !next.is_finite() || self.fluid_zone(next) != Some(zone) {
                break;
            }
            let Some((_, s)) = dir(next) else { break };
            length += next.dist(p);
            p = next;
            out.points.push(p);
            out.speeds.push(s);
            if length >= max_len {
                break;
            }
        }
        out
    }

    /// Seeds where air enters through open domain walls: centres of the air cells next to an open
    /// wall whose velocity points into the domain faster than `min_speed`, thinned evenly to at
    /// most `max` (capped at [`MAX_SEEDS`]).
    pub fn inflow_seeds(&self, walls: &[Wall; 6], max: usize, min_speed: f64) -> Vec<Vec3> {
        let max = max.min(MAX_SEEDS);
        let Some(flow) = self.flow.as_ref() else { return Vec::new() };
        if max == 0 {
            return Vec::new();
        }
        let g = &self.grid;
        let min_speed = if min_speed.is_finite() { min_speed.max(0.0) } else { 0.0 };
        let mut found = Vec::new();
        for (w, wall) in walls.iter().enumerate() {
            if *wall != Wall::Open {
                continue;
            }
            let axis = w / 2;
            let high = w % 2 == 1;
            let n = [g.nx, g.ny, g.nz];
            let (Some(&na), Some(&nb), Some(&nc)) = (n.get(axis), n.get((axis + 1) % 3), n.get((axis + 2) % 3)) else { continue };
            let layer = if high { na.saturating_sub(1) } else { 0 };
            for b in 0..nb {
                for c in 0..nc {
                    let mut ijk = [0usize; 3];
                    if let Some(s) = ijk.get_mut(axis) {
                        *s = layer;
                    }
                    if let Some(s) = ijk.get_mut((axis + 1) % 3) {
                        *s = b;
                    }
                    if let Some(s) = ijk.get_mut((axis + 2) % 3) {
                        *s = c;
                    }
                    let idx = g.index(ijk);
                    if self.voxels.label.get(idx) != Some(&FLUID) {
                        continue;
                    }
                    let Some(v) = flow.velocity.get(idx) else { continue };
                    let inward = f64::from(v.get(axis).copied().unwrap_or(0.0)) * if high { -1.0 } else { 1.0 };
                    if inward > min_speed {
                        found.push(g.centre(ijk));
                    }
                }
            }
        }
        thin(found, max)
    }
}

/// Seeds across a fan: an even grid of points on the mid-plane of its region across its main
/// flow direction, at most `max` (capped at [`MAX_SEEDS`]). A fan with no speed seeds its centre.
pub fn fan_seeds(fan: &Fan, max: usize) -> Vec<Vec3> {
    let max = max.min(MAX_SEEDS);
    let r = fan.region;
    if max == 0 || r.is_empty() || !r.min.is_finite() || !r.max.is_finite() {
        return Vec::new();
    }
    let v = fan.velocity;
    let a = [v.x.abs(), v.y.abs(), v.z.abs()];
    let axis = if a[0] >= a[1] && a[0] >= a[2] {
        0
    } else if a[1] >= a[2] {
        1
    } else {
        2
    };
    let lo = [r.min.x, r.min.y, r.min.z];
    let hi = [r.max.x, r.max.y, r.max.z];
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let side = (max as f64).sqrt().floor().max(1.0) as usize;
    let mut out = Vec::with_capacity(side * side);
    let at = |arr: &[f64; 3], i: usize| arr.get(i).copied().unwrap_or(0.0);
    for i in 0..side {
        for j in 0..side {
            let mut p = [0.0f64; 3];
            if let Some(s) = p.get_mut(axis) {
                *s = 0.5 * (at(&lo, axis) + at(&hi, axis));
            }
            if let Some(s) = p.get_mut(u) {
                *s = at(&lo, u) + (i as f64 + 0.5) / side as f64 * (at(&hi, u) - at(&lo, u));
            }
            if let Some(s) = p.get_mut(w) {
                *s = at(&lo, w) + (j as f64 + 0.5) / side as f64 * (at(&hi, w) - at(&lo, w));
            }
            out.push(Vec3::from(p));
        }
    }
    out
}

/// Keeps at most `max` items, picked evenly through the list.
fn thin(items: Vec<Vec3>, max: usize) -> Vec<Vec3> {
    if items.len() <= max {
        return items;
    }
    let n = items.len();
    (0..max).filter_map(|i| items.get(i * n / max).copied()).collect()
}

#[cfg(test)]
mod tests {
    use solidcraft_geom::{Aabb3, Vec3};

    use super::*;
    use crate::flow::FlowReport;
    use crate::problem::{Air, Limits, Problem, SolidBody, Thermal};
    use crate::thermal::ThermalReport;
    use crate::voxel::Voxels;

    /// A synthetic result on a 10x6x4 grid of 2 mm cells: temperature = 20 + x (mm), velocity a
    /// uniform `v`, with an optional solid column of cells at i = 7.
    fn synthetic(v: [f32; 3], wall_at_7: bool) -> Results {
        let grid = Grid::new(Aabb3 { min: Vec3::ZERO, max: Vec3::new(20.0, 12.0, 8.0) }, 2.0, 10_000).unwrap();
        let n = grid.len();
        let mut label = vec![FLUID; n];
        let mut temperature = vec![0.0; n];
        let mut velocity = vec![v; n];
        for idx in 0..n {
            let [i, ..] = grid.coords(idx);
            temperature[idx] = 20.0 + grid.centre(grid.coords(idx)).x;
            if wall_at_7 && i == 7 {
                label[idx] = 0;
                velocity[idx] = [0.0; 3];
            }
        }
        let body = SolidBody {
            name: "wall".into(),
            positions: vec![Vec3::ZERO; 3],
            triangles: vec![[0, 1, 2]],
            conductivity: 1.0,
            power_w: 0.0,
            fixed_temperature_c: None,
            fluid: None,
        };
        let problem = Problem {
            domain: grid.bounds(),
            cell_size: 2.0,
            bodies: vec![body],
            walls: [Wall::Open; 6],
            air: Air::default(),
            fluids: Vec::new(),
            thermal: Thermal::default(),
            fans: Vec::new(),
            flow: None,
            limits: Limits::default(),
        };
        let thermal =
            ThermalReport { temperature, iterations: 0, residual: 0.0, converged: true, method: "test".into(), heat_in: 0.0, heat_out: 0.0 };
        let flow = FlowReport {
            velocity,
            pressure: vec![0.0; n],
            steps: 0,
            converged: true,
            change: 0.0,
            dt: 1.0,
            max_lattice_speed: 0.0,
            air_cells: n,
            effective_viscosity: 1.5e-5,
            backend: "test".into(),
            mlups: 0.0,
            zones: Vec::new(),
        };
        let counts = vec![label.iter().filter(|l| **l == 0).count()];
        let voxels = Voxels { label, counts };
        let zones = crate::zones::zones(&voxels, &problem);
        Results::new(grid, voxels, zones, &problem, thermal, Some(flow))
    }

    #[test]
    fn trilinear_sampling_is_exact_on_a_linear_field_and_clamped_at_walls() {
        let r = synthetic([1.0, 0.0, 0.0], false);
        // between cell centres a linear field interpolates exactly
        for x in [1.0, 2.5, 7.3, 18.99] {
            let t = r.temperature_smooth(Vec3::new(x, 5.0, 3.0)).unwrap();
            assert!((t - (20.0 + x)).abs() < 1e-9, "{x}: {t}");
        }
        // over the outer half cell the value is held (constant extrapolation)
        assert!((r.temperature_smooth(Vec3::new(0.2, 5.0, 3.0)).unwrap() - 21.0).abs() < 1e-9);
        assert!((r.temperature_smooth(Vec3::new(20.0, 12.0, 8.0)).unwrap() - 39.0).abs() < 1e-9);
        // outside the grid, or a hostile point: nothing
        assert!(r.temperature_smooth(Vec3::new(-0.1, 5.0, 3.0)).is_none());
        assert!(r.temperature_smooth(Vec3::new(f64::NAN, 5.0, 3.0)).is_none());
        assert!(r.velocity_smooth(Vec3::new(5.0, 5.0, 30.0)).is_none());
        let v = r.velocity_smooth(Vec3::new(5.0, 5.0, 3.0)).unwrap();
        assert!((v - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-6);
        assert!((r.max_speed() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_uniform_flow_gives_straight_streamlines_that_stop_at_solids_and_caps() {
        let r = synthetic([1.0, 0.5, 0.0], false);
        let seed = Vec3::new(1.0, 1.0, 4.0);
        let line = r.streamline(seed, &StreamlineOptions::default());
        assert!(line.points.len() > 5, "{line:?}");
        assert_eq!(line.points.len(), line.speeds.len());
        let dir = Vec3::new(1.0, 0.5, 0.0).normalized().unwrap();
        for p in &line.points {
            let off = *p - seed;
            assert!((off - dir * off.dot(dir)).len() < 1e-6, "off the line: {p:?}");
            assert!(r.grid.bounds().min.x <= p.x && p.x <= 20.0 && p.y <= 12.0);
        }
        // a step cap is honoured; a hostile one is capped
        let short = r.streamline(seed, &StreamlineOptions { max_steps: 3, ..Default::default() });
        assert_eq!(short.points.len(), 4);
        let huge = r
            .streamline(seed, &StreamlineOptions { max_steps: usize::MAX, step_cells: f64::NAN, max_length_mm: f64::INFINITY, ..Default::default() });
        assert!(huge.points.len() <= MAX_STREAMLINE_STEPS + 1);
        // a length cap
        let capped = r.streamline(seed, &StreamlineOptions { max_length_mm: 4.0, ..Default::default() });
        let len: f64 = capped.points.windows(2).map(|w| w[0].dist(w[1])).sum();
        assert!((4.0..6.0).contains(&len), "{len}");

        // a solid wall at x in [14, 16) stops the line before it
        let walled = synthetic([1.0, 0.0, 0.0], true);
        let line = walled.streamline(Vec3::new(1.0, 5.0, 4.0), &StreamlineOptions::default());
        assert!(line.points.iter().all(|p| p.x < 14.0), "{line:?}");
        assert!(line.points.last().unwrap().x > 12.0, "{line:?}");
        // a seed inside the solid traces nothing; upstream tracing reaches back to the wall
        assert!(walled.streamline(Vec3::new(15.0, 5.0, 4.0), &StreamlineOptions::default()).points.is_empty());
        let both = walled.streamline(Vec3::new(5.0, 5.0, 4.0), &StreamlineOptions { both_ways: true, ..Default::default() });
        assert!(both.points.first().unwrap().x < 1.0 && both.points.last().unwrap().x > 12.0, "{both:?}");
        assert!(both.points.windows(2).all(|w| w[1].x > w[0].x), "ordered upstream to downstream");
    }

    #[test]
    fn seeds_come_from_inflow_walls_and_fans_and_are_capped() {
        let r = synthetic([1.0, 0.0, 0.0], false);
        // air enters through -x only (6 x 4 = 24 boundary cells)
        let seeds = r.inflow_seeds(&[Wall::Open; 6], 1000, 0.01);
        assert_eq!(seeds.len(), 24);
        assert!(seeds.iter().all(|p| (p.x - 1.0).abs() < 1e-9));
        assert_eq!(r.inflow_seeds(&[Wall::Open; 6], 5, 0.01).len(), 5);
        assert!(r.inflow_seeds(&[Wall::Closed; 6], 1000, 0.01).is_empty());
        assert!(r.inflow_seeds(&[Wall::Open; 6], usize::MAX, f64::NAN).len() <= MAX_SEEDS);

        let fan = Fan {
            name: "f".into(),
            region: Aabb3 { min: Vec3::new(2.0, 0.0, 0.0), max: Vec3::new(4.0, 12.0, 8.0) },
            velocity: Vec3::new(0.0, 0.0, -2.0),
        };
        // flow along -z: seeds on the z mid-plane spread over x and y
        let s = fan_seeds(&fan, 9);
        assert_eq!(s.len(), 9);
        assert!(s.iter().all(|p| (p.z - 4.0).abs() < 1e-9 && p.x > 2.0 && p.x < 4.0 && p.y > 0.0 && p.y < 12.0));
        assert!(fan_seeds(&fan, usize::MAX).len() <= MAX_SEEDS);
        assert!(fan_seeds(&fan, 0).is_empty());
        let bad = Fan { region: Aabb3::EMPTY, ..fan };
        assert!(fan_seeds(&bad, 9).is_empty());
    }
}
