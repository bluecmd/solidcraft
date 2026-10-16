//! Voxelisation of closed triangle meshes by parity along z columns.
//!
//! For every (x, y) column of cell centres, the crossings of the vertical line with each body's
//! triangles are collected (a half-open edge rule, so a crossing on an edge shared by two
//! triangles counts once) and sorted; cells whose centre lies between an odd and an even crossing
//! are inside. Earlier bodies win where bodies overlap.

use solidcraft_geom::Vec3;

use crate::grid::{FLUID, Grid};
use crate::problem::SolidBody;
use crate::{Result, SimError};

#[derive(Clone, Debug, PartialEq)]
pub struct Voxels {
    /// Body index per cell, or [`FLUID`].
    pub label: Vec<u32>,
    /// Cells per body.
    pub counts: Vec<usize>,
}

impl Voxels {
    pub fn is_solid(&self, idx: usize) -> bool {
        self.label.get(idx).is_some_and(|l| *l != FLUID)
    }
    pub fn fluid_cells(&self) -> usize {
        self.label.iter().filter(|l| **l == FLUID).count()
    }
}

/// Label every cell of `grid` with the body containing its centre.
pub fn voxelize(grid: &Grid, bodies: &[SolidBody]) -> Result<Voxels> {
    if bodies.len() >= FLUID as usize {
        return Err(SimError::TooLarge("too many bodies".into()));
    }
    let mut label = vec![FLUID; grid.len()];
    let mut counts = vec![0usize; bodies.len()];
    let columns = grid.nx * grid.ny;
    // Crossing heights (in cell units from the grid origin) per column, per body in turn.
    let mut crossings: Vec<Vec<f64>> = vec![Vec::new(); columns];
    for (b, body) in bodies.iter().enumerate() {
        for c in &mut crossings {
            c.clear();
        }
        column_crossings(grid, body, &mut crossings)?;
        let mut cells = 0usize;
        for (col, zs) in crossings.iter_mut().enumerate() {
            if zs.len() < 2 {
                continue;
            }
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (i, j) = (col % grid.nx, col / grid.nx);
            for [z0, z1] in zs.as_chunks::<2>().0 {
                let (z0, z1) = (*z0, *z1);
                // cells whose centre k + 0.5 lies in (z0, z1)
                let k0 = (z0 - 0.5).ceil().max(0.0) as usize;
                let k1 = (z1 - 0.5).ceil().min(grid.nz as f64) as usize;
                for k in k0..k1 {
                    let idx = grid.index([i, j, k]);
                    if let Some(l) = label.get_mut(idx)
                        && *l == FLUID
                    {
                        *l = b as u32;
                        cells += 1;
                    }
                }
            }
        }
        if let Some(c) = counts.get_mut(b) {
            *c = cells;
        }
    }
    Ok(Voxels { label, counts })
}

/// Collects, for each column, the heights (cell units) where the body's surface crosses it.
fn column_crossings(grid: &Grid, body: &SolidBody, crossings: &mut [Vec<f64>]) -> Result<()> {
    let inv = 1.0 / grid.cell;
    let to_grid = |p: Vec3| (p - grid.origin) * inv;
    for tri in &body.triangles {
        let fetch = |i: u32| body.positions.get(i as usize).copied().filter(|p| p.is_finite()).map(to_grid);
        let (Some(a), Some(b), Some(c)) = (fetch(tri[0]), fetch(tri[1]), fetch(tri[2])) else {
            return Err(SimError::Invalid(format!("body '{}': triangle references a missing or non-finite vertex", body.name)));
        };
        // Orient counter-clockwise in the xy projection; skip vertical (zero-area) triangles.
        let area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if area2.abs() < 1e-18 {
            continue;
        }
        let (b, c) = if area2 < 0.0 { (c, b) } else { (b, c) };
        let area2 = area2.abs();
        let min_x = a.x.min(b.x).min(c.x);
        let max_x = a.x.max(b.x).max(c.x);
        let min_y = a.y.min(b.y).min(c.y);
        let max_y = a.y.max(b.y).max(c.y);
        // Column centres are at i + 0.5; the covered range of i.
        let i0 = (min_x - 0.5).ceil().max(0.0) as usize;
        let i1 = ((max_x - 0.5).floor().min(grid.nx as f64 - 1.0)).max(-1.0);
        let j0 = (min_y - 0.5).ceil().max(0.0) as usize;
        let j1 = ((max_y - 0.5).floor().min(grid.ny as f64 - 1.0)).max(-1.0);
        if i1 < 0.0 || j1 < 0.0 {
            continue;
        }
        let (i1, j1) = (i1 as usize, j1 as usize);
        for j in j0..=j1 {
            let py = j as f64 + 0.5;
            for i in i0..=i1 {
                let px = i as f64 + 0.5;
                let w0 = edge(b, c, px, py);
                let w1 = edge(c, a, px, py);
                let w2 = edge(a, b, px, py);
                if !(covers(w0, b, c) && covers(w1, c, a) && covers(w2, a, b)) {
                    continue;
                }
                // barycentric interpolation of z
                let z = (w0 * a.z + w1 * b.z + w2 * c.z) / area2;
                if let Some(col) = crossings.get_mut(j * grid.nx + i) {
                    col.push(z);
                }
            }
        }
    }
    Ok(())
}

/// Twice the signed area of (p, q, (x, y)); positive on the left of p→q.
fn edge(p: Vec3, q: Vec3, x: f64, y: f64) -> f64 {
    (q.x - p.x) * (y - p.y) - (q.y - p.y) * (x - p.x)
}

/// Half-open coverage (the top-left rule): a point exactly on an edge belongs to the triangle on
/// one side only, so a column through a shared edge gets one crossing, not two.
fn covers(w: f64, p: Vec3, q: Vec3) -> bool {
    if w > 0.0 {
        return true;
    }
    if w < 0.0 {
        return false;
    }
    let (dx, dy) = (q.x - p.x, q.y - p.y);
    dy < 0.0 || (dy == 0.0 && dx > 0.0)
}
