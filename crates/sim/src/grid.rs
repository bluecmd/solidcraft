//! The uniform voxel grid: the domain box cut into `cell` mm cubes.

use solidcraft_geom::{Aabb3, Vec3};

use crate::{Result, SimError};

/// Cell label for air (anything that is not a body).
pub const FLUID: u32 = u32::MAX;

/// Integer cell coordinates.
pub type Ijk = [usize; 3];

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// World position (mm) of the corner of cell (0, 0, 0).
    pub origin: Vec3,
    /// Cell edge in mm.
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Grid {
    /// A grid covering `domain` with cubes of `cell` mm; the box is rounded up to whole cells.
    pub fn new(domain: Aabb3, cell: f64, max_cells: usize) -> Result<Grid> {
        if domain.is_empty() || !domain.min.is_finite() || !domain.max.is_finite() {
            return Err(SimError::Invalid("domain box is empty".into()));
        }
        let cell = crate::finite(cell, 1e-3, 1e4, "cell size (mm)")?;
        let size = domain.size();
        let count = |len: f64| -> Result<usize> {
            let n = (len / cell - 1e-9).ceil().max(1.0);
            if n > 1e7 {
                return Err(SimError::TooLarge(format!("{n} cells along one axis")));
            }
            Ok(n as usize)
        };
        let (nx, ny, nz) = (count(size.x)?, count(size.y)?, count(size.z)?);
        let total = nx.checked_mul(ny).and_then(|n| n.checked_mul(nz)).ok_or_else(|| SimError::TooLarge("cell count overflows".into()))?;
        if total > max_cells {
            return Err(SimError::TooLarge(format!(
                "{nx}x{ny}x{nz} = {total} cells at {cell} mm exceeds the limit of {max_cells}; use a coarser cell size or a smaller domain"
            )));
        }
        Ok(Grid { origin: domain.min, cell, nx, ny, nz })
    }

    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Cell edge in metres.
    pub fn dx_m(&self) -> f64 {
        self.cell * 1e-3
    }
    pub fn index(&self, [i, j, k]: Ijk) -> usize {
        (k * self.ny + j) * self.nx + i
    }
    pub fn coords(&self, idx: usize) -> Ijk {
        let i = idx % self.nx;
        let j = (idx / self.nx) % self.ny;
        let k = idx / (self.nx * self.ny);
        [i, j, k]
    }
    pub fn centre(&self, [i, j, k]: Ijk) -> Vec3 {
        self.origin + Vec3::new((i as f64 + 0.5) * self.cell, (j as f64 + 0.5) * self.cell, (k as f64 + 0.5) * self.cell)
    }
    /// The cell containing a world point, if inside the grid.
    pub fn cell_of(&self, p: Vec3) -> Option<Ijk> {
        if !p.is_finite() {
            return None;
        }
        let f = (p - self.origin) * (1.0 / self.cell);
        if f.x < 0.0 || f.y < 0.0 || f.z < 0.0 {
            return None;
        }
        let (i, j, k) = (f.x as usize, f.y as usize, f.z as usize);
        (i < self.nx && j < self.ny && k < self.nz).then_some([i, j, k])
    }
    /// Continuous grid coordinates (cell units from the origin corner) of a world point.
    pub fn grid_coords(&self, p: Vec3) -> Vec3 {
        (p - self.origin) * (1.0 / self.cell)
    }
    /// The six face neighbours of a cell (`-x, +x, -y, +y, -z, +z`); `None` past the domain wall.
    pub fn neighbours(&self, [i, j, k]: Ijk) -> [Option<usize>; 6] {
        [
            (i > 0).then(|| self.index([i - 1, j, k])),
            (i + 1 < self.nx).then(|| self.index([i + 1, j, k])),
            (j > 0).then(|| self.index([i, j - 1, k])),
            (j + 1 < self.ny).then(|| self.index([i, j + 1, k])),
            (k > 0).then(|| self.index([i, j, k - 1])),
            (k + 1 < self.nz).then(|| self.index([i, j, k + 1])),
        ]
    }
    pub fn bounds(&self) -> Aabb3 {
        Aabb3 { min: self.origin, max: self.origin + Vec3::new(self.nx as f64, self.ny as f64, self.nz as f64) * self.cell }
    }
}
