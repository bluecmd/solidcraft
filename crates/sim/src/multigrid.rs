//! Geometric multigrid on the 7-point stencil, the preconditioner of the Krylov solvers. Cells
//! aggregate 2 × 2 × 2 into a coarse cell and the coarse operator is Galerkin (`R A P` with
//! piecewise-constant prolongation), which keeps every level a 7-point stencil and an M-matrix
//! whatever the conductivity jumps; red-black Gauss-Seidel smooths. One V-cycle from zero is a
//! fixed linear operator, so conjugate gradients (the cycle is symmetric: the post-smoother is
//! the pre-smoother in reverse) and BiCGSTAB can use it as `M⁻¹`.

use crate::linsolve::Stencil;
use crate::par_chunks_mut;

/// Red-black sweeps (one red and one black) before and after the coarse correction.
pub const SMOOTH: usize = 2;
/// Sweep pairs on the coarsest level, each way.
pub const COARSE_SWEEPS: usize = 8;
/// Coarsening stops at this many cells.
const COARSEST: usize = 1000;

/// The hierarchy, finest first.
pub struct Multigrid {
    levels: Vec<Stencil>,
}

/// Per-level vectors a cycle works in.
pub struct Work {
    levels: Vec<LevelWork>,
}

struct LevelWork {
    x: Vec<f64>,
    b: Vec<f64>,
    r: Vec<f64>,
    swap: Vec<f64>,
}

fn dims(a: &Stencil) -> (usize, usize, usize) {
    let plane = a.nx * a.ny;
    (a.nx, a.ny, a.len().checked_div(plane).unwrap_or(0))
}

/// The Galerkin coarse operator of `fine`: one coarse cell per 2 × 2 × 2 block (smaller blocks at
/// odd far edges).
fn coarsen(fine: &Stencil) -> Stencil {
    let (nx, ny, nz) = dims(fine);
    let plane = nx * ny;
    let (cx, cy, cz) = (nx.div_ceil(2), ny.div_ceil(2), nz.div_ceil(2));
    let cplane = cx * cy;
    let cn = cplane * cz;
    let mut diag = vec![0.0; cn];
    let mut off = vec![[0.0; 6]; cn];
    for (idx, (d, o)) in fine.diag.iter().zip(&fine.off).enumerate() {
        let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / plane);
        let block = (i / 2) + cx * ((j / 2) + cy * (k / 2));
        let Some(cd) = diag.get_mut(block) else { continue };
        *cd += d;
        // a face neighbour is in the same block or in the block on that side
        let inside = [i % 2 == 1, i % 2 == 0 && i + 1 < nx, j % 2 == 1, j % 2 == 0 && j + 1 < ny, k % 2 == 1, k % 2 == 0 && k + 1 < nz];
        for side in 0..6 {
            if inside[side] {
                *cd += o[side];
            } else if let Some(co) = off.get_mut(block) {
                co[side] += o[side];
            }
        }
    }
    for (d, o) in diag.iter_mut().zip(off.iter_mut()) {
        if !(*d > 0.0) || !d.is_finite() {
            // cannot happen for an M-matrix with nonnegative column sums; keep the level regular
            *d = 1.0;
            *o = [0.0; 6];
        }
    }
    Stencil { nx: cx, ny: cy, diag, off }
}

impl Multigrid {
    pub fn new(fine: &Stencil) -> Multigrid {
        let mut levels = vec![Stencil { nx: fine.nx, ny: fine.ny, diag: fine.diag.clone(), off: fine.off.clone() }];
        while let Some(last) = levels.last() {
            if last.len() <= COARSEST {
                break;
            }
            let coarse = coarsen(last);
            if coarse.len() >= last.len() || coarse.is_empty() {
                break;
            }
            levels.push(coarse);
        }
        Multigrid { levels }
    }

    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// The operator of every level, finest first.
    pub fn stencils(&self) -> &[Stencil] {
        &self.levels
    }

    pub fn work(&self) -> Work {
        Work {
            levels: self
                .levels
                .iter()
                .map(|a| LevelWork { x: vec![0.0; a.len()], b: vec![0.0; a.len()], r: vec![0.0; a.len()], swap: vec![0.0; a.len()] })
                .collect(),
        }
    }

    /// `x ≈ A⁻¹ b`: one V-cycle from zero.
    pub fn apply(&self, b: &[f64], x: &mut [f64], work: &mut Work) {
        let Some(top) = work.levels.first_mut() else { return };
        if top.b.len() != b.len() || x.len() != b.len() {
            return;
        }
        top.b.copy_from_slice(b);
        self.cycle(0, work);
        if let Some(top) = work.levels.first() {
            x.copy_from_slice(&top.x);
        }
    }

    fn cycle(&self, l: usize, work: &mut Work) {
        let Some(a) = self.levels.get(l) else { return };
        let coarsest = l + 1 == self.levels.len();
        let (lower, upper) = work.levels.split_at_mut(l + 1);
        let Some(w) = lower.last_mut() else { return };
        w.x.iter_mut().for_each(|v| *v = 0.0);
        if coarsest {
            for _ in 0..COARSE_SWEEPS {
                sweep_pair(a, &w.b, &mut w.x, &mut w.swap, false);
            }
            for _ in 0..COARSE_SWEEPS {
                sweep_pair(a, &w.b, &mut w.x, &mut w.swap, true);
            }
            return;
        }
        for _ in 0..SMOOTH {
            sweep_pair(a, &w.b, &mut w.x, &mut w.swap, false);
        }
        a.apply(&w.x, &mut w.r);
        for (r, b) in w.r.iter_mut().zip(&w.b) {
            *r = b - *r;
        }
        let Some(next) = upper.first_mut() else { return };
        let Some(coarse) = self.levels.get(l + 1) else { return };
        restrict(a, &w.r, coarse, &mut next.b);
        self.cycle(l + 1, work);
        let (lower, upper) = work.levels.split_at_mut(l + 1);
        let (Some(w), Some(next)) = (lower.last_mut(), upper.first()) else { return };
        prolong_add(a, &mut w.x, coarse, &next.x);
        for _ in 0..SMOOTH {
            sweep_pair(a, &w.b, &mut w.x, &mut w.swap, true);
        }
    }
}

/// One red and one black Gauss-Seidel sweep (black first when `reversed`): the colours alternate
/// on the 7-point stencil, so each half-sweep is an independent update of every cell of one
/// colour from its neighbours of the other, written through `swap`.
fn sweep_pair(a: &Stencil, b: &[f64], x: &mut Vec<f64>, swap: &mut Vec<f64>, reversed: bool) {
    let order = if reversed { [1usize, 0] } else { [0, 1] };
    for colour in order {
        half_sweep(a, b, x, swap, colour);
        std::mem::swap(x, swap);
    }
}

fn half_sweep(a: &Stencil, b: &[f64], x: &[f64], out: &mut [f64], colour: usize) {
    let (nx, ny, _) = dims(a);
    let plane = nx * ny;
    let n = a.len();
    if x.len() != n || out.len() != n || b.len() != n {
        return;
    }
    par_chunks_mut(out, plane, |k, out| {
        let base = k * plane;
        for (r, o) in out.iter_mut().enumerate() {
            let idx = base + r;
            let i = r % nx;
            let j = r / nx;
            let (Some(&xi), Some(&bi), Some(&d), Some(c)) = (x.get(idx), b.get(idx), a.diag.get(idx), a.off.get(idx)) else { continue };
            if (i + j + k) % 2 != colour {
                *o = xi;
                continue;
            }
            let mut s = bi;
            if i > 0
                && let Some(v) = x.get(idx - 1)
            {
                s -= c[0] * v;
            }
            if i + 1 < nx
                && let Some(v) = x.get(idx + 1)
            {
                s -= c[1] * v;
            }
            if j > 0
                && let Some(v) = x.get(idx - nx)
            {
                s -= c[2] * v;
            }
            if j + 1 < ny
                && let Some(v) = x.get(idx + nx)
            {
                s -= c[3] * v;
            }
            if k > 0
                && let Some(v) = x.get(idx - plane)
            {
                s -= c[4] * v;
            }
            if let Some(v) = x.get(idx + plane) {
                s -= c[5] * v;
            }
            *o = if d.abs() > 0.0 { s / d } else { 0.0 };
        }
    });
}

/// `coarse_b = R fine_r`: each coarse cell sums its block.
fn restrict(fine: &Stencil, fine_r: &[f64], coarse: &Stencil, coarse_b: &mut [f64]) {
    let (nx, ny, nz) = dims(fine);
    let (cx, cy, _) = dims(coarse);
    let cplane = cx * cy;
    par_chunks_mut(coarse_b, cplane, |kc, out| {
        for (r, o) in out.iter_mut().enumerate() {
            let ic = r % cx;
            let jc = r / cx;
            let mut s = 0.0;
            for k in (2 * kc)..(2 * kc + 2).min(nz) {
                for j in (2 * jc)..(2 * jc + 2).min(ny) {
                    for i in (2 * ic)..(2 * ic + 2).min(nx) {
                        s += fine_r.get(i + nx * (j + ny * k)).copied().unwrap_or(0.0);
                    }
                }
            }
            *o = s;
        }
    });
}

/// `fine_x += P coarse_x`: each fine cell takes its block's value.
fn prolong_add(fine: &Stencil, fine_x: &mut [f64], coarse: &Stencil, coarse_x: &[f64]) {
    let (nx, ny, _) = dims(fine);
    let (cx, cy, _) = dims(coarse);
    let plane = nx * ny;
    par_chunks_mut(fine_x, plane, |k, out| {
        for (r, o) in out.iter_mut().enumerate() {
            let i = r % nx;
            let j = r / nx;
            let block = (i / 2) + cx * ((j / 2) + cy * (k / 2));
            *o += coarse_x.get(block).copied().unwrap_or(0.0);
        }
    });
}
