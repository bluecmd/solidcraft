//! Sparse linear algebra on the 7-point stencil: conjugate gradients for the symmetric
//! (pure conduction) case and BiCGSTAB when advection makes the matrix one-sided. Both take the
//! preconditioner as `z = M⁻¹ r` (a multigrid cycle, or Jacobi in tests); both stop at a
//! relative residual or an iteration cap, never a panic.

use crate::par_chunks_mut;

/// `A` as a diagonal plus six off-diagonal coefficients per row (`-x, +x, -y, +y, -z, +z`),
/// with `None`-neighbours stored as 0 and the neighbour index derived from the grid.
pub struct Stencil {
    pub nx: usize,
    pub ny: usize,
    pub diag: Vec<f64>,
    pub off: Vec<[f64; 6]>,
}

impl Stencil {
    pub fn len(&self) -> usize {
        self.diag.len()
    }
    pub fn is_empty(&self) -> bool {
        self.diag.is_empty()
    }

    /// `y = A x`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        let (nx, ny) = (self.nx, self.ny);
        let plane = nx * ny;
        let n = self.len();
        if x.len() != n || y.len() != n {
            return;
        }
        let diag = &self.diag;
        let off = &self.off;
        // One chunk per z plane keeps the neighbour arithmetic simple and the work parallel.
        par_chunks_mut(y, plane, |k, out| {
            let base = k * plane;
            for (r, o) in out.iter_mut().enumerate() {
                let idx = base + r;
                let i = r % nx;
                let j = r / nx;
                let (Some(d), Some(c), Some(xi)) = (diag.get(idx), off.get(idx), x.get(idx)) else { continue };
                let mut s = d * xi;
                if i > 0
                    && let Some(v) = x.get(idx - 1)
                {
                    s += c[0] * v;
                }
                if i + 1 < nx
                    && let Some(v) = x.get(idx + 1)
                {
                    s += c[1] * v;
                }
                if j > 0
                    && let Some(v) = x.get(idx - nx)
                {
                    s += c[2] * v;
                }
                if j + 1 < ny
                    && let Some(v) = x.get(idx + nx)
                {
                    s += c[3] * v;
                }
                if k > 0
                    && let Some(v) = x.get(idx - plane)
                {
                    s += c[4] * v;
                }
                if let Some(v) = x.get(idx + plane) {
                    s += c[5] * v;
                }
                *o = s;
            }
        });
    }

    pub fn is_symmetric(&self) -> bool {
        let (nx, ny) = (self.nx, self.ny);
        let plane = nx * ny;
        self.off.iter().enumerate().all(|(idx, c)| {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let near = |cond: bool, other: usize, mine: f64, theirs: usize| -> bool {
                !cond || self.off.get(other).is_some_and(|o| (o[theirs] - mine).abs() <= 1e-12 * (1.0 + mine.abs()))
            };
            near(i > 0, idx.wrapping_sub(1), c[0], 1)
                && near(i + 1 < nx, idx + 1, c[1], 0)
                && near(j > 0, idx.wrapping_sub(nx), c[2], 3)
                && near(j + 1 < ny, idx + nx, c[3], 2)
                && near(idx >= plane, idx.wrapping_sub(plane), c[4], 5)
                && near(idx + plane < self.len(), idx + plane, c[5], 4)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveReport {
    pub iterations: usize,
    /// Final relative residual `|b - A x| / |b|`.
    pub residual: f64,
    pub converged: bool,
    pub method: &'static str,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Progress hook: `(iteration, relative residual)`; return `false` to stop.
pub type Watch<'a> = &'a mut dyn FnMut(usize, f64) -> bool;

/// The preconditioner: writes `z ≈ A⁻¹ r` (a fixed linear operator; symmetric for CG).
pub type Precond<'a> = &'a mut dyn FnMut(&[f64], &mut [f64]);

/// Jacobi: `z = r / diag(A)`, the baseline the multigrid tests compare against.
#[cfg(test)]
pub fn jacobi(a: &Stencil) -> impl FnMut(&[f64], &mut [f64]) {
    let inv_diag: Vec<f64> = a.diag.iter().map(|d| if d.abs() > 0.0 { 1.0 / d } else { 0.0 }).collect();
    move |r: &[f64], z: &mut [f64]| {
        for ((zi, ri), di) in z.iter_mut().zip(r).zip(&inv_diag) {
            *zi = ri * di;
        }
    }
}

/// Preconditioned conjugate gradients. `x` holds the initial guess and the answer.
pub fn conjugate_gradient(a: &Stencil, b: &[f64], x: &mut [f64], tol: f64, max_iter: usize, precond: Precond, watch: Watch) -> SolveReport {
    let n = a.len();
    let mut report = SolveReport { iterations: 0, residual: f64::INFINITY, converged: false, method: "cg" };
    if b.len() != n || x.len() != n || n == 0 {
        return report;
    }
    let bnorm = norm(b);
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        report.residual = 0.0;
        report.converged = true;
        return report;
    }
    let mut ax = vec![0.0; n];
    a.apply(x, &mut ax);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let mut z = vec![0.0; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut q = vec![0.0; n];
    for it in 1..=max_iter {
        a.apply(&p, &mut q);
        let pq = dot(&p, &q);
        if !(pq.abs() > 0.0) || !pq.is_finite() {
            break;
        }
        let alpha = rz / pq;
        x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
        r.iter_mut().zip(&q).for_each(|(ri, qi)| *ri -= alpha * qi);
        let res = norm(&r) / bnorm;
        report.iterations = it;
        report.residual = res;
        if !res.is_finite() {
            break;
        }
        if res < tol {
            report.converged = true;
            break;
        }
        if it % 25 == 0 && !watch(it, res) {
            break;
        }
        precond(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = if rz.abs() > 0.0 { rz_new / rz } else { 0.0 };
        rz = rz_new;
        p.iter_mut().zip(&z).for_each(|(pi, zi)| *pi = zi + beta * *pi);
    }
    report
}

/// Jacobi-preconditioned BiCGSTAB for the non-symmetric (advective) system.
pub fn bicgstab(a: &Stencil, b: &[f64], x: &mut [f64], tol: f64, max_iter: usize, precond: Precond, watch: Watch) -> SolveReport {
    let n = a.len();
    let mut report = SolveReport { iterations: 0, residual: f64::INFINITY, converged: false, method: "bicgstab" };
    if b.len() != n || x.len() != n || n == 0 {
        return report;
    }
    let bnorm = norm(b);
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        report.residual = 0.0;
        report.converged = true;
        return report;
    }
    let mut tmp = vec![0.0; n];
    a.apply(x, &mut tmp);
    let mut r: Vec<f64> = b.iter().zip(&tmp).map(|(bi, ti)| bi - ti).collect();
    let r0 = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut ph = vec![0.0; n];
    let mut sh = vec![0.0; n];
    let (mut rho, mut alpha, mut omega): (f64, f64, f64) = (1.0, 1.0, 1.0);
    for it in 1..=max_iter {
        let rho_new = dot(&r0, &r);
        if !(rho_new.abs() > 0.0) || !rho_new.is_finite() || !(omega.abs() > 0.0) {
            break;
        }
        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        p.iter_mut().zip(&r).zip(&v).for_each(|((pi, ri), vi)| *pi = ri + beta * (*pi - omega * vi));
        precond(&p, &mut ph);
        a.apply(&ph, &mut v);
        let r0v = dot(&r0, &v);
        if !(r0v.abs() > 0.0) {
            break;
        }
        alpha = rho / r0v;
        s.iter_mut().zip(&r).zip(&v).for_each(|((si, ri), vi)| *si = ri - alpha * vi);
        let snorm = norm(&s) / bnorm;
        if snorm < tol {
            x.iter_mut().zip(&ph).for_each(|(xi, pi)| *xi += alpha * pi);
            report.iterations = it;
            report.residual = snorm;
            report.converged = true;
            break;
        }
        precond(&s, &mut sh);
        a.apply(&sh, &mut t);
        let tt = dot(&t, &t);
        if !(tt > 0.0) {
            break;
        }
        omega = dot(&t, &s) / tt;
        x.iter_mut().zip(&ph).zip(&sh).for_each(|((xi, pi), si)| *xi += alpha * pi + omega * si);
        r.iter_mut().zip(&s).zip(&t).for_each(|((ri, si), ti)| *ri = si - omega * ti);
        let res = norm(&r) / bnorm;
        report.iterations = it;
        report.residual = res;
        if !res.is_finite() {
            break;
        }
        if res < tol {
            report.converged = true;
            break;
        }
        if it % 25 == 0 && !watch(it, res) {
            break;
        }
    }
    report
}
