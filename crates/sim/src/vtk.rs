//! Legacy VTK (structured points) writer so results open in ParaView.

use std::fmt::Write as _;

use crate::grid::FLUID;
use crate::results::Results;

/// The results as an ASCII legacy VTK file with cell data `body`, `temperature` (°C) and, when
/// airflow was solved, `velocity` (m/s) and `pressure` (Pa).
pub fn write_vtk(results: &Results) -> String {
    let g = &results.grid;
    let mut s = String::new();
    let _ = writeln!(s, "# vtk DataFile Version 3.0");
    let _ = writeln!(s, "SolidCraft simulation (mm)");
    let _ = writeln!(s, "ASCII");
    let _ = writeln!(s, "DATASET STRUCTURED_POINTS");
    let _ = writeln!(s, "DIMENSIONS {} {} {}", g.nx + 1, g.ny + 1, g.nz + 1);
    let _ = writeln!(s, "ORIGIN {} {} {}", g.origin.x, g.origin.y, g.origin.z);
    let _ = writeln!(s, "SPACING {} {} {}", g.cell, g.cell, g.cell);
    let _ = writeln!(s, "CELL_DATA {}", g.len());
    let _ = writeln!(s, "SCALARS body int 1");
    let _ = writeln!(s, "LOOKUP_TABLE default");
    for l in &results.voxels.label {
        let _ = writeln!(s, "{}", if *l == FLUID { -1 } else { *l as i64 });
    }
    let _ = writeln!(s, "SCALARS temperature float 1");
    let _ = writeln!(s, "LOOKUP_TABLE default");
    for t in &results.thermal.temperature {
        let _ = writeln!(s, "{}", *t as f32);
    }
    if let Some(flow) = &results.flow {
        let _ = writeln!(s, "VECTORS velocity float");
        for v in &flow.velocity {
            let _ = writeln!(s, "{} {} {}", v[0], v[1], v[2]);
        }
        let _ = writeln!(s, "SCALARS pressure float 1");
        let _ = writeln!(s, "LOOKUP_TABLE default");
        for p in &flow.pressure {
            let _ = writeln!(s, "{p}");
        }
    }
    s
}
