//! Which fluid each cell holds: the air (zone 0) outside the bodies, a coolant (zone `k + 1`)
//! in bodies tagged with fluid `k`, or none (a solid).

use crate::grid::FLUID;
use crate::problem::Problem;
use crate::voxel::Voxels;

/// The zone of a solid cell.
pub const ZONE_SOLID: u16 = u16::MAX;
/// The zone of the air.
pub const ZONE_AIR: u16 = 0;

/// The fluid zone per cell.
pub fn zones(voxels: &Voxels, problem: &Problem) -> Vec<u16> {
    voxels
        .label
        .iter()
        .map(|l| {
            if *l == FLUID {
                ZONE_AIR
            } else {
                problem.bodies.get(*l as usize).and_then(|b| b.fluid).map_or(ZONE_SOLID, |f| (f as u16).saturating_add(1))
            }
        })
        .collect()
}

/// The properties of a zone (`None` for a solid).
pub fn fluid_of(problem: &Problem, zone: u16) -> Option<&crate::problem::Fluid> {
    match zone {
        ZONE_SOLID => None,
        ZONE_AIR => Some(&problem.air),
        k => problem.fluids.get(k as usize - 1),
    }
}

/// What a zone is called in reports.
pub fn zone_name(zone: u16) -> String {
    match zone {
        ZONE_SOLID => "solid".into(),
        ZONE_AIR => "air".into(),
        k => format!("fluid{k}"),
    }
}
