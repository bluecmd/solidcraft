use solidcraft_geom::{Aabb3, Vec3};

use crate::*;

fn boxed(name: &str, min: [f64; 3], max: [f64; 3], k: f64, power: f64) -> SolidBody {
    let (a, b) = (Vec3::from(min), Vec3::from(max));
    let p = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
    let positions = vec![
        p(a.x, a.y, a.z),
        p(b.x, a.y, a.z),
        p(b.x, b.y, a.z),
        p(a.x, b.y, a.z),
        p(a.x, a.y, b.z),
        p(b.x, a.y, b.z),
        p(b.x, b.y, b.z),
        p(a.x, b.y, b.z),
    ];
    // outward, counter-clockwise from outside
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2], // bottom (-z)
        [4, 5, 6],
        [4, 6, 7], // top (+z)
        [0, 1, 5],
        [0, 5, 4], // -y
        [2, 3, 7],
        [2, 7, 6], // +y
        [0, 4, 7],
        [0, 7, 3], // -x
        [1, 2, 6],
        [1, 6, 5], // +x
    ];
    SolidBody { name: name.into(), positions, triangles, conductivity: k, power_w: power, fixed_temperature_c: None }
}

fn domain(min: [f64; 3], max: [f64; 3]) -> Aabb3 {
    Aabb3 { min: Vec3::from(min), max: Vec3::from(max) }
}

fn problem(domain: Aabb3, cell: f64, bodies: Vec<SolidBody>) -> Problem {
    Problem {
        domain,
        cell_size: cell,
        bodies,
        walls: [Wall::Open; 6],
        air: Air::default(),
        thermal: Thermal::default(),
        fans: Vec::new(),
        flow: None,
        limits: Limits::default(),
    }
}

fn go() -> impl FnMut(&str, usize, usize) -> bool {
    |_, _, _| true
}

#[test]
fn voxelizer_counts_a_cube_exactly() {
    let grid = Grid::new(domain([0.0; 3], [20.0; 3]), 1.0, 1_000_000).unwrap();
    let v = voxelize(&grid, &[boxed("cube", [5.0; 3], [15.0; 3], 1.0, 0.0)]).unwrap();
    assert_eq!(v.counts, vec![1000]);
    assert_eq!(v.fluid_cells(), 8000 - 1000);
    // cell centres: (5.5..14.5) inside
    assert!(v.is_solid(grid.index([5, 5, 5])));
    assert!(v.is_solid(grid.index([14, 14, 14])));
    assert!(!v.is_solid(grid.index([4, 5, 5])));
    assert!(!v.is_solid(grid.index([15, 5, 5])));
}

#[test]
fn voxelizer_handles_half_cell_offsets_and_overlaps() {
    let grid = Grid::new(domain([0.0; 3], [10.0; 3]), 1.0, 1_000_000).unwrap();
    // 2.3..7.3 covers centres 2.5..6.5 → 5 per axis
    let v = voxelize(&grid, &[boxed("a", [2.3; 3], [7.3; 3], 1.0, 0.0), boxed("b", [0.0; 3], [10.0; 3], 1.0, 0.0)]).unwrap();
    assert_eq!(v.counts, vec![125, 1000 - 125]);
}

#[test]
fn grid_refuses_hostile_sizes() {
    assert!(Grid::new(domain([0.0; 3], [10.0; 3]), f64::NAN, 1000).is_err());
    assert!(Grid::new(domain([0.0; 3], [10.0; 3]), 0.0, 1000).is_err());
    assert!(Grid::new(domain([0.0; 3], [1e6; 3]), 1.0, 1000).is_err());
    assert!(Grid::new(Aabb3::EMPTY, 1.0, 1000).is_err());
    assert!(Grid::new(domain([0.0; 3], [f64::INFINITY; 3]), 1.0, 1000).is_err());
}

/// A bar heated uniformly with one end held cold and everything else insulated: the far end
/// sits `P L / (2 k A)` above the cold end.
#[test]
fn slab_conduction_matches_the_analytic_profile() {
    let mut bar = boxed("bar", [0.0, 0.0, 0.0], [100.0, 10.0, 10.0], 10.0, 1.0);
    bar.conductivity = 10.0;
    let mut sink = boxed("sink", [-2.0, 0.0, 0.0], [0.0, 10.0, 10.0], 1e5, 0.0);
    sink.fixed_temperature_c = Some(0.0);
    let mut p = problem(domain([-2.0, 0.0, 0.0], [100.0, 10.0, 10.0]), 1.0, vec![bar, sink]);
    p.walls = [Wall::Closed; 6];
    p.thermal.film_coefficient = 0.0;
    p.air.conductivity = 0.0;
    let r = solve(&p, &mut go()).unwrap();
    assert!(r.summary.thermal_converged, "{:?}", r.summary);
    let expected = 1.0 * 0.1 / (2.0 * 10.0 * 1e-4); // 50 K
    let far = r.temperature_at(Vec3::new(99.5, 5.0, 5.0)).unwrap();
    assert!((far - expected).abs() < 0.01 * expected, "far end {far} vs {expected}");
    let mid = r.temperature_at(Vec3::new(49.5, 5.0, 5.0)).unwrap();
    let x = 0.0495;
    let expected_mid = 1.0 / (10.0 * 1e-4 * 0.1) * (0.1 * x - x * x / 2.0);
    assert!((mid - expected_mid).abs() < 0.01 * expected, "middle {mid} vs {expected_mid}");
    assert!((r.summary.heat_out_w - 1.0).abs() < 1e-3, "heat balance {:?}", r.summary);
    let bar = &r.summary.bodies[0];
    assert!(bar.max_c > bar.min_c && bar.mean_c > bar.min_c);
}

/// A hot cube in well-mixed air: the film coefficient alone sets the rise, `P / (h A)`.
#[test]
fn film_coefficient_limits_a_cube_in_free_air() {
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 1e4, 1.0);
    let mut p = problem(domain([0.0; 3], [60.0; 3]), 2.0, vec![cube]);
    p.thermal.film_coefficient = 10.0;
    p.thermal.ambient_c = 25.0;
    p.air.mixing_factor = 1e6;
    let r = solve(&p, &mut go()).unwrap();
    let expected = 25.0 + 1.0 / (10.0 * 6.0 * 0.02 * 0.02);
    let b = &r.summary.bodies[0];
    assert!((b.mean_c - expected).abs() < 0.03 * (expected - 25.0), "cube {b:?} vs {expected}");
    assert!((b.max_c - b.min_c) < 0.5, "a conductive cube is nearly isothermal: {b:?}");
    assert!((r.summary.heat_out_w - 1.0).abs() < 1e-3, "heat balance {:?}", r.summary);
    assert!((r.summary.air_mean_c - 25.0).abs() < 0.1);
    // the surface query finds the cube from a point on its face
    let t = r.surface_temperature(0, Vec3::new(20.0, 30.0, 30.0)).unwrap();
    assert!((t - b.mean_c).abs() < 0.5);
    assert!(write_vtk(&r).contains("SCALARS temperature float 1"));
}

#[test]
fn a_sealed_insulated_box_has_nowhere_to_send_heat() {
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 1e4, 1.0);
    let mut p = problem(domain([0.0; 3], [60.0; 3]), 5.0, vec![cube]);
    p.walls = [Wall::Closed; 6];
    assert!(matches!(solve(&p, &mut go()), Err(SimError::Invalid(_))));
}

#[test]
fn a_body_thinner_than_a_cell_is_refused_with_advice() {
    let cube = boxed("foil", [20.0, 20.0, 20.0], [40.0, 40.0, 20.2], 1e4, 1.0);
    let p = problem(domain([0.0; 3], [60.0; 3]), 5.0, vec![cube]);
    match solve(&p, &mut go()) {
        Err(SimError::Invalid(m)) => assert!(m.contains("smaller cell size"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn cancel_stops_the_solve() {
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 1e4, 1.0);
    let p = problem(domain([0.0; 3], [60.0; 3]), 2.0, vec![cube]);
    let mut stop = |_: &str, _: usize, _: usize| false;
    assert!(matches!(solve(&p, &mut stop), Err(SimError::Cancelled)));
}

#[test]
fn hostile_problems_never_panic() {
    let cube = boxed("cube", [20.0; 3], [40.0; 3], 1e4, 1.0);
    let base = problem(domain([0.0; 3], [60.0; 3]), 5.0, vec![cube]);
    let mut bad = Vec::new();
    for v in [f64::NAN, f64::INFINITY, -1.0, 1e300] {
        let mut p = base.clone();
        p.cell_size = v;
        bad.push(p);
        let mut p = base.clone();
        p.bodies[0].conductivity = v;
        bad.push(p);
        let mut p = base.clone();
        p.bodies[0].power_w = v;
        bad.push(p);
        let mut p = base.clone();
        p.thermal.film_coefficient = v;
        bad.push(p);
        let mut p = base.clone();
        p.air.mixing_factor = v;
        bad.push(p);
        let mut p = base.clone();
        p.limits.tolerance = v;
        bad.push(p);
    }
    let mut p = base.clone();
    p.bodies[0].triangles.push([0, 1, 99]);
    bad.push(p);
    let mut p = base.clone();
    p.bodies[0].positions[0] = Vec3::new(f64::NAN, 0.0, 0.0);
    bad.push(p);
    let mut p = base.clone();
    p.bodies.clear();
    bad.push(p);
    let mut p = base.clone();
    p.limits.max_cells = 10;
    bad.push(p);
    let mut p = base.clone();
    p.fans.push(Fan { name: "f".into(), region: Aabb3::EMPTY, velocity: Vec3::X });
    bad.push(p);
    for p in bad {
        assert!(solve(&p, &mut go()).is_err());
    }
}

/// A fan blows air along a closed channel: downstream the mean speed matches the fan (mass is
/// conserved), the core moves faster than the mean (a developing profile with no-slip walls),
/// and a heated block in the stream runs cooler than in still air.
#[test]
fn fan_driven_channel_flow_conserves_mass_and_cools() {
    let block = boxed("block", [25.0, 4.0, 0.0], [29.0, 8.0, 4.0], 200.0, 0.05);
    let mut p = problem(domain([0.0, 0.0, 0.0], [60.0, 12.0, 12.0]), 1.0, vec![block]);
    p.walls = [Wall::Open, Wall::Open, Wall::Closed, Wall::Closed, Wall::Closed, Wall::Closed];
    p.thermal.film_coefficient = 20.0;
    p.air.mixing_factor = 1.0;
    let still = solve(&p, &mut go()).unwrap();
    p.fans.push(Fan { name: "fan".into(), region: domain([1.0, 0.0, 0.0], [4.0, 12.0, 12.0]), velocity: Vec3::new(1.0, 0.0, 0.0) });
    p.flow = Some(FlowSettings { max_steps: 3000, buoyancy: false, ..Default::default() });
    let r = solve(&p, &mut go()).unwrap();
    let f = r.summary.flow.as_ref().unwrap();
    assert!(f.max_lattice_speed < 0.3, "{f:?}");
    // mean x-velocity over a cross-section at x = 15 mm
    let (mut sum, mut n, mut centre) = (0.0, 0, 0.0);
    for j in 0..12 {
        for k in 0..12 {
            let v = r.velocity_at(Vec3::new(15.5, j as f64 + 0.5, k as f64 + 0.5)).unwrap();
            sum += v.x;
            n += 1;
            if j == 6 && k == 6 {
                centre = v.x;
            }
        }
    }
    let mean = sum / n as f64;
    assert!((mean - 1.0).abs() < 0.2, "mean speed {mean} m/s, {f:?}");
    assert!(centre > mean, "centre {centre} vs mean {mean}");
    assert!(r.summary.bodies[0].mean_c < still.summary.bodies[0].mean_c - 1.0, "flow {:?} still {:?}", r.summary.bodies[0], still.summary.bodies[0]);
    assert!(r.summary.heat_out_w > 0.0);
    assert!(write_vtk(&r).contains("VECTORS velocity float"));
}
