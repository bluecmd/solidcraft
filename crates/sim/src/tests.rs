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
    SolidBody { name: name.into(), positions, triangles, conductivity: k, power_w: power, fixed_temperature_c: None, fluid: None }
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
        fluids: Vec::new(),
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

/// A block in a wider duct, as an enclosure study sees it: the flow stays stable and the air
/// never gets hotter than the block that heats it.
#[test]
fn bluff_body_in_a_duct_stays_stable() {
    let block = boxed("block", [0.0, 0.0, 0.0], [20.0, 20.0, 10.0], 200.0, 0.5);
    let mut p = problem(domain([-12.0, -12.0, -12.0], [32.0, 32.0, 22.0]), 2.0, vec![block]);
    p.walls = [Wall::Open, Wall::Open, Wall::Closed, Wall::Closed, Wall::Closed, Wall::Closed];
    p.air.mixing_factor = 1.0;
    p.fans.push(Fan { name: "fan".into(), region: domain([-11.0, -12.0, -12.0], [-8.0, 32.0, 22.0]), velocity: Vec3::new(1.5, 0.0, 0.0) });
    p.flow = Some(FlowSettings { max_steps: 1500, buoyancy: false, ..Default::default() });
    let r = solve(&p, &mut go()).unwrap();
    let f = r.summary.flow.as_ref().unwrap();
    assert!(f.max_lattice_speed < 0.3, "{f:?}");
    assert!(f.max_speed_m_s < 4.0 && f.max_speed_m_s > 1.0, "{f:?}");
    assert!(r.summary.air_max_c <= r.summary.bodies[0].max_c + 0.5, "{:?}", r.summary);
    // With the air kept near ambient by the fan, the film coefficient alone sets the rise: P/(hA).
    let expected = 20.0 + 0.5 / (8.0 * 1.6e-3);
    assert!((r.summary.bodies[0].mean_c - expected).abs() < 2.0, "{:?} vs {expected}", r.summary.bodies[0]);
    let inlet = r.temperature_at(Vec3::new(-5.0, 10.0, 5.0)).unwrap();
    let outlet = r.temperature_at(Vec3::new(25.0, 10.0, 5.0)).unwrap();
    assert!(outlet > inlet + 0.5, "the air leaves warmer than it came: {inlet} → {outlet}");
}

#[test]
#[ignore]
fn debug_duct() {
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let block = boxed("block", [0.0, 0.0, 0.0], [20.0, 20.0, 10.0], 200.0, 0.5);
    let mut p = problem(domain([-12.0, -12.0, -12.0], [32.0, 32.0, 22.0]), 2.0, vec![block]);
    p.walls = [Wall::Open, Wall::Open, Wall::Closed, Wall::Closed, Wall::Closed, Wall::Closed];
    p.air.mixing_factor = 1.0;
    p.fans.push(Fan {
        name: "fan".into(),
        region: domain([env("SIM_FAN", -11.0), -12.0, -12.0], [-8.0, 32.0, 22.0]),
        velocity: Vec3::new(1.5, 0.0, 0.0),
    });
    let mut out = Vec::new();
    for steps in [1000, 2000, 4000] {
        p.flow = Some(FlowSettings {
            max_steps: steps,
            buoyancy: false,
            tolerance: 1e-12,
            trt_lambda: env("SIM_LAMBDA", 1.0 / 12.0),
            max_cell_reynolds: env("SIM_RE", 10.0),
            smagorinsky: env("SIM_CS", 0.12),
            lattice_velocity: env("SIM_U", 0.05),
            ..Default::default()
        });
        match solve(&p, &mut go()) {
            Ok(r) => {
                let f = r.summary.flow.as_ref().unwrap();
                out.push(format!("{steps}: max {:.2} change {:.4}", f.max_speed_m_s, f.change));
            }
            Err(e) => out.push(format!("{steps}: {e}")),
        }
    }
    eprintln!(
        "RESULT lambda={} re={} cs={} u={} fan={}: {}",
        env("SIM_LAMBDA", 1.0 / 12.0),
        env("SIM_RE", 10.0),
        env("SIM_CS", 0.12),
        env("SIM_U", 0.05),
        env("SIM_FAN", -11.0),
        out.join(" | ")
    );
}

/// A water loop (four legs of coolant, a pump in one) over a hot copper block: the pump moves
/// the water around the loop, the water picks the heat up at the block and the block runs
/// cooler than with the water standing still.
#[test]
fn a_pumped_water_loop_cools_a_block() {
    let water = |name: &str, min: [f64; 3], max: [f64; 3]| {
        let mut b = boxed(name, min, max, 0.6, 0.0);
        b.fluid = Some(0);
        b
    };
    let legs = vec![
        water("leg_a", [10.0, 6.0, 14.0], [90.0, 12.0, 20.0]),
        water("leg_b", [10.0, 28.0, 14.0], [90.0, 34.0, 20.0]),
        water("leg_c", [10.0, 6.0, 14.0], [16.0, 34.0, 20.0]),
        water("leg_d", [84.0, 6.0, 14.0], [90.0, 34.0, 20.0]),
    ];
    let block = boxed("block", [40.0, 6.0, 6.0], [60.0, 14.0, 14.0], 385.0, 2.0);
    let mut bodies = legs;
    bodies.push(block);
    let mut p = problem(domain([0.0, 0.0, 0.0], [100.0, 40.0, 30.0]), 2.0, bodies);
    p.fluids.push(Fluid::water());
    p.air.mixing_factor = 10.0;
    p.fans.push(Fan { name: "pump".into(), region: domain([30.0, 6.0, 14.0], [36.0, 12.0, 20.0]), velocity: Vec3::new(0.5, 0.0, 0.0) });
    // still water first (no flow solve at all)
    let still = solve(&p, &mut go()).unwrap();
    assert_eq!(still.summary.fluids.len(), 2, "{:?}", still.summary.fluids);
    assert_eq!(still.summary.fluids[1].name, "fluid1");
    // then pumped
    p.flow = Some(FlowSettings { max_steps: 2000, buoyancy: false, ..Default::default() });
    let r = solve(&p, &mut go()).unwrap();
    let f = r.flow.as_ref().unwrap();
    assert_eq!(f.zones.len(), 2, "{:?}", f.zones);
    let water_run = &f.zones[1];
    assert!(water_run.max_lattice_speed < 0.3, "{water_run:?}");
    // water moves along leg A past the pump and comes back along leg B
    let a = r.velocity_at(Vec3::new(70.0, 9.0, 17.0)).unwrap();
    let b = r.velocity_at(Vec3::new(50.0, 31.0, 17.0)).unwrap();
    assert!(a.x > 0.1, "leg A flows +x: {a:?}");
    assert!(b.x < -0.1, "leg B flows -x: {b:?}");
    assert_eq!(r.zone_at(Vec3::new(70.0, 9.0, 17.0)), Some(1));
    assert_eq!(r.zone_at(Vec3::new(50.0, 10.0, 10.0)), Some(solidcraft_sim_zone_solid()));
    assert_eq!(r.zone_at(Vec3::new(5.0, 2.0, 2.0)), Some(0));
    // the air zone never carries the pump: nothing blows outside the loop
    let air = r.velocity_at(Vec3::new(5.0, 20.0, 5.0)).unwrap();
    assert!(air.len() < 0.05, "{air:?}");
    let (hot_still, hot) = (still.summary.bodies[4].max_c, r.summary.bodies[4].max_c);
    assert!(hot < hot_still - 2.0, "pumped {hot} vs still {hot_still}");
    let water = &r.summary.fluids[1];
    assert!(water.max_c > water.min_c + 0.5 && water.max_speed_m_s > 0.3, "{water:?}");
    assert!(write_vtk(&r).contains("SCALARS zone int 1"));
}

fn solidcraft_sim_zone_solid() -> u16 {
    crate::zones::ZONE_SOLID
}

/// Streamlines seeded in the water follow the loop and never leave it; seeded in the air they
/// never enter the water.
#[test]
fn streamlines_stay_in_their_fluid_zone() {
    let water = |name: &str, min: [f64; 3], max: [f64; 3]| {
        let mut b = boxed(name, min, max, 0.6, 0.0);
        b.fluid = Some(0);
        b
    };
    let bodies = vec![
        water("leg_a", [10.0, 6.0, 14.0], [90.0, 12.0, 20.0]),
        water("leg_b", [10.0, 28.0, 14.0], [90.0, 34.0, 20.0]),
        water("leg_c", [10.0, 6.0, 14.0], [16.0, 34.0, 20.0]),
        water("leg_d", [84.0, 6.0, 14.0], [90.0, 34.0, 20.0]),
    ];
    let mut p = problem(domain([0.0, 0.0, 0.0], [100.0, 40.0, 30.0]), 2.0, bodies);
    p.fluids.push(Fluid::water());
    p.fans.push(Fan { name: "pump".into(), region: domain([30.0, 6.0, 14.0], [36.0, 12.0, 20.0]), velocity: Vec3::new(0.5, 0.0, 0.0) });
    p.flow = Some(FlowSettings { max_steps: 1500, buoyancy: false, ..Default::default() });
    let r = solve(&p, &mut go()).unwrap();
    let line = r.streamline(Vec3::new(50.0, 9.0, 17.0), &crate::sample::StreamlineOptions { max_steps: 4000, ..Default::default() });
    assert!(line.points.len() > 20, "a water streamline: {} points", line.points.len());
    assert!(line.points.iter().all(|q| r.zone_at(*q) == Some(1)), "it stays in the water");
    // it turns the corners of the loop: reaches leg B (y around 31)
    assert!(line.points.iter().any(|q| q.y > 28.0), "it goes round the loop");
}

/// A fan wall without a baffle in a duct open at both ends: air recirculates back through the
/// gaps beside the fans at several times the fan speed, and mixes in and out of the open inlet.
/// TRT with a zero-gradient open boundary blew up here; regularized collision with a sponge at
/// the open walls stays bounded.
#[test]
fn an_unbaffled_fan_wall_stays_stable() {
    let block = boxed("block", [30.0, 90.0, 0.0], [70.0, 110.0, 20.0], 200.0, 2.0);
    let mut p = problem(domain([0.0, 0.0, 0.0], [100.0, 160.0, 30.0]), 2.0, vec![block]);
    p.walls = [Wall::Closed, Wall::Closed, Wall::Open, Wall::Open, Wall::Closed, Wall::Closed];
    p.air.mixing_factor = 1.0;
    for k in 0..2 {
        let x0 = 10.0 + 45.0 * f64::from(k);
        p.fans.push(Fan { name: format!("fan{k}"), region: domain([x0, 50.0, 2.0], [x0 + 35.0, 58.0, 28.0]), velocity: Vec3::new(0.0, 4.0, 0.0) });
    }
    p.flow = Some(FlowSettings { max_steps: 2500, buoyancy: false, tolerance: 1e-12, ..Default::default() });
    let r = solve(&p, &mut go()).unwrap();
    let f = r.summary.flow.as_ref().unwrap();
    assert!(f.max_lattice_speed < 0.3, "{f:?}");
    // the fans move air from front to back overall
    let (mut sum, mut n) = (0.0, 0);
    for i in 0..50 {
        for k in 0..15 {
            if let Some(v) = r.velocity_at(Vec3::new(1.0 + 2.0 * i as f64, 140.0, 1.0 + 2.0 * k as f64)) {
                sum += v.y;
                n += 1;
            }
        }
    }
    assert!(sum / n as f64 > 0.3, "net flow to the back: {}", sum / n as f64);
}

/// Solver experiments on a system `sim-bench --dump-thermal` wrote: the path in
/// `SOLIDCRAFT_THERMAL_DUMP`, residuals traced with `SOLIDCRAFT_SIM_DEBUG`.
#[test]
#[ignore]
fn dumped_system_on_the_cpu() {
    let Some(path) = std::env::var_os("SOLIDCRAFT_THERMAL_DUMP") else { return };
    let bytes = std::fs::read(path).expect("read the dump");
    let system = ThermalSystem::from_bytes(&bytes, 1e-8, 20_000).expect("parse the dump");
    let started = std::time::Instant::now();
    let solved = CpuAccelerator.thermal(&system, &mut |_, _, _| true).expect("solve");
    let mut ax = vec![0.0; system.rhs.len()];
    system.stencil.apply(&solved.x, &mut ax);
    let bnorm = system.rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let true_res = system.rhs.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum::<f64>().sqrt() / bnorm;
    let hottest = solved.x.iter().cloned().fold(f64::MIN, f64::max);
    eprintln!(
        "cpu {}: {} iterations, converged {}, residual {:.3e}, true residual {true_res:.3e}, hottest {hottest:.2} K above ambient, {:.1} s",
        solved.method,
        solved.iterations,
        solved.converged,
        solved.residual,
        started.elapsed().as_secs_f64()
    );
}

/// A box of unit conductances, open (held at ambient) on its z = 0 face, with upwind advection
/// of `speed` along +x: an M-matrix with zero column sums inside, like the thermal system.
fn box_stencil(nx: usize, ny: usize, nz: usize, speed: f64) -> Stencil {
    let n = nx * ny * nz;
    let mut diag = vec![0.0; n];
    let mut off = vec![[0.0; 6]; n];
    for idx in 0..n {
        let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
        let has = [i > 0, i + 1 < nx, j > 0, j + 1 < ny, k > 0, k + 1 < nz];
        let mut d = 0.0;
        for side in 0..6 {
            if has[side] {
                let (out_flux, in_flux) = match side {
                    0 => (0.0, speed),
                    1 => (speed, 0.0),
                    _ => (0.0, 0.0),
                };
                d += 1.0 + out_flux;
                off[idx][side] = -(1.0 + in_flux);
            }
        }
        if k == 0 {
            d += 1.0;
        }
        if i + 1 == nx {
            d += speed;
        }
        diag[idx] = d;
    }
    Stencil { nx, ny, diag, off }
}

fn true_residual(a: &Stencil, b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.apply(x, &mut ax);
    let r: f64 = b.iter().zip(&ax).map(|(bi, ai)| (bi - ai) * (bi - ai)).sum::<f64>().sqrt();
    r / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn multigrid_is_symmetric_and_far_stronger_than_jacobi() {
    let a = box_stencil(33, 29, 25, 0.0);
    let n = a.len();
    let b: Vec<f64> = (0..n).map(|i| if i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let mg = Multigrid::new(&a);
    assert!(mg.levels() >= 3, "{} levels", mg.levels());
    // symmetric: <V r1, r2> = <r1, V r2>
    let mut work = mg.work();
    let r1: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
    let r2: Vec<f64> = (0..n).map(|i| ((i * 13) % 11) as f64 - 5.0).collect();
    let (mut v1, mut v2) = (vec![0.0; n], vec![0.0; n]);
    mg.apply(&r1, &mut v1, &mut work);
    mg.apply(&r2, &mut v2, &mut work);
    let lhs: f64 = v1.iter().zip(&r2).map(|(x, y)| x * y).sum();
    let rhs: f64 = r1.iter().zip(&v2).map(|(x, y)| x * y).sum();
    assert!((lhs - rhs).abs() <= 1e-9 * lhs.abs().max(rhs.abs()), "{lhs} vs {rhs}");
    // and a much better preconditioner
    let mut watch = |_: usize, _: f64| true;
    let mut x_j = vec![0.0; n];
    let mut jac = crate::linsolve::jacobi(&a);
    let jacobi = crate::linsolve::conjugate_gradient(&a, &b, &mut x_j, 1e-9, 5000, &mut jac, &mut watch);
    let mut x_m = vec![0.0; n];
    let mut pre = |r: &[f64], z: &mut [f64]| mg.apply(r, z, &mut work);
    let multigrid = crate::linsolve::conjugate_gradient(&a, &b, &mut x_m, 1e-9, 5000, &mut pre, &mut watch);
    assert!(jacobi.converged && multigrid.converged, "{jacobi:?} {multigrid:?}");
    assert!(true_residual(&a, &b, &x_m) < 1e-8);
    assert!(multigrid.iterations * 4 < jacobi.iterations, "multigrid {} vs jacobi {} iterations", multigrid.iterations, jacobi.iterations);
}

#[test]
fn multigrid_bicgstab_solves_a_strongly_one_sided_system() {
    let a = box_stencil(24, 20, 18, 50.0);
    assert!(!a.is_symmetric());
    let n = a.len();
    let b: Vec<f64> = (0..n).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
    let mg = Multigrid::new(&a);
    let mut work = mg.work();
    let mut watch = |_: usize, _: f64| true;
    let mut x = vec![0.0; n];
    let mut pre = |r: &[f64], z: &mut [f64]| mg.apply(r, z, &mut work);
    let report = crate::linsolve::bicgstab(&a, &b, &mut x, 1e-9, 500, &mut pre, &mut watch);
    assert!(report.converged, "{report:?}");
    assert!(report.iterations < 100, "{} iterations", report.iterations);
    assert!(true_residual(&a, &b, &x) < 1e-8);
}
