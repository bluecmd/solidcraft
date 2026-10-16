use serde_json::json;

use crate::scenario::Harness;
use solidcraft_engine::SimulationDisplay;

#[test]
fn workspace_switches_by_control_and_menu_and_solves_from_the_panel() {
    let mut h = Harness::new();
    h.app.session.set_simulation_accelerator(Some(std::sync::Arc::new(solidcraft_engine::sim::CpuAccelerator)));
    h.call("engine.execute", json!({"command": "solid.box", "params": {"length": 20, "width": 20, "height": 10}}));
    let r = h.call("ui.workspace", json!({"workspace": "simulation"}));
    assert_eq!(r["ok"], true, "{r}");
    assert_eq!(r["result"]["workspace"], "simulation");
    assert!(super::active(&h.app));
    h.frames(2);
    // The panel's draft: a heat source on the body, then a solve lands in the session.
    h.app.simulation_workspace.view.draft.cell_size_mm = 4.0;
    h.app.simulation_workspace.view.draft.bodies.entry("Body1".into()).or_default().power_w = 1.0;
    h.app.simulation_workspace.view.dirty = true;
    super::solve(&mut h.app);
    for _ in 0..400 {
        h.frames(1);
        if h.app.session.simulation.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let run = h.app.session.simulation.clone().expect("the solve finished");
    assert_eq!(run.study, "Thermal");
    assert!(h.app.session.doc.simulation.studies.contains_key("Thermal"), "solving saved the study");
    assert!(!h.app.simulation_workspace.view.dirty);
    assert_eq!(h.app.session.simulation_display, SimulationDisplay::Temperature);
    assert!(run.results.summary.bodies[0].max_c > 25.0);
    // The viewport colours the body: a scene key change rebuilds the model scene.
    h.frames(2);
    let r = h.call("engine.execute", json!({"command": "sim.display", "params": {"mode": "off"}}));
    assert_eq!(r["ok"], true, "{r}");
    h.frames(2);
    // Back to Design, and the render workspace still works.
    let r = h.call("ui.workspace", json!({"workspace": "design"}));
    assert_eq!(r["result"]["workspace"], "design");
    assert!(!super::active(&h.app));
    let r = h.call("ui.workspace", json!({"workspace": "render"}));
    assert_eq!(r["result"]["workspace"], "render");
    let r = h.call("ui.workspace", json!({"workspace": "simulation"}));
    assert_eq!(r["result"]["workspace"], "simulation");
    assert!(!crate::render_workspace::active(&h.app));
    assert!(h.call("ui.workspace", json!({"workspace": "sideways"}))["ok"] == false);
}

#[test]
fn heat_colours_span_the_body_range_and_arrows_follow_the_flow() {
    let (lo, hi) = (super::heat_color(0.0), super::heat_color(1.0));
    assert!(lo[2] > lo[0], "cold is blue: {lo:?}");
    assert!(hi[0] > hi[2], "hot is red: {hi:?}");
    assert_eq!(super::heat_color(f64::NAN), super::heat_color(0.0));
    let mut s = solidcraft_engine::Session::default();
    s.set_simulation_accelerator(Some(std::sync::Arc::new(solidcraft_engine::sim::CpuAccelerator)));
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 10})).unwrap();
    s.execute(
        "sim.study.set",
        &json!({"name": "T", "set": {"cell_size_mm": 2.0, "padding_mm": 10.0,
        "walls": ["open", "open", "closed", "closed", "closed", "closed"], "flow": {"max_steps": 600, "buoyancy": false}}}),
    )
    .unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 1.0})).unwrap();
    s.execute("sim.fan.set", &json!({"study": "T", "name": "fan", "region": {"min": [-9, -10, -10], "max": [-6, 30, 20]}, "velocity": [1, 0, 0]}))
        .unwrap();
    s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    let run = s.simulation.clone().unwrap();
    let (lo, hi) = super::body_range(&run);
    assert!(hi > lo);
    let mesh = s.world_state().bodies[0].mesh();
    let (refined, colours) = super::heat_display(&run, 0, &mesh, lo, hi);
    assert_eq!(colours.len(), refined.positions.len());
    assert!(refined.triangles.len() > mesh.triangles.len(), "the box is refined to the grid");
    assert!(colours.iter().all(|c| c[3] == 255 && *c != [120, 120, 120, 255]), "every vertex found its cell");
    let mut sc = crate::gpu::GpuScene::default();
    super::velocity_arrows(&mut sc, &run);
    assert!(!sc.lines.is_empty(), "arrows were drawn");
}
