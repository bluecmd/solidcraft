use serde_json::json;
use solidcraft_engine::SimulationDisplay;

use crate::scenario::Harness;

fn exec(h: &mut Harness, command: &str, params: serde_json::Value) -> serde_json::Value {
    let r = h.call("engine.execute", json!({"command": command, "params": params}));
    assert_eq!(r["ok"], true, "{command}: {r}");
    r["result"].clone()
}

/// A heated 20 x 20 x 10 block, solved, in the Simulation workspace.
fn solved_block() -> Harness {
    let mut h = Harness::new();
    exec(&mut h, "solid.box", json!({"length": 20, "width": 20, "height": 10}));
    exec(&mut h, "sim.study.set", json!({"name": "Thermal", "set": {"cell_size_mm": 2.0, "padding_mm": 10.0, "air": {"mixing_factor": 50.0}}}));
    exec(&mut h, "sim.body.set", json!({"study": "Thermal", "body": "Body1", "power_w": 1.0}));
    exec(&mut h, "sim.solve", json!({"study": "Thermal"}));
    let r = h.call("ui.workspace", json!({"workspace": "simulation"}));
    assert_eq!(r["ok"], true, "{r}");
    h.frames(3);
    h
}

#[test]
fn hovering_reads_the_body_and_clicking_pins_a_probe() {
    let mut h = solved_block();
    // the top face of the block, seen from the default view
    h.step(&json!({"move": {"world": [10, 10, 10]}})).unwrap();
    h.frames(2);
    let readout = crate::scenario::rect_of("simulation:readout").expect("a readout under the pointer");
    assert!(readout.width() > 40.0);
    let shown = crate::scenario::count_of("simulation:readout_c").unwrap();
    let run = h.app.session.simulation.clone().unwrap();
    let s = &run.results.summary.bodies[0];
    assert!(shown >= s.min_c - 1e-6 && shown <= s.max_c + 1e-6, "{shown} within the block's {}..{}", s.min_c, s.max_c);
    // a click pins it; the panel lists it and a second click elsewhere adds another
    h.step(&json!({"click": {"world": [10, 10, 10]}})).unwrap();
    h.frames(2);
    let probes = h.app.session.simulation_view.probes.clone();
    assert_eq!(probes.len(), 1, "{probes:?}");
    assert_eq!(probes[0].body.as_deref(), Some("Body1"));
    assert!((probes[0].point.z - 10.0).abs() < 0.5, "{probes:?}");
    assert!(crate::scenario::handle_at("simulation:probe:0").is_some(), "the pinned probe is labelled in the view");
    assert!(h.app.session.selection.is_empty(), "in the Simulation workspace a click probes instead of selecting");
    // the panel's delete button
    let [x, y] = h.widget_near(crate::i18n::tr("sim.delete"), "1 · Body1").or_else(|| h.widget(crate::i18n::tr("sim.clear_probes"))).unwrap();
    h.step(&json!({"click": [x, y]})).unwrap();
    h.frames(2);
    assert!(h.app.session.simulation_view.probes.is_empty());
    // over empty space with no slice: no readout, and a click pins nothing
    crate::scenario::clear_handles();
    h.step(&json!({"click": [1500.0, 950.0]})).unwrap();
    h.frames(2);
    assert!(h.app.session.simulation_view.probes.is_empty());
}

#[test]
fn a_slice_from_the_panel_is_probed_where_no_body_is() {
    let mut h = solved_block();
    // the panel's X checkbox adds a slice through the middle of the domain
    let [x, y] = h.widget("X").expect("the slice row for X");
    h.step(&json!({"click": [x, y]})).unwrap();
    h.frames(2);
    let slices = h.app.session.simulation_view.slices.clone();
    assert_eq!(slices.len(), 1, "{slices:?}");
    assert!((slices[0].position - 10.0).abs() < 1e-9);
    // the legend now spans the air too, and names the slice
    let run = h.app.session.simulation.clone().unwrap();
    let (_, hi) = run.results.temperature_range();
    assert!((crate::scenario::count_of("simulation:legend_hi").unwrap() - hi).abs() < 1e-9);
    // in the air beside the block, on the slice plane
    exec(&mut h, "sim.slice", json!({"axis": "x", "position": 10.0}));
    h.step(&json!({"click": {"world": [10, -6, 5]}})).unwrap();
    h.frames(2);
    let probes = h.app.session.simulation_view.probes.clone();
    assert_eq!(probes.len(), 1, "{probes:?}");
    assert!(probes[0].body.is_none());
    assert!((probes[0].point.x - 10.0).abs() < 1e-6, "on the slice plane: {probes:?}");
    // the toolbar's slice command toggles it off again
    h.call("ui.start", json!({"command": "sim.slice"}));
    h.frames(2);
    assert!(h.app.session.simulation_view.slices.is_empty());
}

#[test]
fn the_body_table_frames_a_body_and_pins_its_hottest_point() {
    let mut h = Harness::new();
    exec(&mut h, "solid.box", json!({"length": 20, "width": 20, "height": 10, "body_name": "Cool"}));
    exec(&mut h, "solid.box", json!({"length": 6, "width": 6, "height": 4, "corner": [40, 0, 0], "body_name": "Hot"}));
    exec(&mut h, "sim.study.set", json!({"name": "Thermal", "set": {"cell_size_mm": 2.0, "padding_mm": 8.0, "air": {"mixing_factor": 50.0}}}));
    exec(&mut h, "sim.body.set", json!({"study": "Thermal", "body": "Hot", "power_w": 2.0}));
    exec(&mut h, "sim.solve", json!({"study": "Thermal"}));
    h.call("ui.workspace", json!({"workspace": "simulation"}));
    h.frames(3);
    let hot = crate::scenario::handle_at("simulation:body:Hot").expect("a row for Hot");
    let cool = crate::scenario::handle_at("simulation:body:Cool").expect("a row for Cool");
    assert!(hot.y < cool.y, "hottest first");
    let before = h.app.cam;
    h.step(&json!({"click": [hot.x, hot.y]})).unwrap();
    h.settle();
    assert!(h.app.cam.target.x > before.target.x + 5.0, "framed the hot body: {:?} → {:?}", before.target, h.app.cam.target);
    let probes = h.app.session.simulation_view.probes.clone();
    assert_eq!(probes.len(), 1);
    let run = h.app.session.simulation.clone().unwrap();
    let reading = solidcraft_engine::cmd::simulation_view::reading(&run, probes[0].point, None).unwrap();
    let max = run.results.summary.bodies.iter().find(|b| b.name == "Hot").unwrap().max_c;
    assert!((reading.temperature_c - max).abs() < 1e-9, "pinned at its hottest cell");
}

#[test]
fn fans_are_drawn_and_picked_and_streamlines_shown() {
    let mut h = Harness::new();
    exec(&mut h, "solid.box", json!({"length": 20, "width": 20, "height": 10}));
    exec(
        &mut h,
        "sim.study.set",
        json!({"name": "Thermal", "set": {"cell_size_mm": 2.0, "padding_mm": 12.0, "air": {"mixing_factor": 1.0},
        "walls": ["open", "open", "closed", "closed", "closed", "closed"], "flow": {"max_steps": 600, "buoyancy": false}}}),
    );
    exec(&mut h, "sim.body.set", json!({"study": "Thermal", "body": "Body1", "power_w": 0.5}));
    exec(
        &mut h,
        "sim.fan.set",
        json!({"study": "Thermal", "name": "Intake", "region": {"min": [-11, -12, -12], "max": [-8, 32, 22]}, "velocity": [1.5, 0, 0]}),
    );
    h.call("ui.workspace", json!({"workspace": "simulation"}));
    h.frames(3);
    // before any solve the setup shows: the fan's label is in the view
    let fan = crate::scenario::handle_at("simulation:fan:0").expect("the fan's label");
    h.step(&json!({"click": [fan.x, fan.y]})).unwrap();
    h.frames(2);
    assert_eq!(h.app.simulation_workspace.selected_fan, Some(0));
    exec(&mut h, "sim.solve", json!({"study": "Thermal"}));
    // the panel's Streamlines choice
    h.frames(2);
    let [x, y] = h.widget(crate::i18n::tr("sim.display_streamlines")).expect("a Streamlines choice");
    h.step(&json!({"click": [x, y]})).unwrap();
    h.frames(2);
    assert_eq!(h.app.session.simulation_display, SimulationDisplay::Streamlines);
    // the scene is built with the lines (the viewport builds it only with a GPU target)
    crate::ui_shot::enable_viewport(&mut h);
    h.frames(1);
    let with = super::glyphs(&h.app).unwrap();
    assert_eq!(with.boxes.len(), 1);
    let mut sc = crate::gpu::GpuScene::default();
    super::add_to_scene(&h.app, &mut sc, Some((20.0, 30.0)));
    let streamlines = sc.lines.len() / crate::gpu::LINE_SIZE;
    exec(&mut h, "sim.display", json!({"mode": "temperature"}));
    let mut sc = crate::gpu::GpuScene::default();
    super::add_to_scene(&h.app, &mut sc, Some((20.0, 30.0)));
    let now = sc.lines.len() / crate::gpu::LINE_SIZE;
    assert!(streamlines > now + 50, "{streamlines} vs {now}");
}

/// Pictures of each feature for review, written to `$SIM_VISUALS_SHOTS` (run with
/// `--ignored`; needs a GPU adapter, lavapipe will do).
#[test]
#[ignore = "writes review pictures; run by hand with SIM_VISUALS_SHOTS set"]
fn review_pictures() {
    let Some(dir) = std::env::var_os("SIM_VISUALS_SHOTS") else { return };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = Harness::new();
    let script: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/simulation-airflow.json")).unwrap())
            .unwrap();
    for c in script["commands"].as_array().unwrap() {
        exec(&mut h, c["command"].as_str().unwrap(), c["params"].clone());
    }
    h.call("ui.workspace", json!({"workspace": "simulation"}));
    crate::ui_shot::enable_viewport(&mut h);
    h.app.animate_view("home");
    h.settle();
    let shot = |h: &mut Harness, name: &str| {
        h.frames(2);
        crate::ui_shot::png(h, &dir.join(format!("{name}.png"))).expect("a GPU adapter");
    };
    exec(&mut h, "sim.display", json!({"mode": "temperature"}));
    shot(&mut h, "4-setup-glyphs");
    exec(&mut h, "sim.slice", json!({"axis": "z", "off": true}));
    exec(&mut h, "sim.slice", json!({"axis": "y", "position": 50.0}));
    shot(&mut h, "1-slice-temperature");
    exec(&mut h, "sim.slice", json!({"axis": "y", "off": true}));
    exec(&mut h, "sim.slice", json!({"axis": "y", "position": 50.0, "field": "speed"}));
    shot(&mut h, "1-slice-speed");
    exec(&mut h, "sim.slice", json!({"off": true}));
    exec(&mut h, "sim.display", json!({"mode": "streamlines", "seeds": 120}));
    shot(&mut h, "2-streamlines");
    exec(&mut h, "sim.display", json!({"mode": "temperature"}));
    exec(&mut h, "sim.slice", json!({"axis": "z", "position": 12.0}));
    let run = h.app.session.simulation.clone().unwrap();
    let (cpu, _) = solidcraft_engine::cmd::simulation_view::hottest_point(&run, "CPU").unwrap();
    exec(&mut h, "sim.probe.pin", json!({"point": [cpu.x, cpu.y, 6.0], "body": "CPU"}));
    exec(&mut h, "sim.probe.pin", json!({"point": [150, 50, 12]}));
    h.step(&json!({"move": {"world": [125, 80, 33]}})).unwrap();
    shot(&mut h, "3-probes-hover");
    exec(&mut h, "sim.slice", json!({"off": true}));
    exec(&mut h, "sim.probe.delete", json!({"all": true}));
    let row = crate::scenario::handle_at("simulation:body:CPU").unwrap();
    h.step(&json!({"click": [row.x, row.y]})).unwrap();
    h.settle();
    shot(&mut h, "5-summary-focus");
}
