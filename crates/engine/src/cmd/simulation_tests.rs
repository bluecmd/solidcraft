use serde_json::json;

use crate::{Session, SimulationDisplay};

fn session_with_block() -> Session {
    let mut s = Session::default();
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 10})).unwrap();
    s
}

#[test]
fn study_edits_are_one_undo_step_and_need_no_evaluation() {
    let mut s = session_with_block();
    let evaluations = s.evaluations;
    let undo = s.undo.len();
    let r = s.execute("sim.study.set", &json!({"name": "Thermal", "set": {"cell_size_mm": 2.0, "thermal": {"ambient_c": 30.0}}})).unwrap();
    assert_eq!(r["study"]["cell_size_mm"], 2.0);
    assert_eq!(r["study"]["thermal"]["ambient_c"], 30.0);
    assert_eq!(r["study"]["thermal"]["film_coefficient"], 8.0, "untouched fields keep their defaults");
    assert_eq!(s.undo.len(), undo + 1);
    assert_eq!(s.evaluations, evaluations, "a study edit does not re-evaluate the model");
    // identical edit: no-op
    let doc = s.doc.clone();
    s.execute("sim.study.set", &json!({"name": "Thermal", "set": {"cell_size_mm": 2.0}})).unwrap();
    assert!(std::sync::Arc::ptr_eq(&doc, &s.doc));
    // round trip through the design file
    let back = solidcraft_doc::Document::from_json(&s.doc.to_json()).unwrap();
    assert_eq!(back.simulation, s.doc.simulation);
    // body and fan settings
    let r = s.execute("sim.body.set", &json!({"study": "Thermal", "body": "Body1", "power_w": 2.5, "conductivity": 200.0})).unwrap();
    assert_eq!(r["setup"]["power_w"], 2.5);
    let r = s.execute("sim.body.set", &json!({"study": "Thermal", "body": "Body1", "conductivity": null})).unwrap();
    assert!(r["setup"]["conductivity"].is_null());
    assert_eq!(r["setup"]["power_w"], 2.5, "merging keeps the other fields");
    s.execute(
        "sim.fan.set",
        &json!({"study": "Thermal", "name": "Intake", "region": {"min": [-5, 0, 0], "max": [-2, 20, 10]}, "velocity": [1, 0, 0]}),
    )
    .unwrap();
    assert_eq!(s.doc.simulation.studies["Thermal"].fans.len(), 1);
    s.execute("sim.fan.delete", &json!({"study": "Thermal", "name": "Intake"})).unwrap();
    assert!(s.doc.simulation.studies["Thermal"].fans.is_empty());
    assert!(s.execute("sim.fan.delete", &json!({"study": "Thermal", "name": "Intake"})).is_err());
    let r = s.execute("sim.study.get", &json!({})).unwrap();
    assert!(r["studies"]["Thermal"].is_object());
    s.execute("sim.study.delete", &json!({"name": "Thermal"})).unwrap();
    assert!(s.doc.simulation.is_empty());
    // invalid edits are atomic
    assert!(s.execute("sim.study.set", &json!({"name": "X", "set": {"cell_size_mm": -1}})).is_err());
    assert!(s.execute("sim.study.set", &json!({"name": "X", "set": {"nonsense": 1}})).is_err());
    assert!(s.doc.simulation.is_empty());
}

#[test]
fn a_heated_block_in_free_air_solves_and_can_be_probed() {
    let mut s = session_with_block();
    s.execute("material.assign", &json!({"bodies": ["Body1"], "material": "Aluminum"})).unwrap();
    s.execute("sim.study.set", &json!({"name": "Thermal", "set": {"cell_size_mm": 2.0, "padding_mm": 10.0, "air": {"mixing_factor": 1e5}}})).unwrap();
    s.execute("sim.body.set", &json!({"study": "Thermal", "body": "Body1", "power_w": 1.0})).unwrap();
    assert!(s.execute("sim.results", &json!({})).is_err(), "nothing solved yet");
    let revision = s.revision;
    let r = s.execute("sim.solve", &json!({"study": "Thermal"})).unwrap();
    assert_eq!(s.revision, revision, "solving changes no design state");
    assert_eq!(r["bodies"][0]["name"], "Body1");
    assert_eq!(r["bodies"][0]["cells"], 20 * 20 * 10 / 8);
    // 1 W over 6 faces of 20x20, 20x20, 20x10 ×2 ... = 2*(400+200+200) mm² = 1.6e-3 m² at h = 8
    let expected = 20.0 + 1.0 / (8.0 * 1.6e-3);
    let mean = r["bodies"][0]["mean_c"].as_f64().unwrap();
    assert!((mean - expected).abs() < 0.05 * (expected - 20.0), "mean {mean} vs {expected}: {r}");
    assert!(r["thermal_converged"].as_bool().unwrap());
    assert!((r["heat_out_w"].as_f64().unwrap() - 1.0).abs() < 1e-3);
    assert_eq!(r["stale"], false);
    assert_eq!(s.simulation_display, SimulationDisplay::Temperature);
    let p = s.execute("sim.probe", &json!({"point": [10, 10, 5]})).unwrap();
    assert_eq!(p["body"], "Body1");
    assert!((p["temperature_c"].as_f64().unwrap() - mean).abs() < 1.0);
    assert!(p["velocity_m_s"].is_null());
    assert!(s.execute("sim.probe", &json!({"point": [1000, 0, 0]})).is_err());
    let r2 = s.execute("sim.results", &json!({})).unwrap();
    assert_eq!(r2["max_c"], r["max_c"]);
    // a geometry change makes the result stale
    s.execute("solid.box", &json!({"length": 5, "width": 5, "height": 5})).unwrap();
    assert_eq!(s.execute("sim.results", &json!({})).unwrap()["stale"], true);
    let d = s.execute("sim.display", &json!({"mode": "off"})).unwrap();
    assert_eq!(d["solved"], true);
    assert_eq!(s.simulation_display, SimulationDisplay::Off);
    assert!(s.execute("sim.display", &json!({"mode": "sideways"})).is_err());
}

#[test]
fn export_writes_a_vtk_file_once() {
    let mut s = session_with_block();
    s.execute("sim.study.set", &json!({"name": "T", "set": {"cell_size_mm": 5.0}})).unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 0.5})).unwrap();
    let dir = std::env::temp_dir().join(format!("solidcraft-sim-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("out.vtk");
    let p = path.to_string_lossy().to_string();
    assert!(s.execute("sim.export", &json!({"path": p})).is_err(), "nothing solved yet");
    s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    assert!(s.execute("sim.export", &json!({"path": dir.join("out.txt").to_string_lossy()})).is_err());
    let r = s.execute("sim.export", &json!({"path": p})).unwrap();
    assert!(r["bytes"].as_u64().unwrap() > 100);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("# vtk DataFile Version 3.0"));
    assert!(s.execute("sim.export", &json!({"path": p})).is_err(), "no silent overwrite");
    s.execute("sim.export", &json!({"path": p, "overwrite": true})).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn solving_refuses_what_it_cannot_do() {
    let mut s = session_with_block();
    assert!(s.execute("sim.solve", &json!({"study": "nope"})).is_err());
    s.execute(
        "sim.study.set",
        &json!({"name": "T", "set": {"cell_size_mm": 5.0, "walls": ["closed", "closed", "closed", "closed", "closed", "closed"]}}),
    )
    .unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 1.0})).unwrap();
    let e = s.execute("sim.solve", &json!({"study": "T"})).unwrap_err();
    assert!(e.to_string().contains("nowhere to go"), "{e}");
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "exclude": true})).unwrap();
    let e = s.execute("sim.solve", &json!({"study": "T"})).unwrap_err();
    assert!(e.to_string().contains("no visible bodies"), "{e}");
    // a tiny cell cap is a clear refusal, not a hang
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "exclude": false})).unwrap();
    s.execute(
        "sim.study.set",
        &json!({"name": "T", "set": {"walls": ["open", "open", "open", "open", "open", "open"], "limits": {"max_cells": 10}}}),
    )
    .unwrap();
    let e = s.execute("sim.solve", &json!({"study": "T"})).unwrap_err();
    assert!(e.to_string().contains("exceeds the limit"), "{e}");
}

#[test]
fn a_fan_cools_the_block() {
    let mut s = session_with_block();
    s.execute(
        "sim.study.set",
        &json!({"name": "T", "set": {"cell_size_mm": 2.0, "padding_mm": 12.0, "air": {"mixing_factor": 1.0},
        "walls": ["open", "open", "closed", "closed", "closed", "closed"]}}),
    )
    .unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 0.5})).unwrap();
    let still = s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    s.execute("sim.fan.set", &json!({"study": "T", "name": "fan", "region": {"min": [-11, -12, -12], "max": [-8, 32, 22]}, "velocity": [1.5, 0, 0]}))
        .unwrap();
    s.execute("sim.study.set", &json!({"name": "T", "set": {"flow": {"max_steps": 1500, "buoyancy": false}}})).unwrap();
    let blown = s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    let (a, b) = (still["bodies"][0]["mean_c"].as_f64().unwrap(), blown["bodies"][0]["mean_c"].as_f64().unwrap());
    assert!(b < a, "with a fan {b} < still {a}");
    assert!(blown["flow"]["max_speed_m_s"].as_f64().unwrap() > 1.0, "{blown}");
    let p = s.execute("sim.probe", &json!({"point": [-5, 10, 5]})).unwrap();
    assert!(p["velocity_m_s"][0].as_f64().unwrap() > 0.5, "{p}");
    // the override keeps the air still without touching the study
    let r = s.execute("sim.solve", &json!({"study": "T", "flow": false})).unwrap();
    assert!(r["flow"].is_null());
}
