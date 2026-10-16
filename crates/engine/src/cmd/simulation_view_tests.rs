use serde_json::json;
use solidcraft_geom::Vec3;

use super::*;
use crate::{Session, SimulationDisplay};

/// A 20 x 20 x 10 block heated with 1 W, solved on 2 mm cells with 10 mm of air around it.
fn solved_block() -> Session {
    let mut s = Session::default();
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 10})).unwrap();
    s.execute("sim.study.set", &json!({"name": "T", "set": {"cell_size_mm": 2.0, "padding_mm": 10.0, "air": {"mixing_factor": 50.0}}})).unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 1.0})).unwrap();
    s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    s
}

/// The block in a duct with a fan blowing +x through it.
fn solved_duct() -> Session {
    let mut s = Session::default();
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 10})).unwrap();
    s.execute(
        "sim.study.set",
        &json!({"name": "T", "set": {"cell_size_mm": 2.0, "padding_mm": 12.0, "air": {"mixing_factor": 1.0},
        "walls": ["open", "open", "closed", "closed", "closed", "closed"], "flow": {"max_steps": 800, "buoyancy": false}}}),
    )
    .unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 0.5})).unwrap();
    s.execute(
        "sim.fan.set",
        &json!({"study": "T", "name": "Intake", "region": {"min": [-11, -12, -12], "max": [-8, 32, 22]}, "velocity": [1.5, 0, 0]}),
    )
    .unwrap();
    s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    s
}

#[test]
fn slices_are_one_per_axis_and_cut_every_cell_of_the_plane() {
    let mut s = Session::default();
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 10})).unwrap();
    // before a solve a slice needs its position
    assert!(s.execute("sim.slice", &json!({"axis": "z"})).is_err());
    s.execute("sim.slice", &json!({"axis": "z", "position": 5})).unwrap();
    assert_eq!(s.simulation_view.slices.len(), 1);
    s.execute("sim.slice", &json!({"off": true})).unwrap();
    let mut s = solved_block();
    let undo = s.undo.len();
    // the middle of the domain by default; one slice per axis, replaced in place
    let r = s.execute("sim.slice", &json!({"axis": "z"})).unwrap();
    assert_eq!(r["slices"][0]["position"], 5.0, "{r}");
    assert_eq!(r["inside"], true);
    s.execute("sim.slice", &json!({"axis": "x", "position": 10.0})).unwrap();
    s.execute("sim.slice", &json!({"axis": "z", "position": 4.0})).unwrap();
    let r = s.execute("sim.slice", &json!({})).unwrap();
    assert_eq!(r["slices"].as_array().unwrap().len(), 2);
    assert_eq!(r["slices"][0]["axis"], "x", "kept in axis order");
    assert_eq!(r["slices"][1]["position"], 4.0);
    assert_eq!(s.undo.len(), undo, "slices are view state, not undo steps");
    // no airflow was solved: a speed slice is refused
    assert!(s.execute("sim.slice", &json!({"axis": "y", "field": "speed"})).is_err());
    // hostile input
    for bad in [
        json!({"axis": "w"}),
        json!({"axis": 3}),
        json!({"axis": "x", "position": "1"}),
        json!({"axis": "x", "position": 1e300}),
        json!({"axis": "x", "field": "pressure"}),
        json!({"axis": "x", "off": "yes"}),
        json!({"axis": "x", "colour": "red"}),
        json!({"position": 2}),
        json!([1, 2]),
    ] {
        assert!(s.execute("sim.slice", &bad).is_err(), "{bad}");
    }
    // the z slice covers every cell of its plane, coloured by the cells' temperatures
    let run = s.simulation.clone().unwrap();
    let g = &run.results.grid;
    let z = *s.simulation_view.slice(SliceAxis::Z).unwrap();
    let range = temperature_scale(&run, &s.simulation_view);
    let (glo, ghi) = run.results.temperature_range();
    assert!((range.0 - glo).abs() < 1e-9 && (range.1 - ghi).abs() < 1e-9, "a temperature slice spans the whole grid's range");
    let quads = slice_quads(&run, &z, range);
    assert_eq!(quads.corners.len(), g.nx * g.ny);
    assert_eq!(quads.stride, 1);
    assert!(quads.corners.iter().flatten().all(|p| (p.z - 4.0).abs() < 1e-9));
    // hottest in the block, coolest at the domain edge
    let hot = heat_color(1.0);
    let centre = quads.corners.iter().position(|q| q[0].x <= 10.0 && q[2].x > 10.0 && q[0].y <= 10.0 && q[2].y > 10.0).unwrap();
    let edge = 0;
    let warmth = |c: [u8; 3]| i32::from(c[0]) - i32::from(c[2]);
    assert!(warmth(quads.colors[centre]) > warmth(quads.colors[edge]), "{:?} vs {:?}", quads.colors[centre], quads.colors[edge]);
    assert!(warmth(quads.colors[centre]) <= warmth(hot));
    // a plane outside the grid draws nothing
    assert!(slice_quads(&run, &Slice { axis: SliceAxis::X, position: 1e6, field: SliceField::Temperature }, range).corners.is_empty());
    // without a temperature slice the scale is the bodies' own
    s.execute("sim.slice", &json!({"axis": null})).unwrap();
    assert!(s.simulation_view.slices.is_empty());
    assert_eq!(temperature_scale(&run, &s.simulation_view), body_range(&run));
}

#[test]
fn a_fine_slice_is_sampled_down_to_the_quad_cap() {
    let mut s = Session::default();
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 2, "corner": [140, 140, 0]})).unwrap();
    s.execute("sim.study.set", &json!({"name": "T", "set": {"cell_size_mm": 1.0, "domain": {"min": [0, 0, 0], "max": [300, 300, 2]}}})).unwrap();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 0.1})).unwrap();
    s.execute("sim.solve", &json!({"study": "T"})).unwrap();
    s.execute("sim.slice", &json!({"axis": "z", "position": 1.0})).unwrap();
    let run = s.simulation.clone().unwrap();
    let z = *s.simulation_view.slice(SliceAxis::Z).unwrap();
    let quads = slice_quads(&run, &z, temperature_scale(&run, &s.simulation_view));
    assert_eq!(quads.stride, 2);
    assert_eq!(quads.corners.len(), 150 * 150);
    assert!(quads.corners.len() <= MAX_SLICE_QUADS);
    // the quads still tile the whole plane
    let area: f64 = quads.corners.iter().map(|q| (q[2].x - q[0].x) * (q[2].y - q[0].y)).sum();
    assert!((area - 300.0 * 300.0).abs() < 1e-6, "{area}");
}

#[test]
fn probes_pin_read_and_delete() {
    let mut s = Session::default();
    s.execute("solid.box", &json!({"length": 20, "width": 20, "height": 10})).unwrap();
    assert!(s.execute("sim.probe.pin", &json!({"point": [10, 10, 5]})).is_err(), "nothing solved");
    let mut s = solved_block();
    let undo = s.undo.len();
    // on the top face of the block: its surface temperature, as the colour map shows it
    let r = s.execute("sim.probe.pin", &json!({"point": [10, 10, 10], "body": "Body1"})).unwrap();
    assert_eq!(r["index"], 0);
    assert_eq!(r["reading"]["body"], "Body1");
    let run = s.simulation.clone().unwrap();
    let surface = run.results.surface_temperature(0, Vec3::new(10.0, 10.0, 10.0)).unwrap();
    assert!((r["reading"]["temperature_c"].as_f64().unwrap() - surface).abs() < 1e-9);
    // in the air: the cell's temperature, cooler than the block
    let air = s.execute("sim.probe.pin", &json!({"point": [-8, 10, 5]})).unwrap();
    assert!(air["reading"]["body"].is_null());
    assert!(air["reading"]["temperature_c"].as_f64().unwrap() < surface);
    // an unknown body name reads the cell there
    let r = s.execute("sim.probe.pin", &json!({"point": [10, 10, 5], "body": "Nope"})).unwrap();
    assert_eq!(r["reading"]["body"], "Body1");
    let list = s.execute("sim.probe.list", &json!({})).unwrap();
    assert_eq!(list["probes"].as_array().unwrap().len(), 3);
    assert_eq!(list["probes"][1]["index"], 1);
    s.execute("sim.probe.delete", &json!({"index": 1})).unwrap();
    assert_eq!(s.simulation_view.probes.len(), 2);
    assert!(s.simulation_view.probes.iter().all(|p| p.point.x == 10.0));
    for bad in [
        json!({"point": [1e9, 0, 0]}),
        json!({"point": [f64::MAX, 0, 0]}),
        json!({"point": [0, 0]}),
        json!({"point": [10, 10, 5], "body": ""}),
        json!({"point": [10, 10, 5], "extra": 1}),
    ] {
        assert!(s.execute("sim.probe.pin", &bad).is_err(), "{bad}");
    }
    assert!(s.execute("sim.probe.delete", &json!({"index": 9})).is_err());
    assert!(s.execute("sim.probe.delete", &json!({})).is_err());
    assert!(s.execute("sim.probe.list", &json!({"x": 1})).is_err());
    // capped
    s.execute("sim.probe.delete", &json!({"all": true})).unwrap();
    for _ in 0..MAX_PROBES {
        s.execute("sim.probe.pin", &json!({"point": [10, 10, 5]})).unwrap();
    }
    assert!(s.execute("sim.probe.pin", &json!({"point": [10, 10, 5]})).is_err());
    assert_eq!(s.undo.len(), undo, "probes are view state");
    // the hottest point of the block is inside it, at its maximum
    let (p, t) = hottest_point(&run, "Body1").unwrap();
    assert!(p.x > 0.0 && p.x < 20.0 && p.y > 0.0 && p.y < 20.0 && p.z > 0.0 && p.z < 10.0);
    assert!((t - run.results.summary.bodies[0].max_c).abs() < 1e-9);
    assert!(hottest_point(&run, "Nope").is_none());
}

#[test]
fn streamlines_follow_the_fan_through_the_duct() {
    let mut s = solved_duct();
    let run = s.simulation.clone().unwrap();
    let study = s.doc.simulation.studies.get("T").cloned();
    let lines = streamlines(&run, study.as_ref(), 40);
    assert!(lines.len() >= 10, "{} lines", lines.len());
    let b = run.results.grid.bounds();
    let mut downstream = 0;
    for l in &lines {
        assert_eq!(l.points.len(), l.colors.len());
        assert!(l.points.iter().all(|p| p.x >= b.min.x - 1e-6 && p.x <= b.max.x + 1e-6 && p.z >= b.min.z - 1e-6 && p.z <= b.max.z + 1e-6));
        // inside the block is never reached
        assert!(l.points.iter().all(|p| !(p.x > 1.0 && p.x < 19.0 && p.y > 1.0 && p.y < 19.0 && p.z > 1.0 && p.z < 9.0)), "through the block");
        if l.points.last().unwrap().x > 25.0 {
            downstream += 1;
        }
    }
    assert!(downstream * 2 >= lines.len(), "most lines carry on past the block: {downstream} of {}", lines.len());
    let total: usize = lines.iter().map(|l| l.points.len()).sum();
    assert!(total <= MAX_STREAMLINE_POINTS);
    // the display mode and a bounded seed count
    let r = s.execute("sim.display", &json!({"mode": "streamlines", "seeds": 12})).unwrap();
    assert_eq!(r["mode"], "streamlines");
    assert_eq!(s.simulation_display, SimulationDisplay::Streamlines);
    assert_eq!(s.simulation_view.seeds, 12);
    assert!(s.execute("sim.display", &json!({"seeds": 0})).is_err());
    assert!(s.execute("sim.display", &json!({"seeds": 100_000})).is_err());
    assert!(s.execute("sim.display", &json!({"seeds": -1})).is_err());
    assert_eq!(s.simulation_view.seeds, 12);
    // and the picture draws them
    let cam = solidcraft_render::Camera::default();
    let lines_before = crate::view::scene(&s, &cam).lines.len();
    s.execute("sim.display", &json!({"mode": "temperature"})).unwrap();
    let lines_after = crate::view::scene(&s, &cam).lines.len();
    assert!(lines_before > lines_after + 10, "{lines_before} vs {lines_after}");
    // a speed slice through the duct
    s.execute("sim.slice", &json!({"axis": "y", "field": "speed"})).unwrap();
    let y = *s.simulation_view.slice(SliceAxis::Y).unwrap();
    let quads = slice_quads(&run, &y, temperature_scale(&run, &s.simulation_view));
    assert!(quads.colors.contains(&SOLID_GREY), "the block shows grey on a speed slice");
    assert!(shows_speed(&run, &s.simulation_view, s.simulation_display));
    // a ray from above meets it in the middle of the domain
    let (p, _, hit) = slice_hit(&run, &s.simulation_view, Vec3::new(-5.0, -100.0, 5.0), Vec3::Y).unwrap();
    assert_eq!(hit.axis, SliceAxis::Y);
    assert!((p.y - y.position).abs() < 1e-9);
    assert!(slice_hit(&run, &s.simulation_view, Vec3::new(-5.0, -100.0, 5.0), Vec3::X).is_none());
    let r = reading(&run, p, None).unwrap();
    assert!(r.body.is_none() && r.speed_m_s.unwrap() > 0.3, "{r:?}");
    assert!(reading_text(&r, "Air").starts_with("Air · "));
}

#[test]
fn setup_glyphs_show_fans_sources_and_the_domain() {
    let mut s = solved_duct();
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "power_w": 3.0})).unwrap();
    let study = s.doc.simulation.studies.get("T").cloned().unwrap();
    let run = s.simulation.clone().unwrap();
    let g = setup_glyphs(&s, &study, Some(&run));
    assert_eq!(g.boxes.len(), 1);
    assert_eq!(g.lines.iter().filter(|l| l.color == FAN_COLOR).count(), 12, "box edges");
    assert_eq!(g.lines.iter().filter(|l| l.color == FAN_ARROW && l.on_top).count(), 5, "an arrow over the model");
    let fan = g.labels.iter().find(|l| l.kind == GlyphKind::Fan(0)).unwrap();
    assert_eq!(fan.text, "Intake · 1.5 m/s");
    let heat = g.labels.iter().find(|l| l.kind == GlyphKind::Heat).unwrap();
    assert_eq!(heat.text, "⚡ 3 W");
    assert_eq!(g.domain, Some(run.results.grid.bounds()));
    assert!(g.lines.iter().filter(|l| l.color == DOMAIN_COLOR).count() > 12, "a dashed outline");
    // a fixed temperature wins over power
    s.execute("sim.body.set", &json!({"study": "T", "body": "Body1", "fixed_temperature_c": 20.0})).unwrap();
    let study = s.doc.simulation.studies.get("T").cloned().unwrap();
    let g = setup_glyphs(&s, &study, None);
    assert_eq!(g.labels.iter().find(|l| l.kind == GlyphKind::Fixed).unwrap().text, "❄ 20 °C");
    // without a solve the domain is the bodies grown by the padding
    let d = g.domain.unwrap();
    assert!((d.min.x + 12.0).abs() < 1e-9 && (d.max.z - 22.0).abs() < 1e-9, "{d:?}");
    // pictures draw the glyphs unless switched off
    let cam = solidcraft_render::Camera::default();
    let with = crate::view::scene(&s, &cam).lines.len();
    s.execute("sim.display", &json!({"setup": false})).unwrap();
    let without = crate::view::scene(&s, &cam).lines.len();
    assert!(with >= without + 17, "{with} vs {without}");
}

#[test]
fn a_shown_slice_is_a_picture_mesh_in_field_colours() {
    let mut s = solved_block();
    let cam = solidcraft_render::Camera::default();
    let before = crate::view::scene(&s, &cam).meshes.len();
    s.execute("sim.slice", &json!({"axis": "x", "position": 10.0})).unwrap();
    let sc = crate::view::scene(&s, &cam);
    assert_eq!(sc.meshes.len(), before + 1);
    let m = sc.meshes.iter().find(|m| m.mesh.positions.iter().all(|p| (p.x - 10.0).abs() < 1e-9)).unwrap();
    let g = &s.simulation.as_ref().unwrap().results.grid;
    assert_eq!(m.mesh.triangles.len(), 2 * g.ny * g.nz);
    assert_eq!(m.vertex_colors.len(), m.mesh.positions.len());
    // pinned probes draw as small crosses
    let lines = sc.lines.len();
    s.execute("sim.probe.pin", &json!({"point": [10, 10, 5]})).unwrap();
    assert_eq!(crate::view::scene(&s, &cam).lines.len(), lines + 3);
    // off hides it all
    s.execute("sim.display", &json!({"mode": "off"})).unwrap();
    assert_eq!(crate::view::scene(&s, &cam).meshes.len(), before);
}
