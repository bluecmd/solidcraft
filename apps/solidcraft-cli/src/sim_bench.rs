//! `solidcraft-cli sim-bench`: a 2U server-sized airflow and heat study on the simulation
//! backend, timed. It measures what a machine can do (million lattice updates per second, the
//! thermal solve time) on a geometry like the one the simulation is for: a 440 × 700 × 88 mm
//! chassis, two CPUs under finned heat sinks, a fan wall and memory, open front and back.

use std::sync::Arc;

use serde_json::json;
use solidcraft_engine::geom::{Aabb3, Vec3};
use solidcraft_engine::sim::{Accelerator, Air, CpuAccelerator, Fan, FlowSettings, Limits, Problem, SolidBody, Thermal, Wall, solve_with};

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
    let triangles =
        vec![[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]];
    SolidBody { name: name.into(), positions, triangles, conductivity: k, power_w: power, fixed_temperature_c: None, fluid: None }
}

/// The benchmark chassis: x across (440 mm), y front to back (700 mm), z up (88 mm).
pub fn chassis(cell: f64, power_w: f64) -> Problem {
    let mut bodies = vec![boxed("Board", [10.0, 120.0, 4.0], [430.0, 600.0, 6.0], 0.3, 0.0)];
    for (n, cx) in [(1, 130.0), (2, 310.0)] {
        // CPU package, heat sink base and 20 fins of 0.8 mm at 2.4 mm pitch, fins along y (airflow)
        bodies.push(boxed(&format!("CPU{n}"), [cx - 35.0, 300.0, 6.0], [cx + 35.0, 370.0, 10.0], 150.0, power_w));
        bodies.push(boxed(&format!("Sink{n} base"), [cx - 45.0, 290.0, 10.0], [cx + 45.0, 380.0, 16.0], 385.0, 0.0));
        for f in 0..20 {
            let x0 = cx - 44.0 + 4.5 * f64::from(f);
            bodies.push(boxed(&format!("Sink{n} fin{f}"), [x0, 290.0, 16.0], [x0 + 1.6, 380.0, 80.0], 205.0, 0.0));
        }
    }
    for d in 0..8 {
        let x = 30.0 + 50.0 * f64::from(d);
        bodies.push(boxed(&format!("DIMM{d}"), [x, 420.0, 6.0], [x + 6.0, 560.0, 40.0], 20.0, 5.0));
    }
    bodies.push(boxed("PSU", [330.0, 610.0, 4.0], [430.0, 696.0, 84.0], 5.0, 30.0));
    // the fan wall's bulkhead: sheet metal around the six fans, so the air goes through them
    let (y0, y1) = (212.0, 216.0);
    bodies.push(boxed("Bulkhead floor", [0.0, y0, 0.0], [440.0, y1, 4.0], 50.0, 0.0));
    bodies.push(boxed("Bulkhead top", [0.0, y0, 84.0], [440.0, y1, 88.0], 50.0, 0.0));
    for k in 0..7 {
        let x0 = if k == 0 { 0.0 } else { 70.0 * f64::from(k) };
        let x1 = if k == 6 { 440.0 } else { 70.0 * f64::from(k) + 10.0 };
        bodies.push(boxed(&format!("Bulkhead post{k}"), [x0, y0, 4.0], [x1, y1, 84.0], 50.0, 0.0));
    }
    let mut fans = Vec::new();
    for k in 0..6 {
        let x0 = 10.0 + 70.0 * f64::from(k);
        fans.push(Fan {
            name: format!("Fan{k}"),
            region: Aabb3 { min: Vec3::new(x0, 200.0, 4.0), max: Vec3::new(x0 + 60.0, 230.0, 84.0) },
            velocity: Vec3::new(0.0, 4.0, 0.0),
        });
    }
    Problem {
        domain: Aabb3 { min: Vec3::ZERO, max: Vec3::new(440.0, 700.0, 88.0) },
        cell_size: cell,
        bodies,
        // sheet-metal sides, floor and lid; open front (-y) and back (+y)
        walls: [Wall::Closed, Wall::Closed, Wall::Open, Wall::Open, Wall::Closed, Wall::Closed],
        air: Air::default(),
        fluids: Vec::new(),
        thermal: Thermal { ambient_c: 25.0, film_coefficient: 40.0 },
        fans,
        flow: None,
        limits: Limits { max_cells: solidcraft_engine::sim::MAX_CELLS, max_iterations: 20_000, tolerance: 1e-7 },
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    crate::known_flags(
        "sim-bench",
        args,
        &[
            "--cell",
            "--steps",
            "--power",
            "--json-out",
            "--max-cell-reynolds",
            "--lattice-velocity",
            "--smagorinsky",
            "--trt-lambda",
            "--fan-speed",
            "--sponge-cells",
            "--sponge-viscosity",
        ],
        &["--cpu"],
    )?;
    let num = |name: &str, default: f64| -> Result<f64, String> {
        match crate::flag(args, name) {
            Some(v) => v.parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(|| format!("sim-bench: {name} needs a number ≥ 0")),
            None => Ok(default),
        }
    };
    let cell = num("--cell", 2.0)?;
    let steps = num("--steps", 2000.0)? as usize;
    let power = num("--power", 200.0)?;
    let defaults = FlowSettings::default();
    let reynolds = num("--max-cell-reynolds", defaults.max_cell_reynolds)?;
    let lattice_velocity = num("--lattice-velocity", defaults.lattice_velocity)?;
    let smagorinsky = num("--smagorinsky", defaults.smagorinsky)?;
    let lambda = num("--trt-lambda", defaults.trt_lambda)?;
    let fan_speed = num("--fan-speed", 4.0)?;
    let sponge_cells = num("--sponge-cells", defaults.sponge_cells as f64)? as usize;
    let sponge_viscosity = num("--sponge-viscosity", defaults.sponge_viscosity)?;
    let backend: Arc<dyn Accelerator> = if args.iter().any(|a| a == "--cpu") {
        Arc::new(CpuAccelerator)
    } else {
        let gpu = solidcraft_sim_gpu::GpuAccelerator::headless().map_err(|e| format!("sim-bench: {e}"))?;
        Arc::new(gpu)
    };
    let mut problem = chassis(cell, power);
    for f in &mut problem.fans {
        f.velocity = Vec3::new(0.0, fan_speed, 0.0);
    }
    // a fixed number of steps: no early stop, so runs are comparable
    problem.flow = Some(FlowSettings {
        max_steps: steps.max(1),
        tolerance: 1e-12,
        buoyancy: false,
        max_cell_reynolds: reynolds,
        lattice_velocity,
        smagorinsky,
        trt_lambda: lambda,
        sponge_cells,
        sponge_viscosity,
        ..defaults
    });
    let cells = ((440.0 / cell).ceil() * (700.0 / cell).ceil() * (88.0 / cell).ceil()) as u64;
    eprintln!("sim-bench: {} on {} cells at {cell} mm, {steps} steps", backend.name(), cells);
    let started = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let mut phases: Vec<(String, f64)> = Vec::new();
    let mut phase = String::new();
    let mut progress = |p: &str, done: usize, total: usize| -> bool {
        if p != phase {
            phases.push((p.to_string(), started.elapsed().as_secs_f64()));
            phase = p.to_string();
        }
        if last.elapsed().as_secs() >= 10 {
            eprintln!("  {p} {done}/{total} at {:.0} s", started.elapsed().as_secs_f64());
            last = std::time::Instant::now();
        }
        true
    };
    let r = solve_with(&problem, &*backend, &mut progress).map_err(|e| format!("sim-bench: {e}"))?;
    let total = started.elapsed().as_secs_f64();
    let flow = r.summary.flow.clone();
    let hottest: Vec<_> = {
        let mut b: Vec<_> =
            r.summary.bodies.iter().filter(|b| b.power_w > 0.0).map(|b| json!({"body": b.name, "max_c": b.max_c, "mean_c": b.mean_c})).collect();
        b.sort_by(|a, c| c["max_c"].as_f64().partial_cmp(&a["max_c"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
        b
    };
    let report = json!({
        "backend": backend.name(),
        "cell_mm": cell,
        "cells": r.summary.cells,
        "total_cells": cells,
        "air_cells": r.summary.air_cells,
        "steps": flow.as_ref().map(|f| f.steps),
        "mlups": flow.as_ref().map(|f| f.mlups),
        "max_lattice_speed": flow.as_ref().map(|f| f.max_lattice_speed),
        "max_speed_m_s": flow.as_ref().map(|f| f.max_speed_m_s),
        "thermal_method": r.summary.thermal_method,
        "thermal_iterations": r.summary.thermal_iterations,
        "thermal_converged": r.summary.thermal_converged,
        "heat_in_w": r.summary.heat_in_w,
        "heat_out_w": r.summary.heat_out_w,
        "phases_started_s": phases.iter().map(|(p, t)| json!({"phase": p, "at_s": t})).collect::<Vec<_>>(),
        "total_s": total,
        "hottest": hottest,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    match crate::flag(args, "--json-out") {
        Some(path) => std::fs::write(path, &text).map_err(|e| format!("sim-bench: {path}: {e}"))?,
        None => println!("{text}"),
    }
    Ok(())
}
