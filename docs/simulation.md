# Simulation: heat and airflow

The Simulation workspace answers two questions about an enclosure or a board: how hot
do the parts get, and where does the air go. It is a coarse, fast model meant for design
decisions (fan or no fan, vent here or there, is this heat sink enough), not a certified
CFD code. Everything is a command (`sim.*`), so the CLI, scripts, the control channel and
MCP agents reach the same solver as the workspace panel.

## Model

The visible bodies are voxelised on a uniform grid over a domain box (the bodies' bounds
plus a padding, or a box you set). Each cell is solid (one body) or air.

**Temperature** is steady-state conjugate heat transfer on that grid: a 7-point
finite-volume stencil solved with conjugate gradients (BiCGSTAB when air moves).

- Solids conduct with their material's conductivity: the built-in physical materials have
  textbook values (`material_conductivity` in `assets/materials/library.rs`); a design's own
  materials and the default count as a generic plastic (0.2 W/mK). A study can override
  any body's conductivity.
- Heat sources are powers per body (W), spread over the body's volume.
- At every solid/air face a **film coefficient** (W/m²K) stands in for the boundary layer the
  grid cannot resolve: 5–10 for natural convection, 20–100 for forced air. It is a setting,
  not computed from the flow.
- Air cells conduct with the air's conductivity times a **mixing factor** when the air is
  still (1 for truly still air, 5–20 for a sealed enclosure with natural convection, 100 or
  more for a part in free air). When airflow is solved the factor is replaced by the eddy
  diffusivity that goes with the flow's viscosity, and the air also carries heat by advection
  (first-order upwind).
- Domain walls are **open** (ambient temperature; air may enter and leave) or **closed**
  (insulated, no flow). A body may be held at a **fixed temperature** (a cold plate, a heat
  sink outside the study). At least one open wall or fixed body is needed, or the heat has
  nowhere to go and the solve refuses.
- Radiation is not modelled.

**Airflow** is a D3Q19 lattice-Boltzmann solver on the air cells, run until the velocity
field stops changing (a steady or time-averaged flow).

- Fans are axis-aligned regions of air driven at a velocity vector; put one just inside an
  opening, or across a duct.
- Solid cells and closed walls bounce back; open walls are pressure outlets that also let air
  in.
- Two-relaxation-time collision (Λ = 1/12) with a Smagorinsky eddy viscosity, plus a viscosity
  floor that keeps the cell Reynolds number (u·dx/ν) at or below `max_cell_reynolds` (10 by
  default). On a 1–2 mm grid at fan speeds this raises the viscosity well above air's: the
  flow you see is the bulk routing of the air with turbulent mixing standing in for the detail
  the grid cannot carry. A finer grid or a higher limit is more faithful and less stable.
- Buoyancy (Boussinesq, gravity along −Z) is on by default; with it the thermal and flow
  solves alternate `coupling_iterations` times. Turn it off for purely fan-driven cases.

## Commands

| Command | Does |
|---|---|
| `sim.study.set {name, set}` | Create or edit a study (one undo step; fields below). |
| `sim.study.get {name?}` | A study, or all studies and the defaults. |
| `sim.study.delete {name}` | Remove a study. |
| `sim.body.set {study, body, power_w?, conductivity?, fixed_temperature_c?, exclude?}` | A body's heat, conductivity override, fixed temperature or exclusion. `null` clears an override. An instance's definition name applies to every instance. |
| `sim.fan.set {study, name, region: {min, max}, velocity}` / `sim.fan.delete` | Fans (world mm box, m/s). |
| `sim.solve {study, flow?}` | Solve; `flow: true/false` overrides the study's airflow setting for this solve. Returns the summary. |
| `sim.results {}` | The last solve's summary, `stale: true` once the geometry changed. |
| `sim.probe {point}` | Temperature and air velocity at a world point. |
| `sim.export {path, overwrite?}` | Legacy VTK (structured points: body, temperature, velocity, pressure) for ParaView. |
| `sim.display {mode}` | `off`, `temperature` (colour map on the bodies) or `velocity` (arrows too). |

Study fields (`sim.study.set` → `set`): `cell_size_mm` (0 = about 60 cells along the
domain's longest side), `domain {min, max}` or `padding_mm`, `bodies` (as `sim.body.set`),
`walls` (six of `open`/`closed`: −x, +x, −y, +y, −z, +z), `air {conductivity, density,
specific_heat, kinematic_viscosity, mixing_factor, thermal_expansion}`, `thermal {ambient_c,
film_coefficient}`, `fans`, `flow` (`null` for still air, or `{max_steps, tolerance,
smagorinsky, buoyancy, coupling_iterations, lattice_velocity, max_cell_reynolds,
trt_lambda}`), `limits {max_cells, max_iterations, tolerance}`.

The summary reports, per body, the min/max/mean temperature and the cells it got; the air's
max and mean; the heat generated and the heat leaving (they agree when the solve converged);
solver iterations and residual; and for airflow the steps, convergence, time step, fastest
lattice speed (stability needs it well below 0.3) and the viscosity used.

Studies are saved with the design (`simulation` in the file); results are not.

## Workspace

Choose **Simulation** in the workspace switcher. The right panel edits the study (domain,
air and walls, a heat row per visible body, fans, airflow) and solves it off the window
thread with a progress bar; the viewport colours the bodies by surface temperature with a
legend, and in airflow mode draws velocity arrows through the air. A result stays until the
next solve and is flagged as stale once the design changes. **Export VTK…** writes the fields
for ParaView.

Headless pictures (`solidcraft-cli snapshot`, `ui.render`, MCP screenshots) draw the same
colour map and arrows while `sim.display` is on.

## Example

```sh
solidcraft-cli snapshot examples/simulation-enclosure.json --out enclosure.png --view iso
```

A 3 W chip and a 1 W regulator on an aluminium plate in free air: the chip's temperature
comes out at about 25 °C + 4 W / (8 W/m²K × plate area) ≈ 75 °C, as the film coefficient
predicts for a plate that spreads the heat well.

## Choosing settings

- **Cell size.** Start coarse (2–3 mm for a 100 mm box) and refine; a body thinner than a cell
  gets no cells and the solve says so. Cells cost memory: 4 million is the default cap.
- **Still or moving air.** Without a fan, keep `flow` off and set the mixing factor for the
  situation (sealed box ≈ 10, open frame ≈ 100). With fans, turn flow on; buoyancy off unless
  the case is fanless and you want the plume.
- **Film coefficient.** The single most important number for a part in air; it is not derived
  from the flow. Use 5–10 still, 20–50 with a fan over the part.
- **Validation.** `crates/sim/src/tests.rs` checks the solver against analytic conduction, the
  P/(hA) film limit, mass conservation of a fan-driven channel and a bluff body in a duct.
