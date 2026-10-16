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
finite-volume stencil solved with conjugate gradients (BiCGSTAB when air moves),
preconditioned by geometric multigrid (`crates/sim/src/multigrid.rs`): cells aggregate
2 × 2 × 2, the coarse operator is Galerkin with piecewise-constant prolongation (every level
stays a 7-point M-matrix, whatever the conductivity jumps), red-black Gauss-Seidel smooths, and
the cycle is symmetric so CG can use it. A Jacobi preconditioner was not enough: the 2 mm
benchmark system (3.4 M cells) took thousands of BiCGSTAB iterations on the CPU and ran away on
the GB10; with multigrid it converges in about a hundred.

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

**Coolant** is a second fluid zone: model the water as its own body (the inside of the tubes
and blocks, as the CAD shape of the coolant) and tag it as a fluid. Its cells get water's
conductivity, heat capacity, viscosity, expansion and a water-side film coefficient against
solids (thousands of W/m²K); a fan whose region lies inside the water acts as a pump. Each
fluid zone is its own lattice with its own time step and viscosity floor; a coolant never meets
the domain walls and bounces off every cell that is not its own zone. The thermal grid couples
the zones through the solids they touch. Resolved water channels need cells a few times
smaller than the channel: tubes of 8 mm at 2 mm cells are fine, the micro-channels of a cold
plate are not (a cold plate is modelled as a water-filled cavity until grid refinement lands).

**Airflow** (and every fluid zone) is a D3Q19 lattice-Boltzmann solver on the zone's cells, run
until the velocity field stops changing (a steady or time-averaged flow).

- Fans are axis-aligned regions of air driven at a velocity vector; put one just inside an
  opening, or across a duct.
- Solid cells and closed walls bounce back; open walls are pressure outlets that also let air
  in.
- Regularized BGK collision (Latt & Chopard 2006) by default: the non-equilibrium part is
  projected on its second-order moment before relaxing, which removes the modes that grow where
  jets and recirculation push the local cell Reynolds number far above the design value (a fan
  wall without a baffle drives air back through its gaps at several times the fan speed).
  Two-relaxation-time collision remains available (`trt_lambda` > 0; 1/12 recommended). Both use
  a Smagorinsky eddy viscosity and a viscosity floor that keeps the cell Reynolds number
  (u·dx/ν) at or below `max_cell_reynolds` (10 by default).
- A sponge layer beside every open wall (`sponge_cells`, 6 by default; `sponge_viscosity`): the
  viscosity ramps up towards the wall so vortices are damped before they reach the boundary,
  where air may flow in and out at once. On a 1–2 mm grid at fan speeds this raises the viscosity well above air's: the
  flow you see is the bulk routing of the air with turbulent mixing standing in for the detail
  the grid cannot carry. A finer grid or a higher limit is more faithful and less stable.
- Buoyancy (Boussinesq, gravity along −Z) is on by default; with it the thermal and flow
  solves alternate `coupling_iterations` times. Turn it off for purely fan-driven cases.

## On the GPU

Simulation requires a GPU. The application and the CLI open a device of their own for it
(`solidcraft-sim-gpu`, a wgpu compute backend; the `SOLIDCRAFT_GPU` policy chooses the adapter)
on the first solve, and `sim.solve` refuses without one. The lattice-Boltzmann step runs
entirely on the device: one population buffer with in-place AA-pattern streaming (bounce-back
falls out of the slot swap), the two-relaxation-time collision with the Smagorinsky eddy
viscosity and Guo forcing, fans as equilibrium cells, open walls as ambient equilibrium; a
reduction kernel reports the change of the velocity field every 50 steps. The temperature
system (7-point stencil) is solved on the device by conjugate gradients or BiCGSTAB, resident:
vectors, the multigrid hierarchy and the scalar recurrences (ρ, α, ω) all stay on the GPU, a
batch of 25 iterations is one submission, and the host reads back one residual per batch. The
solver keeps the best iterate, restarts from it after a breakdown and never returns a diverged
answer. On GPUs with doubles (`SHADER_F64`: NVIDIA, AMD, Intel, lavapipe) it runs in f64 like
the CPU reference. Without doubles (Apple) it runs a symmetrically scaled f32 solve with f64
refinement, and when that cannot carry a stiff system (metal beside air spans ten orders of
conductance) the f64 CPU reference finishes the temperature solve and the report names it
(`cpu-cg (GPU f32 insufficient)`); the airflow stays on the GPU. The summary reports the
backend and the throughput (`mlups`, million lattice updates per second).

The CPU implementation remains as the reference the GPU kernels are checked against
(`crates/sim-gpu/src/tests.rs`: the fields agree to better than 0.1% on a fan-driven channel,
temperatures to 0.05 K).

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
| `sim.display {mode?, seeds?, setup?}` | `off`, `temperature` (colour map on the bodies), `velocity` (arrows too) or `streamlines` (lines through the air too); `seeds` streamline seeds (1–512, default 96); `setup` draws the setup glyphs in pictures (default on). |
| `sim.slice {axis?, position?, field?, off?}` | A slice plane across the domain (one per axis); `off: true` or `axis: null` removes slices; `{}` lists them. |
| `sim.probe.pin {point, body?}` / `sim.probe.delete {index \| all}` / `sim.probe.list` | Probes pinned to points, labelled in the view with their reading. |

Study fields (`sim.study.set` → `set`): `cell_size_mm` (0 = about 60 cells along the
domain's longest side), `domain {min, max}` or `padding_mm`, `bodies` (as `sim.body.set`),
`walls` (six of `open`/`closed`: −x, +x, −y, +y, −z, +z), `air {conductivity, density,
specific_heat, kinematic_viscosity, mixing_factor, thermal_expansion}`, `thermal {ambient_c,
film_coefficient}`, `fans`, `flow` (`null` for still air, or `{max_steps, tolerance,
smagorinsky, buoyancy, coupling_iterations, lattice_velocity, max_cell_reynolds,
trt_lambda}`), `limits {max_cells, max_iterations, tolerance}`.

The summary reports, per body, the min/max/mean temperature and the cells it got; the air's
max and mean; the heat generated and the heat leaving (they agree when the solve converged);
solver iterations, residual and method (`gpu-cg64`, …); and for airflow the steps, convergence,
time step, fastest lattice speed (stability needs it well below 0.3), the viscosity used, the
backend and its throughput.

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

## Exploring a result

Everything below is view state: kept per open design while it is open, never saved and never
an undo step. The viewport and headless pictures (`solidcraft-cli snapshot`, `ui.render`) draw
it from the same geometry (`crates/engine/src/cmd/simulation_view.rs`).

**Slice planes.** `sim.slice {axis: "z", position: 12, field: "temperature"}` cuts the domain
at z = 12 mm and colours every cell the plane passes through, air and solids alike, so the
inside of an enclosure shows: one flat quad per cell in the cell's own value (no smoothing,
so a solid's edge stays where the voxels put it). Up to three slices, one per axis; setting an
axis again moves or changes that slice. Without `position` a new slice goes through the middle
of the solved domain. `field: "speed"` colours the air by speed from 0 to the fastest cell,
with solid cells grey; it needs a solve with airflow. A plane finer than 65 536 cells is
sampled every few cells (each quad then stands for a block of cells, coloured by its middle
cell). In the viewport slices are translucent and unlit; in headless pictures they are opaque.
The panel's **Slice planes** rows switch each axis on and off, drag its position and pick its
field.

*Colour scale.* Bodies alone use the bodies' own range (coolest to hottest surface). While a
temperature slice is shown, bodies and slices share one scale widened to the whole grid's range,
which takes in the air (usually down to the ambient): the legend then reads the same for both.
The legend names each slice and, for a speed slice or streamlines, adds an air-speed bar.

**Streamlines.** `sim.display {mode: "streamlines"}` traces lines through the solved air:
seeded across each fan on the mid-plane of its region (traced both up- and downstream, so a
line shows where the fan draws air from and where it sends it) and on the air cells of open
walls where air flows in (traced downstream). Integration is fourth-order Runge–Kutta in steps
of half a cell along the trilinearly interpolated velocity; a line stops at a solid, at the
domain wall, where the air is slower than 1 % of the fastest cell, or at a step cap. Lines are
coloured by local speed. `seeds` (at most 512) bounds the count; all lines together hold at
most 200 000 points. The sampling helpers live in `crates/sim/src/sample.rs`
(`temperature_smooth`, `velocity_smooth`, `streamline`, `inflow_seeds`, `fan_seeds`).

**Probes.** In the Simulation workspace, hovering a solved body shows its surface temperature
under the pointer ("Chip · 63.4 °C", the value the colour map shows there); where no body is
under the pointer but a slice is, the reading is the slice cell's temperature and air speed.
A click pins a probe (`sim.probe.pin`): a dot with its label in the view, and a row with a
delete button in the panel's **Probes** list. In this workspace a click probes instead of
selecting.

**Setup glyphs.** While the Simulation workspace is open, the study being edited is drawn in
the model view: each fan's region as a translucent box with an arrow along its air and its name
and speed, a "⚡ 3 W" badge on each heat source, a "❄ 20 °C" badge on each body held at a fixed
temperature, and the domain box as a dashed outline. Clicking a fan's label opens it in the
panel's Fans list. Headless pictures draw the lines (no labels) of the solved study while
`setup` is on.

**Results table.** The panel lists the bodies hottest first with a swatch of each body's
maximum on the colour scale. Clicking a row frames the body and pins a probe at its hottest cell.

## Example

```sh
solidcraft-cli snapshot examples/simulation-enclosure.json --out enclosure.png --view iso
```

A 3 W chip and a 1 W regulator on an aluminium plate in free air: the chip's temperature
comes out at about 25 °C + 4 W / (8 W/m²K × plate area) ≈ 75 °C, as the film coefficient
predicts for a plate that spreads the heat well.

```sh
solidcraft-cli snapshot examples/simulation-airflow.json --out airflow.png --view iso
```

A small server-like chassis with its sides, floor and lid closed and its front and back open:
a fan behind the front opening blows 3 m/s along X over a 20 W CPU with a finned heat sink,
two memory modules and a power supply. The picture shows streamlines and a temperature slice
at z = 12 mm.

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

## Benchmark

`solidcraft-cli sim-bench [--cell MM] [--steps N] [--power W] [--cpu] [--json-out FILE]`
(and the solver-development flags below) times a
2U server study on the GPU backend: a 440 × 700 × 88 mm chassis with sheet-metal sides, open front
and back, a fan wall of six fans in a bulkhead, two CPUs of `--power` W (200 by default) under
heat sinks of 20 fins 1.6 mm thick at 4.5 mm pitch, eight DIMMs and a PSU. It reports the cells,
the lattice throughput (million lattice updates per second), the time of each phase, the heat
balance and the hottest parts. The fins are resolved from about 1 mm cells (27 million cells,
about 3 GB of GPU memory). The stability settings (`--max-cell-reynolds`, `--lattice-velocity`,
`--smagorinsky`, `--trt-lambda`, `--sponge-cells`, `--sponge-viscosity`, `--fan-speed`) are
exposed for tuning.

For solver work: `--cpu-flow` runs the airflow on the CPU reference and only the temperature
solve on the GPU; `--dump-thermal FILE` writes the assembled temperature system
(`ThermalSystem::to_bytes`) before solving it, and the ignored tests `dumped_system_on_the_cpu`
(`crates/sim`) and `dumped_system_on_the_gpu` (`crates/sim-gpu`) solve such a file named in
`SOLIDCRAFT_THERMAL_DUMP` without the airflow solve. `SOLIDCRAFT_SIM_DEBUG=1` traces the
lattice every 50 steps and the solver residual every 25 iterations.
