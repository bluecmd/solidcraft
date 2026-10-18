# Native motion-study animation inputs

Original SolidCraft coffee scene, authored as 27 as-built slider joints connected
to grounded root. `motion.study` keys all 90 frames; each frame recipe ends in
`motion.play` and `joint.list`. Initial `solid.move` calls only author the assembled
pose before creating the joints. The camera yaws 12 degrees independently.

`frames.tgz` contains recipe-NNN.json, bound render-NNN.json, study.json,
plan.json, poses.json, verification.json and coffee-study.solidcraft.
Render settings: 1920×1080, 512 spp, depth 12, seed 1948, spatial denoise,
original physical glass and studio illumination. Rendering is offline.

Verification in a fresh private native SolidCraft session passed 2,430 joint-pose
checks, before/after joint bounding boxes and centers of mass, and world-space
bounds, volume and mass at steps 0, 44 and 89. No conflicts or floating parts.
Maximum pose error: 1.14e-13 mm. Maximum bounds error: 1.71e-13 mm.
The verification app was built from render-ui eb28a819 (archive build reports
unknown commit); it verifies motion, not the final renderer. Renderer revision
and image timings belong in the eventual rendered-frame manifest.

Use `make_study_frames.py /path/to/solidcraft/examples/studio OUTPUT` to regenerate,
then run verify_study.py against a NEW private --control session with the same
output directory accessible to the app. The verifier replaces that session's
contents. It binds material names to actual render.scene ordinals after component
conversion; do not assume the old pre-component body order.

To render the verified package using the studio renderer:

```sh
tar -xzf frames.tgz -C inputs
cargo build --release -p solidcraft-cli --example studio_evidence
SOLIDCRAFT_GPU=hardware WGPU_BACKEND=vulkan target/release/examples/studio_evidence \
  inputs/recipe-044.json inputs/render-044.json frame-044.png
```

These are original procedural project assets under the SolidCraft project licence.
The UI playback teaser is a separate opt-in mockup, not an integrated animation
export feature or a final Render UI design.

Per-frame recipes omit the 54 read-only occurrence.properties measurements that
remain in study.json for one-time verification; modelling and motion commands
are identical. This avoids repeating mass-property integration on every render.
