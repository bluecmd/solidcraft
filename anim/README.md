# Exploded-view proof of concept

An original editable coffee-machine scene moves from a hand-placed assembled
pose to the `coffee-framing` exploded layout. Every frame replays the CAD recipe
and appends eased `solid.move` translations; the camera yaws by 12 degrees.
This is offline per-frame rendering, not a real-time animation feature.

- MP4: 90 frames, 1280×720, 30 FPS, 3 seconds, H.264 CRF 18.
- GIF: 640×360, 15 FPS, 3 seconds.
- Rendering: NVIDIA GB10 / Vulkan, 64 spp, depth 12, seed 1948, optional spatial
  denoising enabled. Glass bypasses that filter and retains visible grain.
- Roughly 10 seconds per frame including CAD replay: recovered frame 059 took
  3.14 seconds for CAD and 6.62 seconds for rendering and PNG encoding.

The prior session supplied 89 PNGs, but frame 059 had failed its glass-medium
check. It was rendered successfully with `0b9cfd0e` and inserted before encoding.
The previous frames predate that final policy; their exact renderer revision was
not recorded per frame. This video demonstrates the workflow and is not a
claim that all 90 frames were validated on the final PR head. `frames.json`
records the settings, provenance and hashes of all 90 input PNGs.

To reproduce from a `render-studio` source tree containing the coffee examples:

```sh
python3 make_frames.py /path/to/solidcraft/examples/studio frames
cargo build --release -p solidcraft-cli --example studio_evidence
mkdir -p out
for f in $(seq -f %03g 0 89); do
  SOLIDCRAFT_GPU=hardware WGPU_BACKEND=vulkan \
    target/release/examples/studio_evidence \
    frames/recipe-$f.json frames/render-$f.json out/frame-$f.png
done
ffmpeg -framerate 30 -i out/frame-%03d.png -frames:v 90 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart coffee-explode-720p30.mp4
ffmpeg -framerate 30 -i out/frame-%03d.png -frames:v 90 \
  -vf 'fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' \
  coffee-explode.gif
```
