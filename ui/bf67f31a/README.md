# Desktop UI: progressive GPU path render (PR #2067, render-path @ bf67f31a)

- `ui-gpu-path.mp4` (51.7 s, 1280x720, H.264) and `ui-gpu-path.gif` (preview): the real
  `solidcraft` desktop app (release build of bf67f31a) under Xvfb on an NVIDIA GB10
  (DGX Spark, Vulkan), recorded with `ffmpeg -f x11grab`.
- `stage1-gpu-path-512spp.png`: the PNG written by the panel's Export PNG button in that run.

Scripted through the app's `--control` server: the scene (`examples/studio/stage1.json` from
render-studio, an original five-body recipe replayed with `engine.script`) and the camera
(`ui.camera`, the recipe's studio view). Everything else is pointer/keyboard input
(`ui.click`, `ui.key`, `ui.text`) on the visible widgets: workspace menu -> Render,
Perspective camera, Width 1280 / Height 720, Progressive rendering, 512 samples per pixel,
Check Render Request, Start Render, Export PNG (GTK file chooser via xdg-desktop-portal,
path typed with xdotool). The app has no in-app viewer for the finished frame, so the last
seconds show the exported PNG in `feh`. Default lighting: the UI exposes no lighting controls
on this branch. Render time for 1280x720 at 512 spp: about 11 s.
