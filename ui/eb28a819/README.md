Actual SolidCraft desktop capture for tangency-labs/solidcraft#2780, built from
`render-ui` at `eb28a819`. The completed image appears below Export PNG in the
existing Render Image panel.

The original stage-1 scene and camera were set through the control server.
Perspective, dimensions, Progressive rendering, Samples per pixel, Check and
Start used synthetic pointer/keyboard events on the visible controls. The image
was rendered by the production `gpu-path` host at 1280×720, 512 spp, on NVIDIA
GB10 / Vulkan. The viewport uses its own preview lighting; the panel shows the
completed path-traced image. No studio lighting or glass from #2191 is implied.

The screenshot is unedited. The build used a source archive without `.git`, so
the binary reports an unknown commit. `capture.json` records the source commit
and archive hash. This capture is of the standalone UI branch; combined CPU/UI
validation against #1640 is separate.
