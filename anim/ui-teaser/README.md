# Render animation WIP teaser

Native SolidCraft Render workspace playback mockup, using original offline
ray-traced frames driven by native motion.study / motion.play commands.
This is a teaser concept, not an integrated animation-export feature or final UI.
The Export Animation button is disabled. Playback reads pre-rendered PNGs.

[Native in-UI layout preview (QHD, 14 seconds)](layout-preview-1440p.mp4).
This preview uses the earlier 1280x720, 64-spp frame set, faithfully labelled in
the panel; see ../frames.json and ../README.md for its recovered-frame provenance.
The final high-quality recording and aggregate frame manifest are being prepared.

Preview the new unmodified 1920x1080, 512-spp GB10 production quality frames:
[assembled 000](quality/frame-000.png), [middle 044](quality/frame-044.png),
[exploded 089](quality/frame-089.png). All three completed on renderer 9a7b5349.
Glass bypasses spatial denoising and can retain visible residual grain. The prototype source is
separate from production PR #2780 and preserves 0xlillith’s Render UI ownership;
remaining #1640 integration is coordinated with SychPL.

The viewer prototype is based on render-ui eb28a819. Apply prototype.patch there,
or use the three complete Rust files in prototype-source.tgz. Build the native
solidcraft app, then point SOLIDCRAFT_RENDER_TEASER_DIR at 90 frame-NNN.png files.
Set SOLIDCRAFT_RENDER_TEASER_SPP=512 for the final frame set. Without the directory
flag, the normal Render UI is used. The private prototype is native-only.

Native release build passed. Actual pointer checks confirm: pause holds an
identical image after 0.8s, scrubbing changes the image, and playback advances it.
The final source also has a format-only module-order correction after that build.
The prototype is evidence tooling; it is not proposed for production merging.

Both GB10 runners use the identical validated studio_evidence executable SHA-256:
d9f1ba3d19b886aca60a8448ad0932853a0ad230525450293dfb677c9f3f0166.
The production renderer source is 9a7b53498269866567ae2cb2b126609fd04f4c8e.
The three Windows quality frames used the original verified input package
3b194c7f; the batch uses 91464fef, which only omits repeated read-only mass-property
queries. Render settings and modelling/motion commands are unchanged.

cross-gpu-comparison.json compares decoded RGB pixels of overlapping frame 044
from NVIDIA GB10 and RTX 4090, both Vulkan, 1920x1080, 512 spp. No alignment,
normalization or image editing was applied. Full-image per-channel RMS differences
are 0.0124, 0.0127, 0.0125 in 8-bit display units; maximum differences are 2, 2, 3.
This overlap supports combining these machines’ frames; it is not a general
cross-platform rendering equivalence proof.

All scene geometry, lighting and captured UI are original SolidCraft project
assets under the project licence. No commercial CAD media or reference images.
