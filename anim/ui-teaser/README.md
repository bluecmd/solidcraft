# Render animation WIP teaser

[Watch the native Render UI teaser (2560×1440, 30 fps, 14 seconds)](render-animation-wip-1440p.mp4).
[1080p sharing copy](render-animation-wip-1080p.mp4) · [Poster](render-animation-wip-poster.png).

This is a recording of the actual native SolidCraft window playing 90 original
1920×1080, 512-sample ray-traced frames. The UI is drawn by SolidCraft; it is not
composited over a video. The recording shows two playbacks of the three-second
study with holds between them. It has no audio.

The frame player is an opt-in **work in progress mockup**, not a finished
animation-export integration or final UI proposal. It reads offline PNGs.
The Export Animation button is disabled. The model animation uses the existing
native motion.study / motion.play workflow; see [the editable study and verified
reproduction inputs](../study/README.md). No second animation framework.

All 90 frames use renderer `9a7b53498269866567ae2cb2b126609fd04f4c8e`,
1920×1080, 512 spp, depth 12, seed 1948 and the renderer's spatial denoiser.
The [frame manifest](frames-1080p512.json) identifies every PNG hash, runner,
GPU/backend/driver, input-package revision, render-parameter hash and timings.
Frames were rendered on RTX 4090 and two GB10 runners, all hardware Vulkan.
These per-frame timings are from shared machines, not isolated benchmarks.
Every final PNG was fully decoded and its dimensions checked. Windows hashes
and byte counts were checked against the delegate's manifest, and every frame's
render parameters against the verified public input package.

The glass model is unchanged. Glass bypasses spatial denoising and retains some
grain. Exact tolerated glass-failure counts are unavailable because the example
does not initialize the log logger; successful jobs do not establish zero failures.
No external denoising, image retouching, optical flow or frame interpolation was
applied. The shareable MP4s use H.264, YUV 4:2:0, CRF 15 and fast-start metadata;
the capture master is lossless RGB H.264.

## Native prototype source

The final viewer uses SolidCraft 0.15.0 and the combined review revision
[`60f788b7994688b94545999292395892d6c17c5a`](https://github.com/bluecmd/solidcraft/commit/60f788b7994688b94545999292395892d6c17c5a).
This retains 0xlillith's current #1640 Render workspace, the reviewed GPU job seam
and the three #2780 UI commits. It does not replace her Render UI with an older
version. Remaining integration is coordinated with SychPL.

Apply `prototype-on-1640.patch` to that base, or use the three complete Rust files
in `prototype-on-1640-source.tgz`. Build with `cargo build --profile ci -p solidcraft`.
Set `SOLIDCRAFT_RENDER_TEASER_DIR` to a directory containing all 90 `frame-NNN.png`
files and `SOLIDCRAFT_RENDER_TEASER_SPP=512`. Without the directory flag the normal
Render workspace is used. This separate, native-only evidence prototype is not
included in production PR #2780. The Render owner remains responsible for the
final UI design. See `capture.json` for exact source-file and binary hashes.

The native viewer was built and visually inspected. Actual pointer interactions
verify pause, scrubbing and advancing playback. The recording uses the real app
at 2560×1440 with UI scale 1.2 and a private settings directory.

## Earlier preview and quality comparison

[Earlier layout preview](layout-preview-1440p.mp4) uses the recovered 1280×720,
64-spp set and labels it accordingly. Its original 0.14 prototype source is
`prototype.patch` / `prototype-source.tgz`, based on `eb28a819`; these are not the
source used for the final 0.15 recording. The older final source has a format-only
module-order correction after its build. See ../frames.json for its frame provenance.

Unmodified high-quality GB10 stills:
[assembled 000](quality/frame-000.png), [middle 044](quality/frame-044.png),
[exploded 089](quality/frame-089.png).

`cross-gpu-comparison.json` compares decoded RGB pixels of overlapping frame 044
from GB10 and RTX 4090 with the final render settings. Full-image per-channel RMS
differences are 0.0124, 0.0127, 0.0125 in 8-bit display units; maximum differences
are 2, 2, 3. No alignment, normalization or image editing was applied. This overlap
supports combining these machines' frames; it is not a general cross-platform
rendering equivalence proof.

The three Windows quality frames used package `3b194c7f`; the batch used
`91464fef`, which only omits 54 repeated read-only mass-property queries per frame.
Render parameters and modelling/motion commands are unchanged.

All geometry, lighting, UI and media are original SolidCraft project assets under
the project licence. No commercial CAD media or reference images were used.
