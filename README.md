Original renders from the GPU path-tracing executor proposed in tangency-labs/solidcraft#1948.
All scenes are generated procedurally; no imported models, textures or photographs.

## studio/ (draft PR for studio lighting, clear coat and glass, follows #2067)

All 1920x1080 on an RTX 4090 with Vulkan through the production job host; no editing or denoising.
Original procedural scenes only. Each image shows the source snapshot it was rendered from, not
necessarily the final combined commit.

| File | What |
|---|---|
| `studio-stage1-256spp-vulkan.png` | Studio stage 1: area lights, white environment, cyclorama; 256 spp. |
| `coffee-opaque-glass-64spp-vulkan.png` | Original exploded coffee machine, glass bodies drawn opaque; 64 spp, 14.0 s. |
| `coffee-physical-glass-8spp-vulkan.png` | Same scene with physical closed glass (strict medium check passes); 8 spp, 61.7 s: too slow, the main open problem. |
