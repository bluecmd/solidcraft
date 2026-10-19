# Hardware ray tracing prototype and benchmark (not for merging)

This branch is evidence for the `unsafe` decision on hardware ray tracing. It is not meant to be merged: it allows `unsafe` in the workspace and in `solidcraft-gpu`.

## The prototype renderer

`SC_HWRT=1` switches the gpu-path executor to wgpu hardware ray queries (`crates/gpu/src/path/mod.rs`). It works like this:

- It requests `EXPERIMENTAL_RAY_QUERY` with the `unsafe` `ExperimentalFeatures::enabled()` token.
- It builds one BLAS and TLAS from the uploaded triangles.
- It replaces `trace_ordered_within`, the f32 BVH traversal, with a ray query.

It is valid for **opaque scenes only**. Glass continuations still use the precise software path, and their ordering rules are not mirrored.

## Dumping a scene

`SC_DUMP=<dir>` makes a gpu-path job write the buffers it uploads (`geometry.bin`, `nodes.bin`, `params.bin`).

## The benchmark

`tools/hwrt-bench` is a standalone crate. It traces the same rays against the dumped triangles twice: once with the renderer's own WGSL traversal over its own BVH, and once with hardware ray queries.

```sh
SC_DUMP=/tmp/dump SOLIDCRAFT_GPU=hardware cargo run --release -p solidcraft-cli --example studio_evidence -- \
  examples/studio/coffee-exploded-sweep.json coffee-opaque-1spp-1080p.json /tmp/out.png
cat crates/gpu/src/path/{path,shadows,textures}.wgsl > /tmp/full.wgsl
cargo run --release --manifest-path tools/hwrt-bench/Cargo.toml -- /tmp/dump /tmp/full.wgsl 20 12
```

The fourth argument is the number of path vertices.
