#!/bin/sh
# Build the sim-bench image for arm64 GPU nodes (DGX Spark / GB10) on any Linux host, without
# emulation: the CLI is cross-compiled with cargo-zigbuild, the tracing tools with zig, and the
# image's arm64 stage only copies files. Run from the repository root.
#   tools/sim-bench/build.sh [TAG]      (TAG defaults to the current commit)
set -eu
tag=${1:-$(git rev-parse --short=8 HEAD)}
here=tools/sim-bench
rustup target add aarch64-unknown-linux-gnu >/dev/null
CARGO_PROFILE_RELEASE_STRIP=true CARGO_PROFILE_RELEASE_DEBUG=0 \
    cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.35 -p solidcraft-cli
cp "${CARGO_TARGET_DIR:-target}/aarch64-unknown-linux-gnu/release/solidcraft-cli" "$here/solidcraft-cli"
zig cc -target aarch64-linux-gnu.2.35 -shared -fPIC -O2 -Wno-pointer-bool-conversion -o "$here/trace/libtrace.so" "$here/trace/trace.c"
zig cc -target aarch64-linux-gnu.2.35 -shared -fPIC -O2 -o "$here/trace/libhooks.so" "$here/trace/hooks.c"
zig cc -target aarch64-linux-gnu.2.35 -O2 -o "$here/trace/icdprobe" "$here/trace/icdprobe.c" -ldl
podman build --platform linux/arm64 -t "localhost/solidcraft-sim-bench:$tag" -f "$here/Containerfile" "$here"
podman save --format oci-archive -o "solidcraft-sim-bench-$tag-arm64.tar" "localhost/solidcraft-sim-bench:$tag"
rm -f "$here/solidcraft-cli" "$here/trace/libtrace.so" "$here/trace/libhooks.so" "$here/trace/icdprobe"
echo "built localhost/solidcraft-sim-bench:$tag and solidcraft-sim-bench-$tag-arm64.tar"
