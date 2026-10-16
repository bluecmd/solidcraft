#!/bin/sh
# The linker for aarch64 targets: zig's c++ from hermetic_cc_toolchain, minus one flag. rustc
# always passes -Wl,--fix-cortex-a53-843419 for aarch64-unknown-linux-gnu (the Cortex-A53 erratum
# fixup) and zig's driver rejects it as unsupported, so it is dropped here; lld applies the fixup
# on its own. cargo-zigbuild does the same. Runs with the execroot as working directory.
set -eu
real=external/hermetic_cc_toolchain++toolchains+zig_config/tools/aarch64-linux-gnu.2.28/c++
n=$#
i=0
while [ "$i" -lt "$n" ]; do
    a=$1
    shift
    i=$((i + 1))
    [ "$a" = "-Wl,--fix-cortex-a53-843419" ] || set -- "$@" "$a"
done
exec "$real" "$@"
