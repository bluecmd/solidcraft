#!/bin/sh
# Runs sim-bench with the given binary and flags; the JSON goes to the undeclared outputs.
set -eu
bin=$1
shift
out=${TEST_UNDECLARED_OUTPUTS_DIR:-.}
"$bin" gpu --policy hardware
exec "$bin" sim-bench "$@" --json-out "$out/sim-bench.json"
