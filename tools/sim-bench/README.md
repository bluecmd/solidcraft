# sim-bench image for arm64 GPU nodes

`solidcraft-cli sim-bench` (docs/simulation.md) packaged for NVIDIA arm64 nodes such as the DGX
Spark (GB10), plus the tools that found out what NVIDIA's Vulkan driver needs inside a container.

    tools/sim-bench/build.sh [TAG]      # needs cargo-zigbuild, zig and podman; no emulation

`job.yaml` is a template for one run on a time-shared GPU slice.

## What NVIDIA's Vulkan driver needs in a container

The simulation reaches the GPU through Vulkan (wgpu), not CUDA. With the NVIDIA container
toolkit, each of these was needed, found one at a time on a GB10 with driver 595.91.07:

1. `NVIDIA_DRIVER_CAPABILITIES` must include `graphics`, or the runtime mounts no Vulkan ICD.
2. `libX11.so.6` and `libXext.so.6` (with libxcb, libXau, libXdmcp, libbsd, libmd): the ICD,
   `libGLX_nvidia.so.0`, links them even though nothing displays.
3. `libEGL.so.1` (GLVND: `libegl1`, `libglvnd0`) and a vendor file
   `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` naming `libEGL_nvidia.so.0`. With no
   `DISPLAY`, the driver initialises through EGL. Without the dispatcher, its
   `vk_icdNegotiateLoaderICDInterfaceVersion` returns -3 and `vkCreateInstance` is NULL; the
   loader then reports "Found no drivers". The runtime did not mount a vendor file, so the image
   ships one.

Things that looked like causes and were not:
- glibc: `LD_DEBUG` prints `undefined symbol: __malloc_hook … (fatal)` and `ErrorF` for the
  driver's `dlsym` probes. They are harmless, and the libraries load with `RTLD_NOW`.
- The Vulkan loader's version: Ubuntu's 1.3.275 and DGX OS's 1.4.328.1 behave the same. The image
  carries 1.4.328.1, to match DGX OS.
- `/dev/nvidia-caps`: the driver never looks for it before creating an instance.

## Diagnostic tools in the image

The pods' RuntimeDefault seccomp profile blocks ptrace, so `strace` cannot run there. These need
no privileges:

- `/opt/trace/icdprobe [libGLX_nvidia.so.0]` loads a Vulkan ICD the way the loader does and
  prints each step: `dlopen` with `dlerror`, interface negotiation, entry points, instance,
  physical devices.
- `TRACE_FILE=/tmp/t.txt LD_PRELOAD=/opt/trace/libtrace.so <command>` logs every open, stat,
  access, readlink, ioctl, socket, connect, fopen, getenv and dlopen with its result and errno.
- `/opt/trace/libhooks.so` defines the glibc malloc hooks removed in 2.34 and Xorg's `ErrorF`, for
  testing whether a driver needs them (this one did not).

## Through Bazel instead

The image is no longer needed to run the benchmark on a Spark: `bazel test --config=remote
//tools/sim-bench:gb10` (or `:gb10-1_5`) cross-compiles the CLI for aarch64 and runs the test on
the cluster's GB10 BuildBuddy executor (pool `gb10`, `BUILD.bazel` here; the executor and its
image are `apps/buildbuddy` in argocd-apps), with the log and `sim-bench.json` in the BuildBuddy
invocation. The image and its tools remain for diagnosing the driver in a plain container.

Memory on the executor: the GB10's memory is unified, and a measurement of the 1.5 mm case
(2026-10-16) showed the pod's memory cgroup charging only ~2.1 GB while the node lost ~2.9 GB and
the process RSS reached ~2.8 GB — the GPU side escapes the cgroup, so the pod's 5 GiB limit does
not bound a GPU-heavy action and node headroom is the real limit (spark-2 bottomed at ~1.3 GB
free beside GLM). Keep 1.5 mm as the largest case there; 1 mm (27 M cells, ~14 GB) needs a node
with 20 GB free.

### On a Windows workstation (RTX 4090, pool `rtx4090`)

The same through a BuildBuddy executor started by hand on a Windows machine: the package
(`/scratch/bluecmd/bb-win/buildbuddy-executor-rtx4090.zip` on the bastion — the fork's Windows
executor, `config.yaml` with pool `rtx4090` and the bare runner, `start.cmd`, a README) registers
the machine while its window is open. `bazel test --config=remote //tools/sim-bench:rtx4090`
(or `:rtx4090-1_5`) cross-compiles the CLI for `x86_64-pc-windows-gnu` with the llvm-mingw
toolchain (`tools/bazel/llvm_mingw/`) and runs it there as a `native_test` (no shell needed on
that machine); the JSON is in `test.log`.
