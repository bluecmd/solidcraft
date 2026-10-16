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
