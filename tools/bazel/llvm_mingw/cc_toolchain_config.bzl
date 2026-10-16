"""Minimal C++ toolchain config for the llvm-mingw cross linker.

rules_rust only takes the linker (and its link flags) from the C++ toolchain
resolved for the target platform, so nothing beyond tool paths is configured.
The x86_64-w64-mingw32-* wrappers are shell scripts that add `-target
x86_64-w64-mingw32` and call the bundled clang, which finds its own sysroot.
"""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "feature", "tool_path")

_PREFIX = "bin/x86_64-w64-mingw32-"

_TOOLS = {
    "ar": _PREFIX + "ar",
    "cpp": _PREFIX + "cpp",
    "gcc": _PREFIX + "gcc",
    "gcov": "bin/llvm-cov",
    "ld": _PREFIX + "ld",
    "nm": _PREFIX + "nm",
    "objcopy": _PREFIX + "objcopy",
    "objdump": _PREFIX + "objdump",
    "strip": _PREFIX + "strip",
}

def _impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        abi_libc_version = "mingw",
        abi_version = "mingw",
        compiler = "clang",
        features = [
            feature(name = "supports_pic", enabled = True),
            feature(name = "supports_dynamic_linker", enabled = True),
        ],
        host_system_name = "x86_64-unknown-linux-gnu",
        target_cpu = "x86_64",
        target_libc = "mingw",
        target_system_name = "x86_64-w64-mingw32",
        tool_paths = [tool_path(name = k, path = v) for k, v in _TOOLS.items()],
        toolchain_identifier = "llvm-mingw-x86_64",
    )

llvm_mingw_cc_toolchain_config = rule(
    implementation = _impl,
    provides = [CcToolchainConfigInfo],
)
