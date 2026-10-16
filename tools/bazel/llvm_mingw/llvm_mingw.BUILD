# BUILD file for the llvm-mingw tarball (see MODULE.bazel). Defines the C++
# toolchain that rules_rust uses as the linker for x86_64-pc-windows-gnu.
load("@//tools/bazel/llvm_mingw:cc_toolchain_config.bzl", "llvm_mingw_cc_toolchain_config")
load("@rules_cc//cc:defs.bzl", "cc_toolchain")

package(default_visibility = ["//visibility:public"])

# Only the x86_64 target tree is needed; the tarball also ships i686, armv7
# and aarch64 sysroots. lib/ holds the shared libraries clang itself links
# against and the compiler-rt builtins.
filegroup(
    name = "all_files",
    srcs = glob([
        "bin/**",
        "lib/**",
        "x86_64-w64-mingw32/**",
    ]),
)

filegroup(name = "empty")

llvm_mingw_cc_toolchain_config(name = "cc_toolchain_config")

cc_toolchain(
    name = "windows_x86_64_cc",
    all_files = ":all_files",
    ar_files = ":all_files",
    as_files = ":all_files",
    compiler_files = ":all_files",
    dwp_files = ":empty",
    linker_files = ":all_files",
    objcopy_files = ":all_files",
    strip_files = ":all_files",
    supports_param_files = 1,
    toolchain_config = ":cc_toolchain_config",
    toolchain_identifier = "llvm-mingw-x86_64",
)

toolchain(
    name = "windows_x86_64_cc_toolchain",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":windows_x86_64_cc",
    toolchain_type = "@rules_cc//cc:toolchain_type",
)
