"""Tests that are a prebuilt binary run on a GPU executor.

Bazel chooses a test action's execution platform through the rule's `test` exec group, and a
target's `exec_compatible_with` does not reach it; declaring the group with the executor's
constraints does. That makes the executor's platform the test's execution platform — so Bazel
wraps the test for that platform's OS (the Windows test wrapper for the Windows workstation),
and the platform's exec_properties (pool, OS, arch) route the action to the BuildBuddy pool.
One rule per executor platform, since an exec group's constraints are fixed per rule.
"""

def _gpu_test_impl(ctx):
    info = ctx.attr.binary[DefaultInfo]
    src = info.files_to_run.executable
    if src == None:
        files = info.files.to_list()
        if len(files) != 1:
            fail("binary must be an executable target or exactly one file, not %d files" % len(files))
        src = files[0]
    out = ctx.actions.declare_file(ctx.label.name + (".exe" if src.extension == "exe" else ""))
    ctx.actions.symlink(output = out, target_file = src, is_executable = True)
    return [
        DefaultInfo(executable = out, runfiles = ctx.runfiles(files = [out])),
        RunEnvironmentInfo(environment = ctx.attr.env),
    ]

def _gpu_test_rule(constraints):
    return rule(
        implementation = _gpu_test_impl,
        test = True,
        attrs = {
            "binary": attr.label(mandatory = True, doc = "The executable the test runs (an executable target, or one prebuilt file), built for the executor's platform."),
            "env": attr.string_dict(doc = "Environment of the test."),
        },
        exec_groups = {
            "test": exec_group(exec_compatible_with = constraints),
        },
    )

# A test on the DGX Spark executor (//platforms:gb10_executor).
gb10_test = _gpu_test_rule([
    "@platforms//os:linux",
    "@platforms//cpu:aarch64",
    "//platforms:gb10",
])

# A test on the Windows workstation with the RTX 4090 (//platforms:rtx4090_executor).
rtx4090_test = _gpu_test_rule([
    "@platforms//os:windows",
    "@platforms//cpu:x86_64",
    "//platforms:rtx4090",
])

# A run of a prebuilt binary on a GPU executor as a build action: the binary is executed as is
# (no test wrapper, no shell — the Windows test wrapper only ships with a Windows Bazel), with
# `args` and then `--json-out`/`--logfile`-style `out_flag` naming the output file, which is the
# action's result; stdout and stderr are kept by BuildBuddy and shown by Bazel. Never cached:
# a benchmark is meant to run every time.
def _gpu_run_impl(ctx):
    info = ctx.attr.binary[DefaultInfo]
    src = info.files_to_run.executable
    if src == None:
        files = info.files.to_list()
        if len(files) != 1:
            fail("binary must be an executable target or exactly one file, not %d files" % len(files))
        src = files[0]
    out = ctx.actions.declare_file(ctx.attr.out)
    args = list(ctx.attr.args) + [ctx.attr.out_flag, out.path]
    ctx.actions.run(
        executable = src,
        arguments = args,
        inputs = [src],
        outputs = [out],
        env = ctx.attr.env,
        exec_group = "gpu",
        execution_requirements = {"no-cache": "1", "no-remote-cache": "1"},
        mnemonic = "GpuRun",
        progress_message = "Running %s on the GPU executor" % ctx.label.name,
    )
    return [DefaultInfo(files = depset([out]))]

def _gpu_run_rule(constraints):
    return rule(
        implementation = _gpu_run_impl,
        attrs = {
            "binary": attr.label(mandatory = True, doc = "The executable to run (an executable target, or one prebuilt file), built for the executor's platform."),
            "args": attr.string_list(doc = "Arguments before the output flag."),
            "out_flag": attr.string(default = "--json-out", doc = "The flag that takes the output path."),
            "out": attr.string(mandatory = True, doc = "The output file's name."),
            "env": attr.string_dict(doc = "Environment of the run."),
        },
        exec_groups = {
            "gpu": exec_group(exec_compatible_with = constraints),
        },
    )

gb10_run = _gpu_run_rule([
    "@platforms//os:linux",
    "@platforms//cpu:aarch64",
    "//platforms:gb10",
])

rtx4090_run = _gpu_run_rule([
    "@platforms//os:windows",
    "@platforms//cpu:x86_64",
    "//platforms:rtx4090",
])
