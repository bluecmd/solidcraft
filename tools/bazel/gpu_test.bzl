"""A test that is a prebuilt binary run on a GPU executor.

Bazel chooses a test action's execution platform through the rule's `test` exec group, and a
target's `exec_compatible_with` does not reach it; declaring the group here with the constraint
does. That makes the test's execution platform the executor's (so Bazel wraps the test for that
platform's OS: the Windows test wrapper for a Windows executor), and its exec_properties route
the action to the BuildBuddy pool.
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

gpu_test = rule(
    implementation = _gpu_test_impl,
    test = True,
    attrs = {
        "binary": attr.label(mandatory = True, doc = "The executable the test runs (an executable target, or one prebuilt file), built for the executor's platform."),
        "env": attr.string_dict(doc = "Environment of the test."),
    },
    exec_groups = {
        "test": exec_group(exec_compatible_with = ["//platforms:gpu_executor"]),
    },
)
