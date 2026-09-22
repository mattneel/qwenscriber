const std = @import("std");

/// WASM feature set for the freestanding core build.
///
/// `simd128` is requested explicitly rather than inherited from a compiler
/// default, because the preprocessing and CPU-fallback kernels are written as
/// `@Vector` loops and must not silently degrade to scalar code.
const wasm_features = std.Target.wasm.featureSet(&.{.simd128});

pub fn build(b: *std.Build) void {
    const host_target = b.standardTargetOptions(.{});
    const host_optimize = b.standardOptimizeOption(.{});

    // The portable core, compiled for the host. Tests and the conversion
    // tooling consume this instance.
    const core = b.addModule("qwenscriber", .{
        .root_source_file = b.path("src/root.zig"),
        .target = host_target,
        .optimize = host_optimize,
    });

    // The same sources, compiled for the browser. ReleaseFast is not negotiable
    // for a shipped browser artifact: a Debug build of the core is only useful
    // for host-side tests.
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = wasm_features,
    });

    const core_wasm = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });

    const wasm_module = b.createModule(.{
        .root_source_file = b.path("src/wasm/exports.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "qwenscriber", .module = core_wasm },
        },
    });

    // `entry = .disabled` produces a library-shaped module: no `_start`, and
    // nothing runs until JavaScript calls an exported function.
    const wasm_lib = b.addExecutable(.{
        .name = "qwenscriber_core",
        .root_module = wasm_module,
    });
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;

    const install_wasm = b.addInstallArtifact(wasm_lib, .{});
    const wasm_step = b.step("wasm", "Build the wasm32-freestanding core module");
    wasm_step.dependOn(&install_wasm.step);

    // --- native tools ---

    // Checkpoint reading, name mapping, and conversion. Kept in its own module
    // because none of it is compiled for the freestanding target: it uses
    // `std.json`, an allocator, and the operating system.
    //
    // The revision travels with every measured run, so a number can be
    // attributed to a commit. A build that is not in a git checkout, or that is
    // built from a release tarball, says `unknown` rather than guessing;
    // `-Drevision` overrides both, which is what a release build does.
    const probed_revision = switch (b.runFallible(
        &.{ "git", "rev-parse", "--short", "HEAD" },
        .{},
    )) {
        .success => |text| std.mem.trim(u8, text, " \t\r\n"),
        else => @as([]const u8, "unknown"),
    };
    const revision = b.option(
        []const u8,
        "revision",
        "Revision recorded in run metrics (default: git's short HEAD)",
    ) orelse probed_revision;
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "revision", revision);

    const host = b.addModule("host", .{
        .root_source_file = b.path("src/host/root.zig"),
        .target = host_target,
        .optimize = host_optimize,
        .imports = &.{
            .{ .name = "qwenscriber", .module = core },
            .{ .name = "build_options", .module = build_options.createModule() },
        },
    });

    const selftest_exe = b.addExecutable(.{
        .name = "qwenscriber-selftest",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/selftest.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .imports = &.{
                .{ .name = "qwenscriber", .module = core },
            },
        }),
    });
    b.installArtifact(selftest_exe);

    const run_selftest = b.addRunArtifact(selftest_exe);
    const selftest_step = b.step("selftest", "Print the numerical core's self-test report");
    selftest_step.dependOn(&run_selftest.step);

    const transcribe_exe = b.addExecutable(.{
        .name = "qwenscriber-transcribe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/transcribe.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .imports = &.{
                .{ .name = "qwenscriber", .module = core },
                .{ .name = "host", .module = host },
            },
        }),
    });
    b.installArtifact(transcribe_exe);

    const convert_exe = b.addExecutable(.{
        .name = "qwenscriber-convert",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/convert.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .imports = &.{
                .{ .name = "qwenscriber", .module = core },
                .{ .name = "host", .module = host },
            },
        }),
    });
    b.installArtifact(convert_exe);

    const inspect_exe = b.addExecutable(.{
        .name = "qwenscriber-inspect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/inspect.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .imports = &.{
                .{ .name = "qwenscriber", .module = core },
                .{ .name = "host", .module = host },
            },
        }),
    });
    b.installArtifact(inspect_exe);

    // --- tests ---

    const core_tests = b.addTest(.{ .root_module = core });
    const run_core_tests = b.addRunArtifact(core_tests);

    // Reference comparisons live beside the fixtures rather than under `src`,
    // because they embed fixture files and so need a module root that contains
    // them.
    const reference_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/reference_check.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .imports = &.{
                .{ .name = "qwenscriber", .module = core },
            },
        }),
    });
    const run_reference_tests = b.addRunArtifact(reference_tests);

    // The command line tools own a few helpers (a WAV reader, a vocabulary
    // parser) whose tests belong in the same run.
    const cli_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/root.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .imports = &.{
                .{ .name = "qwenscriber", .module = core },
            },
        }),
    });
    const run_cli_tests = b.addRunArtifact(cli_tests);

    // Checkpoint reading, name mapping, and conversion own the larger share of
    // the tests; none of it needs the WASM target.
    const host_tests = b.addTest(.{ .root_module = host });
    const run_host_tests = b.addRunArtifact(host_tests);

    const test_step = b.step("test", "Run unit tests and reference comparisons");
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_reference_tests.step);
    test_step.dependOn(&run_cli_tests.step);
    test_step.dependOn(&run_host_tests.step);

    // `zig build test-wasm` proves the freestanding module instantiates and
    // computes correctly inside a real JavaScript engine.
    const wasm_selftest = b.addSystemCommand(&.{"node"});
    wasm_selftest.addFileArg(b.path("tools/wasm_selftest.mjs"));
    wasm_selftest.addFileArg(wasm_lib.getEmittedBin());
    wasm_selftest.step.dependOn(&install_wasm.step);

    const test_wasm_step = b.step("test-wasm", "Run the freestanding module's ABI self-test under Node");
    test_wasm_step.dependOn(&wasm_selftest.step);

    // --- gates ---

    const fmt_check = b.addSystemCommand(&.{
        "zig", "fmt", "--check", "build.zig", "src", "tests", "tools",
    });
    const fmt_step = b.step("fmt-check", "Check formatting");
    fmt_step.dependOn(&fmt_check.step);

    const check_step = b.step("check", "Format check, host tests, and the WASM ABI self-test");
    check_step.dependOn(&fmt_check.step);
    check_step.dependOn(&run_core_tests.step);
    check_step.dependOn(&run_reference_tests.step);
    check_step.dependOn(&run_cli_tests.step);
    check_step.dependOn(&run_host_tests.step);
    check_step.dependOn(&wasm_selftest.step);
}
