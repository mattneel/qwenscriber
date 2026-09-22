# Zig gotchas

This repository tracks Zig master, whose standard library and build API move. When the compiler
disagrees with what you assumed, **the compiler error is the answer**: fix the call site and re-run
`zig build test`. Do not stand up a probe project to go and discover the API — read the installed
source instead (`zig env` gives `std_dir` / `lib_dir`).

Consult this file before writing code that touches these APIs.

| Symptom | Correct pattern |
| --- | --- |
| `std.fs` does not exist | `std.Io.Dir` / `std.Io.File`, with an explicit `io` value threaded through calls |
| `std.ArrayList(T)` has no `.init(allocator)` | It is unmanaged: `.empty`, then `list.append(gpa, item)`, `list.deinit(gpa)` |
| `declaration shadows declaration` | A parameter named `rows` collides with a method `rows()`. Rename one of them |
| `no field named 'fields' in struct 'Type.Enum'` | `@typeInfo(T).@"enum"` exposes `field_names` / `field_values`; for user-facing parsing prefer an explicit table |
| `std.mem.addWithOverflow` does not exist | Use the builtin `@addWithOverflow` |
| `error` used as an identifier | It is a keyword: never `var error: f32` |
| `for (0..n) \|i\|` yields `usize` even when `n` is `u32` | Pick one index type per file — `usize` for byte offsets — instead of casting at every call site |
| `import of file outside module path` | `@import` may not escape the module directory. Declare the file as a module in `build.zig` and import it by name |
| `TODO: no build.zig.zon file` | `anyzig` resolves the toolchain from `build.zig.zon`; run `zig build` where that file is |
| Error sets do not unify | `pub const Error = other.Error \|\| error{...}` when wrapping another module's failures |
| `no field or member function named 'allocRemaining' in 'Io.File.Reader'` | A buffered reader's `allocRemaining` lives on the generic reader: `reader.interface.allocRemaining(gpa, limit)` |
| A read succeeds and returns 0 bytes from a file that has content | `File.reader` memoizes the size the handle reports, and `/proc/cpuinfo` reports 0. Use `File.readerStreaming(io, &buffer)` and read to end of file instead |
| `runAllowFail is deprecated` | `b.runFallible(argv, .{})` returns a union — `.success` carries stdout, `.bad_exit_code` / `.crashed` / `.spawn_failed` are the rest — so a missing `git` does not fail the build |
| A compile error appears only in `zig build`, never in `zig build test` | Test builds analyze lazily: a function reachable only from an executable is not compiled by the test step. Build the executables before believing the tests |

Entries above the last four were paid for before this file existed; each of the last four was added
in the same change that fixed it.
