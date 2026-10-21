# Vendored OpenCASCADE bindings

`vendor/opencascade-rs/` holds the two Rust binding crates from
[bschwind/opencascade-rs](https://github.com/bschwind/opencascade-rs), at the state of the
commit on pull request #230.

**Only the binding is here.** OpenCASCADE itself is not vendored — `occt-sys` fetches and
builds it as an ordinary dependency, and it is large. What is in this directory is a thin
`cxx` bridge: 108 files, under half a megabyte.

## Why it is vendored, which is not the usual reason

Not for isolation, and not because it was modified first. **The only version of this binding
that compiles on Windows does not exist on any branch.**

OpenCASCADE defines `Handle_X` as a distinct class under MSVC and as a typedef everywhere
else — deliberately, so the type can be exported to C++/CLI, and its header says so. The
binding assumes the typedef form throughout, in constructions *and* in function signatures, so
it builds on Linux and macOS and cannot build on MSVC at all.

That is diagnosed upstream as issue #230, with a fix: give every wrapper-owned handle a
`RustHandle_*` alias and map the public names back with `#[cxx_name]`. The pull request
carrying it was **closed unmerged**, from a fork that has since been **deleted**. GitHub still
serves the commit under `refs/pull/230/head`, and for a while this repository depended on that
— which is to say it depended on GitHub choosing not to garbage-collect an unreferenced object.
That is not a build. Hence: vendored.

## The rule here is the opposite of Fidget's

`vendor/fidget/` must **never be modified**: MPL-2.0 is file-level copyleft, so one edit puts
that file under MPL permanently (D6).

This one **may be modified freely**. It is LGPL-2.1, which attaches at distribution rather than
at editing: if fieldform is ever distributed, the modified source of these two crates must be
offered to whoever receives it. Nothing is distributed today, so nothing is owed today — but
the obligation is real and arrives with the first release.

Every change made here is listed below, because a vendored dependency with undocumented edits
is worse than no vendored dependency. They fall into three groups.

### Removals

1. **The KiCad module.** `opencascade` shipped a KiCad footprint parser and depended on a
   `kicad-parser` crate for it. Nothing here reads PCB files, and vendoring a parser to satisfy
   an unused `pub mod` is how a dependency tree grows without anyone deciding to.
2. **`Error::KicadReadFailed`** went with it.

### Additions, because the upstream API did not reach far enough

3. **Windows system libraries in `build.rs`** — `advapi32`, `user32`, `gdi32`, `shell32`,
   `ole32`, `oleaut32`, `comdlg32`. OCCT's own CMake links these and the crate did not, so the
   build failed at link time on Windows only.
4. **`shape_fix.hxx` / `shape_fix.rs`**, wrapping `BRepBuilderAPI_Sewing` and `ShapeFix_Shape`,
   and **`Shape::sew`** / **`Shape::healed`** over them. This is what makes the importer's
   healing strengths mean anything; without it a nearly-closed STEP body stays open.
5. **`Shape::solids`**, iterating `TopExp_Explorer`. A STEP assembly is a compound, and
   splitting it is the difference between six closed bodies and one mesh with 46 483 open
   edges.
6. **`Mesh::skipped_faces`.** Upstream errors the whole shape when one face has no
   triangulation. An importer reads files it did not write, so one bad face now costs that face
   and is counted, not the import.
7. **`pub use glam;`** in `lib.rs`. The public API takes and returns `glam` types, so a caller
   cannot place a point without it — and depending on it separately lets a consumer pin a
   different version and get a type mismatch instead of an answer.

8. **`Mesher::try_new_angular`**, over a `BRepMesh_IncrementalMesh` constructor upstream does
   not bind. The two-argument form leaves the **angular** deflection at OCCT's default of 0.5
   rad — 28 degrees between neighbouring facet normals — and chord height does not constrain
   that: it bounds distance, and on a small radius a tight chord still turns a large angle.
   Since shading is the normal, that default is what a user sees as faceting on a curved face.
   Measurements in `StepOptions::angular_deflection_deg`.

### A correctness fix to upstream's tests

8. **`occt_test_lock()`, taken by all 20 of the crate's tests.** OCCT is not thread-safe and
   upstream's tests assume it is. Measured on this machine: six runs of the test binary faulted
   three times at the default thread count and zero times at `--test-threads=1`; with the lock,
   eight of eight runs pass. The fault is the library's, not any one test's, so the fix is a
   lock rather than a rewrite of whichever tests happened to collide.

   `fieldform-io` holds the same lock over its own OCCT calls, for the same reason. The two are
   separate locks because they are separate processes, never separate policies.

## How OCCT itself is found, and the two ways that goes wrong

The bindings are vendored here; **OCCT itself is not**. It is a 2.9 GB build sitting at
`target/OCCT`, put there by `occt-sys` — which *is* still a dependency. `opencascade` declares
`default = ["builtin"]`, `builtin` pulls in `opencascade-sys/builtin`, that pulls in `occt-sys`,
and nothing on the way in sets `default-features = false`. Confirm with
`cargo tree -p fieldform-import | grep -i occt`. So the library is reproducible from a fresh
clone — it simply costs hours to reproduce.

**Finding it.** `opencascade-sys/build.rs` configures a tiny CMake project whose only job is
`find_package(OpenCASCADE)`. Pointed at nothing in particular, that search reaches CMake's
**user package registry** — on Windows, `HKCU\Software\Kitware\CMake\Packages\OpenCASCADE`,
which `occt-sys` registered when it built. That entry points at `target/OCCT/build`, the CMake
*build tree*, whose
`OpenCASCADEConfig.cmake` references per-module target files that only exist after install.
Configuring against it fails with `OpenCASCADEFoundationClassesTargets.cmake not found`. The
complete config is next door at `target/OCCT/cmake`.

This hid for a day because `opencascade-sys` only re-runs CMake when it is rebuilt into a fresh
`OUT_DIR`, and nothing did that between vendoring the crate and the first attempt to change a
binding. The failure therefore looked like it was caused by editing a binding, and was not.

The fix is one line in the workspace's `.cargo/config.toml`:

```toml
OpenCASCADE_DIR = { value = "target/OCCT/cmake", relative = true }
```

CMake consults `<Package>_DIR` from the environment ahead of the registry, so this settles it
without touching the registry or this vendored crate. Verified by `cargo clean -p
opencascade-sys` followed by a full rebuild — the fresh configure resolves to `.../OCCT/cmake`.

It is belt-and-braces rather than load-bearing, though, and the reason is the same fact as
above: `builtin` is on, so `build.rs` sets `DEP_OCCT_ROOT` and the `cmake` crate's
`register_dep("occt")` puts `target/OCCT` on `CMAKE_PREFIX_PATH` — which already beats the
registry. Measured with `find_package(OpenCASCADE)` under `cmake -P`, against this machine's
`target/OCCT`:

| environment | resolves to | |
|---|---|---|
| neither set | `target/OCCT/build` (registry) | fails, as above |
| `CMAKE_PREFIX_PATH` only | `target/OCCT/cmake` | ok |
| `OpenCASCADE_DIR` only | `target/OCCT/cmake` | ok |
| both | `target/OCCT/cmake` | ok |

Kept rather than deleted because it names the directory outright instead of trusting the prefix
search to reach `<prefix>/cmake`, and a stale value costs nothing: CMake ignores a
`<Package>_DIR` that does not exist and falls back to the search. That is what happens in a git
worktree, where `relative = true` resolves against a `target/OCCT` that was never built there.

**The landmine.** `target/OCCT` is inside `target/`, so **a plain `cargo clean` deletes 2.9 GB
that costs hours to put back.** It does come back on its own — `builtin` is on, so the next
build of `fieldform-import` fetches `occt-sys` from crates.io and compiles OCCT from source
without being asked. That is the cost, not a broken build: hours, not minutes, and no warning
that it is about to happen. If you need a clean build, clean specific packages
(`cargo clean -p fieldform-app`) rather than everything.

Worth fixing properly one day by moving the install somewhere `cargo clean` does not reach.
**Measured once, 2026-10-15, so the cost is a number and not a fear.** A path mismatch in `target/OCCT/build/CMakeCache.txt` (a worktree had been junctioned to the directory and configured through it) made `cargo build --workspace` exit 101 at configure from the main checkout for five days, on the belief that fixing it meant the full rebuild above. It did not: the next `cargo build --workspace` reconfigured, MSBuild's incremental build accepted every existing object — zero recompiled under `target/OCCT` — and the whole thing, bindings and importer included, took **12 min 25 s** on 16 threads, with a second run a 4 s no-op. So a *reconfigure* is minutes; only deleting the directory is hours. The two are different events, and the first was mistaken for the second for longer than the fix took.

Reproducibility is not the gap — `builtin` already guarantees a fresh clone can rebuild OCCT.
The gap is that the rebuild is expensive and an ordinary command triggers it silently.

## What would let us drop this

The fix landing upstream, on any branch, in a released version. If that happens, this directory
goes and the dependency becomes an ordinary version requirement again. Worth checking
occasionally; worth not waiting for.
