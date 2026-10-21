# Vendored dependency: Fidget

Source: https://github.com/mkeeter/fidget
Vendored from `main` branch commit `5cadd18d42a48249060f145e092b14754a7b5c20`
("Move operations onto types (#452)"), dated 2026-08-13.

This is a plain vendored copy (no nested `.git`, no submodule) — it's tracked directly in
this repo's history so it can be modified freely as part of `fieldform-field`'s integration
with Fidget's `Function`/`Shape` traits. License: MIT/Apache-2.0 (see `fidget/LICENSE.txt`).

To pull upstream changes later: re-clone the desired commit/tag from upstream and diff
against this directory, or re-establish it as a proper `git subtree`/fork if ongoing
upstream sync becomes a priority.

## License note (important)

Fidget is **MPL-2.0** (Mozilla Public License 2.0), not MIT/Apache as first assumed during
planning. MPL-2.0 is file-level copyleft: modifying and distributing an MPL-licensed *file*
requires that file's source to stay available under MPL-2.0, but it does not require the
rest of a combined work (code that merely depends on/links against it) to be open-sourced.
To keep `fieldform`'s own code proprietary with a clean boundary, the working rule is:
**this `vendor/fidget/` directory stays unmodified**; all fieldform-specific logic
(cage editing, SubD/field bridging, scene composition, app) lives in the separate
`fieldform-*` crates, which merely depend on these crates as a library. If a change to
Fidget's own internals ever becomes unavoidable, treat that specific file's MPL obligations
deliberately rather than by accident.

## Merged into the workspace root

`vendor/fidget/Cargo.toml` (the old workspace manifest) and `Cargo.lock` were removed —
their `[workspace.package]`, `[profile.*]`, and `[workspace.dependencies]` content now lives
in the top-level `fieldform/Cargo.toml`, which is the single workspace root covering both
the vendored Fidget crates and the `fieldform-*` crates. `vendor/fidget/demos/*` and
`vendor/fidget/models` were intentionally left out of the workspace members list (not
needed as build targets; still there as reference source).

`vendor/fidget/.cargo/config.toml` sets `wasm32-unknown-unknown` rustflags — irrelevant for
the desktop-first v1, but will need to move to the root `.cargo/config.toml` once a WASM
viewer target is added later.

## A vendored test that fails intermittently on Windows: `fidget-wgpu`'s `ssao_bias`

**Seen 2026-09-27.** `cargo test --workspace` failed once in three completed runs, in vendored
code, on an RTX A500 with the D3D12 backend:

```
test effects::test::ssao_bias ... FAILED
wgpu error: Validation Error
  In Device::create_compute_pipeline, label = 'interval root (8)'
    Internal error: FXC D3DCompile error (Unspecified error (0x80004005)):
    fidget-wgpu/Shader@0x...(931,5-15): error X3511: unable to unroll loop, loop does not
    appear to terminate in a timely manner (14 iterations) or unrolled loop is too large,
    use the [unroll(n)] attribute to force an exact higher number
```

What is established, and what is not:

- **It is not ours.** `vendor/fidget/fidget-wgpu/src/effects/mod.rs`, in Fidget's own test, in a
  pipeline Fidget's own codegen produced. Nothing in `fieldform-*` calls it.
- **The test is deterministic.** It builds one fixed tree — a unit sphere `min` the Z plane — so
  the shader source is the same on every run. An input-dependent explanation is ruled out.
- **It passes in isolation.** Three runs of `cargo test -p fidget-wgpu --lib ssao_bias` in a row,
  all green, on the build that had just failed. The failure has only been seen inside a whole
  workspace run, where many shaders compile at once, which points at FXC's own budget under
  load rather than at the shader.
- **It does not reach the product.** The application's viewport is our own WGSL
  (`fieldform-gpu`); Fidget is used through its `Function`/`Shape` traits on the CPU. A red
  workspace run from this is a red run and not a broken build.

**Why it is written down rather than fixed.** The licence rule above is that this directory stays
unmodified, and `[unroll(n)]` on Fidget's loop would be a modification of an MPL-2.0 file. So the
options are upstream or nothing.

**Two attempts to make it fail on demand, and both failed.** Single-threaded
(`cargo test -p fidget-wgpu --lib -- --test-threads=1`) passes, and a second full workspace run
passes. So the load hypothesis above is *not* confirmed either, and one failure in four completed
workspace runs is the whole of the evidence. Written at that strength on purpose: it is an
observation, not a reproduction.

**Before reporting it upstream**, it needs a repro that fails on demand. Intermittent-under-load
is not something a maintainer can act on, and the interesting question is what makes FXC give up
on a 14-iteration loop sometimes and not others — concurrency, memory, or a pipeline cache. Worth
trying: run the workspace suite with `--test-threads=1` and see whether it stops, then with many
threads and see whether it becomes reliable. If it can be made to fail reliably, the report writes
itself: the shader, the FXC error, the driver, and the loop at line 931.

**The unused dependency noticed while looking, now removed.** `fieldform-app` declared both
`fidget-core` and `fidget-wgpu` and referenced neither — the only mention of Fidget anywhere in
that crate is the attribution in the About window, which is about the *product* and stays right
either way. `fieldform-field`'s `fidget-wgpu` is a dev-dependency already and is genuinely used,
by `examples/gpu_lattice.rs`.

Removing those two lines takes three crates out of the application's build tree —
`fidget-bytecode`, `fidget-raster` and `fidget-wgpu` — leaving `fidget-core` and `fidget-mesh`,
which arrive transitively through `fieldform-field` and `fieldform-io` and are genuinely used.
The binary is **byte-identical** at 22,183,424, so the linker was already dropping the code; what
is saved is compiling three crates, not shipping them.

It does not stop this test running. `fidget-wgpu` is a *workspace member*, so
`cargo test --workspace` builds and runs it whatever our crates depend on.

## Nothing shipped depends on it, 2026-09-29

`fidget-core` was a dependency of `fieldform-field` for exactly one file — `src/bridge.rs`, 245
lines, the M0 spike that embedded a cage field in a Fidget expression as a per-point `Var::V`. It
exported `FieldTerm` and `smooth_union`, and **nothing outside its own tests ever called either**;
every other mention of those names in the tree was a comment.

What happened instead, over the year it sat there: the viewport became generated WGSL
(`fieldform-gpu`), meshing became our own dual contouring (`fieldform-field/src/mesh.rs`, whose
header says why Fidget's mesher could not drive our fields), and the composition graph came to
evaluate `ScalarField`s directly (`graph_field.rs`). The bridge was the road not taken, still
paved.

So it is deleted, and `fidget-core` moved to `[dev-dependencies]`. `fidget-wgpu` and
`fidget-raster` were already dev-only. Three examples — `brick_pruning`, `brick_timing`,
`gpu_lattice` — still measure our interval pruning and marching against Fidget's tape, which is
the second opinion worth keeping on a numeric path and costs nothing, because a dev-dependency
never ships.

Verified rather than argued: `cargo tree -p fieldform-app -e normal` and the same for
`fieldform-import` list no fidget crate at all. `cargo check --workspace --all-targets` is clean,
examples included.

**The licence rule above is unchanged.** The source is still vendored and still MPL-2.0, and an
edit to one of its files still makes that file MPL permanently. What has gone is the *pressure* on
that boundary: there is no product code on the other side of it to tempt a change.

**A third name on the same flake, 2026-10-02: `fidget-raster`'s `wgpu::sphere_wgpu`.** It failed
once in a workspace run and passed on three re-runs of the same binary, which is the behaviour
`ssao_bias` and `pixel_render` already have. Nothing of ours is involved and nothing under
`vendor/fidget/` may be touched to fix it (D6), so the standing arrangement is the answer:
`--no-fail-fast` is mandatory on every workspace run, because without it the run truncates at the
first of these and the STEP canary silently reads zero.
