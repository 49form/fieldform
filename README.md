# FieldForm

FieldForm is a field-based CAD kernel and modeller written in Rust. Geometry is a function from
space to distance — a signed distance field — rather than a mesh or a boundary representation, so
blends, lattices and patterns compose as functions and are drawn by tracing the field directly on
the GPU. Control cages, curves and parameters feed a composition graph that is serialized and
re-evaluated.

> **Status: early development. Not yet supported, and not accepting contributions.**
> Interfaces, file formats and behaviour will change without notice.

## Licence

| Layer | Crates | Licence |
| --- | --- | --- |
| Kernel | `fieldform-cage`, `-subd`, `-field`, `-scene`, `-ops`, `-io`, `-gpu` | MPL-2.0 |
| Application | `fieldform-app`, `fieldform-import` | GPL-3.0-only |

See [LICENSE](LICENSE) for the full table, `LICENSE-MPL-2.0` and `LICENSE-GPL-3.0` for the texts, and
[NOTICE](NOTICE) for third-party attributions.

Models and files created with FieldForm belong to the user and are not covered by the software licence.

## Building

```bash
cargo run -p fieldform-app
cargo test --workspace --no-fail-fast
```

The STEP importer is a separate binary that links OpenCASCADE; build it on its own:
`cargo build --release -p fieldform-app` then `cargo build --release -p fieldform-import`, and ship them in the same directory.
