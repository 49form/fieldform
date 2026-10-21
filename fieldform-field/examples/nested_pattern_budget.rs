// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a pattern of a pattern costs the march, and whether it is the hardware or the divisor.
//!
//! **Reported**: "pattern of a pattern that is a pipe really kills the viewport when there is a
//! sweep active - then a pattern of that sweep while there is a pattern of a pipe also kills it
//! quite a bit - not sure if bug or just limitation of HW".
//!
//! The question is answerable without a GPU, because the thing most likely to be doing it is a
//! number this crate computes: `step_multiplier`. The viewport buys its march budget as
//! `512 · multiplier · 2`, clamped at 16 384 (`traced/rebuild.rs`), and **every divisor along a
//! chain multiplies** — `step_multiplier`'s own walk takes `own · deepest`. A sweep divides by the
//! convergence of its section planes; a rounded pattern divides by `W^(1/n)`; a pipe of a varying
//! radius divides by its own bound. Stack three of those and the budget is bought against their
//! product, so the march does that many times the work *per pixel* whether or not the geometry
//! needs it.
//!
//! So this prints, for each shape, the instances the field evaluates, the multiplier, and the
//! budget the viewport would ask for — and whether that budget is at the cap, which is the state
//! where the viewport has stopped being interactive by arithmetic rather than by hardware.
//!
//! Run with: `cargo run --release -p fieldform-field --example nested_pattern_budget`

use fieldform_scene::{
    Continuity, Curve, CurveId, Document, Fillet, Frame, Graph, Node, ParamRef, PatternKind,
};
use nalgebra::Point3;

/// `traced/mod.rs`'s own numbers.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// A spine to sweep or pipe, bent so a sweep's sections really do converge.
fn spine(doc: &mut Document, name: &str) -> CurveId {
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            name,
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(40.0, 10.0, 5.0),
                Point3::new(80.0, -5.0, 15.0),
            ],
            false,
        )
        .expect("a bent line"),
    );
    id
}

/// The same spine, held well clear of the pattern's axis.
///
/// **The control, and the reason the table can be read at all.** A radial pattern's window is how
/// many angular cells its input spans, so an input that *touches the axis* spans every one of them
/// and the prune cannot cut anything — truthfully, since at the axis every copy really is
/// adjacent. The spine above starts at the origin, which is exactly that case. If this row prunes
/// and that one does not, the window is working and the shape is what costs; if neither prunes,
/// the prune itself is broken.
fn offset_spine(doc: &mut Document, name: &str) -> CurveId {
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            name,
            vec![
                Point3::new(60.0, 0.0, 0.0),
                Point3::new(75.0, 6.0, 5.0),
                Point3::new(90.0, -3.0, 10.0),
            ],
            false,
        )
        .expect("a bent line"),
    );
    id
}

/// A closed square, for a sweep's section.
fn section(doc: &mut Document) -> CurveId {
    let id = CurveId::new();
    let h = 4.0;
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            "section",
            vec![
                Point3::new(-h, -h, 0.0),
                Point3::new(h, -h, 0.0),
                Point3::new(h, h, 0.0),
                Point3::new(-h, h, 0.0),
            ],
            true,
        )
        .expect("closed"),
    );
    id
}

fn pattern(graph: &mut Graph, input: fieldform_scene::NodeId, count: u32, k: f64) -> fieldform_scene::NodeId {
    graph.add(Node::Pattern {
        options: Default::default(),
        input: input.into(),
        count: lit(f64::from(count)),
        kind: PatternKind::Radial { frame: Frame::default() },
        join: k > 0.0,
        fillet: Fillet { radius: lit(k), continuity: Continuity::G1 },
    })
}

/// Every shape the report names, and the controls either side of them.
#[allow(clippy::type_complexity)]
fn shapes() -> Vec<(&'static str, Box<dyn Fn(&mut Document, &mut Graph)>)> {
    vec![
        ("a pipe", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            g.add(Node::Pipe { curve: c, radius: lit(3.0) });
        })),
        ("a pattern of a pipe", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            pattern(g, p, 6, 0.0);
        })),
        ("a pattern of a pattern of a pipe", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            let one = pattern(g, p, 6, 0.0);
            pattern(g, one, 6, 0.0);
        })),
        // The same two, rounded, which is where the divisor actually comes from.
        ("a rounded pattern of a pipe", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            pattern(g, p, 6, 2.0);
        })),
        ("a rounded pattern of a rounded pattern of a pipe", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            let one = pattern(g, p, 6, 2.0);
            pattern(g, one, 6, 2.0);
        })),
        ("a sweep", Box::new(|doc: &mut Document, g: &mut Graph| {
            let sec = section(doc);
            let sp = spine(doc, "path");
            g.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(sec),
                spine: sp,
                section: fieldform_scene::Section::Square,
                law: Default::default(),
            });
        })),
        ("a pattern of a sweep", Box::new(|doc: &mut Document, g: &mut Graph| {
            let sec = section(doc);
            let sp = spine(doc, "path");
            let s = g.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(sec),
                spine: sp,
                section: fieldform_scene::Section::Square,
                law: Default::default(),
            });
            pattern(g, s, 6, 0.0);
        })),
        // **The reported scene**: both at once, as separate bodies in one part, which is what
        // "while there is a sweep active" means -- a part's budget is the worst of its roots.
        ("a pattern of a pattern of a pipe, and a sweep", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            let one = pattern(g, p, 6, 0.0);
            pattern(g, one, 6, 0.0);
            let sec = section(doc);
            let sp = spine(doc, "path");
            g.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(sec),
                spine: sp,
                section: fieldform_scene::Section::Square,
                law: Default::default(),
            });
        })),
        ("a pattern of a sweep, and a pattern of a pipe", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            pattern(g, p, 6, 0.0);
            let sec = section(doc);
            let sp = spine(doc, "path");
            let s = g.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(sec),
                spine: sp,
                section: fieldform_scene::Section::Square,
                law: Default::default(),
            });
            pattern(g, s, 6, 0.0);
        })),
        // **The controls.** The same patterns over a pipe held clear of the axis, where the
        // window has somewhere to cut.
        ("control: a pattern of a pipe off the axis", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = offset_spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            pattern(g, p, 6, 0.0);
        })),
        ("control: 24 copies of a pipe off the axis", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = offset_spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            pattern(g, p, 24, 0.0);
        })),
        ("control: a pattern of a pattern of a pipe off the axis", Box::new(|doc: &mut Document, g: &mut Graph| {
            let c = offset_spine(doc, "spine");
            let p = g.add(Node::Pipe { curve: c, radius: lit(3.0) });
            let one = pattern(g, p, 6, 0.0);
            pattern(g, one, 6, 0.0);
        })),
        ("a pattern of a pattern of a sweep", Box::new(|doc: &mut Document, g: &mut Graph| {
            let sec = section(doc);
            let sp = spine(doc, "path");
            let s = g.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(sec),
                spine: sp,
                section: fieldform_scene::Section::Square,
                law: Default::default(),
            });
            let one = pattern(g, s, 6, 0.0);
            pattern(g, one, 6, 0.0);
        })),
    ]
}

fn main() {
    println!(
        "{:<52} {:>9} {:>11} {:>8} {:>7} {:>9}",
        "shape", "instances", "multiplier", "budget", "windows", "walks/sample"
    );
    for (label, build) in shapes() {
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        build(&mut doc, &mut graph);

        // How many copies of the leaf the field really evaluates, which is the honest measure of
        // "how much geometry is here" and is *not* what the multiplier is about.
        let instances: u32 = graph
            .nodes()
            .filter_map(|(_, n)| match n {
                Node::Pattern { count, .. } => {
                    fieldform_scene::pattern_count(&doc, count).ok().map(|(n, _)| n)
                }
                _ => None,
            })
            .product::<u32>()
            .max(1);

        let multiplier = fieldform_field::step_multiplier(&doc, &graph);
        let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
            .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);

        // **What the prune actually buys**, per pattern, innermost first. `pattern_reach` sizes
        // the window from the extent of the pattern's *input*, so a pattern whose input is small
        // beside its pitch reads a few copies and a pattern whose input already spans the ring
        // reads all of them. Item 34's whole saving is the first case.
        let mut windows: Vec<String> = Vec::new();
        let mut walks = 1usize;
        for (id, node) in graph.nodes() {
            if let Node::Pattern { input, count, kind, fillet, .. } = node {
                let Ok((n, _)) = fieldform_scene::pattern_count(&doc, count) else { continue };
                let k = fillet.radius.resolve(&|pid| doc.parameter_value(pid)).unwrap_or(0.0);
                let reach =
                    fieldform_field::pattern_reach(&doc, &graph, input.source().expect("a pattern of something"), kind, n, k);
                // The copies one sample folds: `2·reach + 1`, or every copy when that covers them.
                let summed = (2 * reach + 1).min(n as usize).max(1);
                windows.push(format!("{summed}/{n}"));
                walks *= summed;
                let _ = id;
            }
        }
        windows.reverse();
        println!(
            "{label:<52} {instances:>9} {multiplier:>11.2} {budget:>8} {:>7} {walks:>9}",
            if windows.is_empty() { "-".to_string() } else { windows.join(" x ") }
        );
    }
    println!(
        "\n`budget` is steps per pixel, from `512 x multiplier x 2` clamped at {MAX_STEP_BUDGET}\n\
         (traced.rs). `windows` is what each pattern's prune leaves, innermost first: `3/6` means\n\
         a sample folds three of six copies. `walks/sample` is their product -- how many times one\n\
         field sample walks the leaf, which for a pipe is a whole spine scan and for a sweep is a\n\
         section solve.\n\n\
         Read the last column against the first: where `walks/sample` is far below `instances` the\n\
         prune is working and the cost is the geometry. Where it equals `instances` the prune has\n\
         bought nothing, and the frame is paying for every copy at every step of every ray."
    );
}
