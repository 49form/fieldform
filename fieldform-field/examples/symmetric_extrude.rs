// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What "both directions" buys over two extrudes and a union.
//!
//! Both make the same solid. One of them describes it with a join `min` cannot express — see
//! `opposed_extrudes.rs` — and the other has no join in it at all.
//!
//! Run with: `cargo run --release -p fieldform-field --example symmetric_extrude`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{BooleanOp, Curve, CurveId, Document, Graph, Node, ParamRef, Sweep};
use nalgebra::Point3;

const HALF: f64 = 20.0;
const DEPTH: f64 = 30.0;

fn main() {
    let joined = built(How::TwoAndAUnion);
    let single = built(How::Symmetric);

    println!(
        "A {} mm square, {DEPTH} mm deep about its plane, built two ways.\n\
         Down the middle, through the plane:\n",
        HALF * 2.0
    );
    println!("  {:>8}  {:>14}  {:>14}", "z", "two + union", "both directions");
    for z in [-16.0f64, -15.0, -8.0, -0.5, 0.0, 0.5, 8.0, 15.0, 16.0] {
        let p = Point3::new(0.0, 0.0, z);
        println!("  {z:>8.2}  {:>14.6}  {:>14.6}", joined.distance(p), single.distance(p));
    }
    println!(
        "\nAt the plane the union reads {:.6}: both halves have a face there, so `min` reports a\n\
         surface through solid material. The single block reads {:.6} \u{2014} the distance to the\n\
         nearer of the two ends, {:.0} mm away, which is the truth.",
        joined.distance(Point3::origin()),
        single.distance(Point3::origin()),
        DEPTH * 0.5
    );

    // **The same solid, not the same field**, and the difference is the whole point. `min` is
    // exact outside both operands and only a bound inside them, so the two agree everywhere
    // outside the material and disagree throughout the inside — the union reading short by up
    // to the half-depth, worst of all at the join.
    //
    // So what has to hold is the *shape*: the sign agrees everywhere, and the values agree
    // wherever the union is exact. Checked rather than assumed — a fold that got its sign wrong
    // would put material on one side only, and nothing else here would notice.
    let (mut disagreed, mut worst_out, mut worst_in) = (0usize, 0.0f64, 0.0f64);
    let mut samples = 0usize;
    for i in 0..40 {
        for j in 0..40 {
            for k in 0..40 {
                let p = Point3::new(
                    -30.0 + 60.0 * i as f64 / 39.0,
                    -30.0 + 60.0 * j as f64 / 39.0,
                    -25.0 + 50.0 * k as f64 / 39.0,
                );
                let (a, b) = (joined.distance(p), single.distance(p));
                samples += 1;
                if (a > 0.0) != (b > 0.0) {
                    disagreed += 1;
                }
                let gap = (a - b).abs();
                if b > 0.0 {
                    worst_out = worst_out.max(gap);
                } else {
                    worst_in = worst_in.max(gap);
                }
            }
        }
    }
    println!(
        "\nOver {samples} points across the box: the two disagree about inside or outside at\n\
         {disagreed} of them, and outside the solid \u{2014} where `min` is exact \u{2014} the\n\
         readings differ by at most {worst_out:.9} mm. Inside they differ by up to\n\
         {worst_in:.3} mm, which is the union understating its own interior and is what a\n\
         subtract turns into a visible sliver."
    );

    // And with a draft, where the profile is the widest section rather than the base.
    let drafted = built(How::SymmetricDrafted(20.0));
    println!(
        "\nWith 20\u{b0} of draft, the walls open away from the plane both ways. Half-width of\n\
         the solid at each height, which should shrink the same amount above and below:\n"
    );
    println!("  {:>8}  {:>12}", "z", "half-width");
    for z in [-12.0f64, -6.0, 0.0, 6.0, 12.0] {
        // Walk out along +X until the field turns positive; that is the wall.
        let mut x = 0.0;
        while x < 60.0 && drafted.distance(Point3::new(x, 0.0, z)) < 0.0 {
            x += 0.001;
        }
        println!("  {z:>8.1}  {x:>12.3}");
    }
    println!(
        "\nThe drawn profile is {HALF} mm and sits at z = 0, which is the widest section. \
         Both\nsides narrow at tan 20\u{b0} = {:.4} per millimetre, symmetrically.",
        20f64.to_radians().tan()
    );
}

enum How {
    /// The shape as it had to be built before: up, down, and `min` between them.
    TwoAndAUnion,
    Symmetric,
    SymmetricDrafted(f64),
}

fn built(how: How) -> GraphField {
    let mut doc = Document::new();
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            "Square".to_string(),
            vec![
                Point3::new(-HALF, -HALF, 0.0),
                Point3::new(HALF, -HALF, 0.0),
                Point3::new(HALF, HALF, 0.0),
                Point3::new(-HALF, HALF, 0.0),
            ],
            true,
        )
        .expect("closed"),
    );
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    match how {
        How::TwoAndAUnion => {
            let up = graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: Sweep::Normal,
                height: lit(DEPTH * 0.5),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            let down = graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: Sweep::Normal,
                height: lit(-DEPTH * 0.5),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            graph.add(Node::Boolean {
                op: BooleanOp::Union,
                inputs: vec![up.into(), down.into()],
            });
        }
        How::Symmetric => {
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: Sweep::Normal,
                height: lit(DEPTH),
                draft: lit(0.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::Symmetric,
                corners: Default::default(),
            });
        }
        How::SymmetricDrafted(degrees) => {
            graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(id),
                sweep: Sweep::Normal,
                height: lit(DEPTH),
                // Negative closes in, which is what makes the profile the widest section and
                // both halves draw out of a mould that parts on its plane.
                draft: lit(-degrees),
                start: lit(0.0),
                sides: fieldform_scene::Sides::Symmetric,
                corners: Default::default(),
            });
        }
    }
    GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles")
}
