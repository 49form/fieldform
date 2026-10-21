// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What happens where two extrudes meet face to face, and why a cut through them shows a sliver.
//!
//! Reported from the application: a profile extruded one way, the same profile extruded the
//! other, unioned, and the union used to subtract — and something appears between the two where
//! there should be nothing. Is it the picture or the geometry?
//!
//! Run with: `cargo run --release -p fieldform-field --example opposed_extrudes`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{BooleanOp, Curve, CurveId, Document, Graph, Node, ParamRef, Place, Sweep};
use nalgebra::{Point3, Vector3};

const HALF: f64 = 20.0;
const DEPTH: f64 = 15.0;
/// What the viewport uses: `radius * 1e-5`, so about this for a part of this size.
const EPSILON: f64 = 1e-3;
/// Where the rays in this measurement start: inside the cut, above the join.
const FROM: f64 = 12.0;

fn main() {
    let cutter = built(Shape::Union);
    println!(
        "A {} mm square extruded +{DEPTH} and -{DEPTH} from the same plane, unioned.\n\
         Down the middle, through the face the two share:\n",
        HALF * 2.0
    );
    println!("  {:>8}  {:>12}   {}", "z", "union", "");
    for z in [-4.0f64, -2.0, -1.0, -0.1, -0.001, 0.0, 0.001, 0.1, 1.0, 2.0, 4.0] {
        let d = cutter.distance(Point3::new(0.0, 0.0, z));
        let note = if d.abs() < EPSILON { "<-- reads as a surface" } else { "" };
        println!("  {z:>8.3}  {d:>12.6}   {note}");
    }
    println!(
        "\nThe nearest real boundary from the middle of that join is {DEPTH} mm away (either end)\n\
         or {HALF} mm (the wall). The field says nought. `min` answers with the nearer *surface*,\n\
         and where two solids share a face both surfaces are exactly there \u{2014} so the union\n\
         carries a sheet of zero straight through solid material.\n"
    );

    // Inside the union that sheet is harmless: a march from outside stops at the outer wall and
    // never reaches it. Negating it is what brings it into open space, where rays go.
    let cut = built(Shape::Difference);
    println!(
        "Subtract that union from a larger box and the sheet lands in the cavity. Marching one\n\
         ray down the z axis from z = {FROM}, exactly as the viewport does:\n"
    );
    match march(&cut, down_from(FROM)) {
        Some((t, steps)) => println!(
            "  stopped after {steps} steps at z = {:.6} \u{2014} which is the join, not a wall",
            FROM - t
        ),
        None => println!("  crossed the cavity without stopping"),
    }

    // How far apart the two have to be pushed before that stops.
    //
    // Not "any overlap at all": inside an overlap of `w` the nearer face is at most `w/2` away,
    // so `min` still reads `-w/2` in the middle of it. The sheet goes when that clears the hit
    // threshold, which the table finds rather than assumes.
    println!("\nHow much the two have to overlap before it stops \u{2014} threshold {EPSILON} mm:\n");
    println!("  {:>10}  {:>14}  {}", "overlap", "worst reading", "");
    for over in [0.0f64, 0.0005, 0.002, 0.005, 0.02, 0.1] {
        let field = built(Shape::Overlapped(over));
        let worst = field.distance(Point3::new(0.0, 0.0, over / 2.0));
        let outcome = match march(&field, down_from(FROM)) {
            Some((t, _)) => format!("stops at z = {:.5}", FROM - t),
            None => "crosses cleanly".to_string(),
        };
        println!("  {over:>10.4}  {worst:>14.6}  {outcome}");
    }

    // And what it would take to fix in the marcher rather than in the model: confirm a small
    // reading by stepping past it and asking whether the field actually went negative.
    println!(
        "\nOr confirm the hit instead of trusting a small number. Same ray, same field, but a\n\
         candidate is only accepted if the field is negative a little further along:"
    );
    match march_confirmed(&cut, down_from(FROM)) {
        Some((t, steps)) => {
            println!("  stopped after {steps} steps at z = {:.5}, {t:.3} mm along", FROM - t)
        }
        None => println!("  crossed the cavity without stopping"),
    }
    // The floor of the cavity is the far end of the downward extrude, so that is the answer to
    // check against \u{2014} reached only by a ray that went through the sheet.
    println!("  (the floor of the cut is at z = -{DEPTH}, {:.3} mm along)", FROM + DEPTH);

    println!(
        "\nSo: not the picture, and not the shapes. The shapes touch exactly as drawn. It is the\n\
         field describing their union, and it shows because a march stops where the value is\n\
         small \u{2014} which cannot tell `there is a surface here` from `this bound is loose\n\
         here`. Three ways out, different in kind:\n\n\
         \x20 \u{2022} Do not build the shape that way. One extrude, symmetric about its plane,\n\
         \x20   has no join to understate.\n\
         \x20 \u{2022} Overlap them by more than the threshold \u{2014} not `by a bit`: half the\n\
         \x20   overlap is what `min` reads in the middle, so it has to clear it twice over.\n\
         \x20 \u{2022} Confirm hits in the marcher, which costs one extra sample per pixel that\n\
         \x20   hits anything and fixes every such join at once, wherever it came from."
    );
}

/// A ray heading straight down the z axis.
fn down_from(z: f64) -> (Point3<f64>, Vector3<f64>) {
    (Point3::new(0.0, 0.0, z), Vector3::new(0.0, 0.0, -1.0))
}

/// Sphere tracing as `march.wgsl` does it, so the answer is about the viewport and not about a
/// tracer written to be kind.
fn march(field: &GraphField, ray: (Point3<f64>, Vector3<f64>)) -> Option<(f64, usize)> {
    let (from, dir) = ray;
    let mut t = 0.0;
    for step in 0..512 {
        if t > 80.0 {
            return None;
        }
        let d = field.distance(from + dir * t);
        if d < EPSILON {
            return Some((t, step));
        }
        t += (d * 0.9).max(EPSILON);
    }
    None
}

/// The same, except that a small reading is a *candidate* and has to be confirmed.
///
/// A real surface approached head on is a sign change: a little past it the field is negative.
/// A loose bound is not — the value dips towards nought and comes back up, because there was
/// never a surface there. This tells the two apart for the cost of one sample.
fn march_confirmed(field: &GraphField, ray: (Point3<f64>, Vector3<f64>)) -> Option<(f64, usize)> {
    let (from, dir) = ray;
    let mut t = 0.0;
    for step in 0..512 {
        if t > 80.0 {
            return None;
        }
        let d = field.distance(from + dir * t);
        if d < EPSILON {
            if field.distance(from + dir * (t + 4.0 * EPSILON)) < 0.0 {
                return Some((t, step));
            }
            // Not a surface. Step past the whole confirmation window, or the next reading is
            // the same small number and the march stands still.
            t += 4.0 * EPSILON;
            continue;
        }
        t += (d * 0.9).max(EPSILON);
    }
    None
}

enum Shape {
    Union,
    Difference,
    /// The same cut, with the downward extrude pushed up through the join by this much.
    Overlapped(f64),
}

fn built(shape: Shape) -> GraphField {
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
    let over = match shape {
        Shape::Overlapped(by) => by,
        _ => 0.0,
    };
    let up = graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        sweep: Sweep::Normal,
        height: lit(DEPTH),
        draft: lit(0.0),
        start: lit(0.0),
        sides: fieldform_scene::Sides::One,
        corners: Default::default(),
    });
    let mut down = graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        sweep: Sweep::Normal,
        height: lit(-DEPTH),
        draft: lit(0.0),
        start: lit(0.0),
        sides: fieldform_scene::Sides::One,
        corners: Default::default(),
    });
    if over > 0.0 {
        // Lifted, not lengthened: the join is at the *top* of this one, and pushing its far end
        // further down leaves the two still meeting exactly at nought.
        down = graph.add(Node::Transform {
            input: down.into(),
            translation: [lit(0.0), lit(0.0), lit(over)],
            rotation: [lit(0.0), lit(0.0), lit(0.0)],
            scale: lit(1.0),
            about: fieldform_scene::Place::origin(),
        });
    }
    let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![up.into(), down.into()] });
    if matches!(shape, Shape::Union) {
        return GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
    }
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        size: [lit(80.0), lit(80.0), lit(60.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), union.into()] });
    GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles")
}
