// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A skewed box as the graph sees it: framed correctly, refused when flat, exact under a blend.
//!
//! The field's own properties (an exact distance, the unit gradient) are held where it is
//! defined, in `primitive.rs`. What is held here is what only the graph can get wrong: the extent
//! the per-tile cull reads, the compile-time refusal, and the walk that decides whether a blend
//! against it owes the panel an `≈ approximate` mark.

use fieldform_field::{
    approximate_blends, carries_true_distance, graph_extent, graph_lipschitz, CompileError, Extent,
    GraphField, Normalization, ScalarField,
};
use fieldform_scene::{BooleanOp, Document, Fillet, Graph, Node, NodeId, ParamRef, Place};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn skewed(centre: Place, edges: [[f64; 3]; 3]) -> Node {
    Node::SkewedBox { centre, edges: edges.map(|e| e.map(lit)) }
}

/// The leaning case the scene tests use: nothing about it is square to an axis.
const LEANING: [[f64; 3]; 3] = [[10.0, 0.0, 0.0], [4.0, 8.0, 0.0], [3.0, 2.0, 6.0]];

fn one(node: Node) -> (Document, Graph, NodeId) {
    let mut graph = Graph::empty();
    let id = graph.add(node);
    (Document::new(), graph, id)
}

/// **The extent is the box of the eight corners, and it holds every solid point.** The cull skips
/// a body wherever its box misses the tile, so a box short of one corner is a bite out of the
/// part. Sampled on the compiled field, on a grid over the box grown by half its size on each
/// side, because the field is what is drawn.
#[test]
fn a_skewed_box_s_extent_holds_every_solid_point() {
    let (doc, graph, id) = one(skewed(Place::at(5.0, -3.0, 8.0), LEANING));
    let Extent::Bounded { lo, hi } = graph_extent(&doc, &graph, id) else {
        panic!("a box is bounded");
    };
    let field = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
    let n = 61usize;
    let grow = |a: f64, b: f64| ((b - a) * 0.5).max(10.0);
    let (glo, ghi) = (
        Point3::new(lo.x - grow(lo.x, hi.x), lo.y - grow(lo.y, hi.y), lo.z - grow(lo.z, hi.z)),
        Point3::new(hi.x + grow(lo.x, hi.x), hi.y + grow(lo.y, hi.y), hi.z + grow(lo.z, hi.z)),
    );
    let (mut solid, mut outside) = (0usize, 0usize);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = |c: usize| c as f64 / (n - 1) as f64;
                let p = Point3::new(
                    glo.x + (ghi.x - glo.x) * t(i),
                    glo.y + (ghi.y - glo.y) * t(j),
                    glo.z + (ghi.z - glo.z) * t(k),
                );
                if field.distance(p) < -1e-6 {
                    solid += 1;
                    let inside = (lo.x..=hi.x).contains(&p.x)
                        && (lo.y..=hi.y).contains(&p.y)
                        && (lo.z..=hi.z).contains(&p.z);
                    if !inside {
                        outside += 1;
                    }
                }
            }
        }
    }
    assert!(solid > 0, "the case measured nothing");
    assert_eq!(outside, 0, "{outside} of {solid} solid samples lie outside {lo:?}..{hi:?}");
}

/// The box is *tight*, not merely safe: the first edge reaches x = 5 ± (10 + 4 + 3)/2 and no
/// further, so a box padded by a margin would pass the test above and fail this one. A frame
/// that over-covers costs every tile a march it did not need.
#[test]
fn a_skewed_box_s_extent_is_the_box_of_its_corners_and_no_wider() {
    let (doc, graph, id) = one(skewed(Place::at(5.0, -3.0, 8.0), LEANING));
    let Extent::Bounded { lo, hi } = graph_extent(&doc, &graph, id) else {
        panic!("a box is bounded");
    };
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(close(lo.x, 5.0 - 8.5) && close(hi.x, 5.0 + 8.5), "x: {lo:?} {hi:?}");
    assert!(close(lo.y, -3.0 - 5.0) && close(hi.y, -3.0 + 5.0), "y: {lo:?} {hi:?}");
    assert!(close(lo.z, 8.0 - 3.0) && close(hi.z, 8.0 + 3.0), "z: {lo:?} {hi:?}");
}

/// Three edges in one plane enclose nothing. The refusal names the node, so the panel can say
/// which block to change, and it is raised at compile rather than as a field of NaN.
#[test]
fn three_coplanar_edges_are_refused_naming_the_node() {
    let (doc, graph, id) =
        one(skewed(Place::at(0.0, 0.0, 0.0), [[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [5.0, 4.0, 0.0]]));
    let err = GraphField::compile_graph(&doc, &graph, 1e-4).err().expect("a flat box is refused");
    assert!(matches!(err, CompileError::FlatSkewedBox(n) if n == id), "{err}");
}

/// A box that leans is a true distance, so both walks agree it is one and a blend against it is
/// not marked approximate — the property the compiler and `carries_true_distance` must share.
#[test]
fn a_skewed_box_is_a_true_distance_and_a_blend_against_it_is_exact() {
    let mut graph = Graph::empty();
    let a = graph.add(skewed(Place::at(0.0, 0.0, 0.0), LEANING));
    let b = graph.add(Node::Sphere { centre: Place::at(14.0, 0.0, 0.0), radius: lit(5.0) });
    let blend = graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: a.into(),
        b: b.into(),
        fillet: Fillet::of(lit(2.0)),
    });
    let doc = Document::new();
    assert!(carries_true_distance(&doc, &graph, a));
    assert!(
        !approximate_blends(&doc, &graph).contains(&blend),
        "a blend of a skewed box and a sphere was marked approximate"
    );
    // The blend itself is a bound by design (a rounded blend understates inside its reach), so the
    // compiler's side of the agreement is asked of the box alone.
    let (alone_doc, alone_graph, _) = one(skewed(Place::at(0.0, 0.0, 0.0), LEANING));
    let field = GraphField::compile_graph(&alone_doc, &alone_graph, 1e-4).expect("compiles");
    assert_eq!(field.normalization(), Normalization::Distance);
}

/// Its Lipschitz rate is one, as every exact distance's is — a bound looser than that makes the
/// tracer step shorter for nothing; one tighter would let it step through a wall.
#[test]
fn a_skewed_box_has_a_rate_of_one() {
    let (doc, graph, id) = one(skewed(Place::at(0.0, 0.0, 0.0), LEANING));
    assert_eq!(graph_lipschitz(&doc, &graph, id), 1.0);
}
