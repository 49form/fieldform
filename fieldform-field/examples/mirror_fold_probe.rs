// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Can a joined mirror ask its body **once** instead of twice?
//!
//! `Node::Mirror { join: true }` is `min(f(p), f(mirror(p)))`, so the whole subtree below it is
//! written out twice — and on the GPU that is what makes a mirrored sweep cost four walks of its
//! spine instead of two, which the driver's compile time is worse than cubic in.
//!
//! When the body lies wholly on one side of the plane, one reading should do: reflect the query
//! onto the body's side and ask there. For a *true* distance that is exact — for any body point
//! `b` and any query `p` on the body's side,
//! `|p − mirror(b)|² − |p − b|² = 4 (p·n − c)(b·n − c) ≥ 0`, because both factors carry the same
//! sign, so the nearer half is always the one the query is already on and the `min` never had
//! anything to choose.
//!
//! A field here need not be a true distance. It may understate ([`Normalization::LowerBound`]),
//! and then the ordering argument does not carry: nothing says an understating field is monotone
//! in how far away the body is. What must never happen is the fold coming out **larger** than the
//! `min`, because that is a claim of clear room that is not there, and a ray given it steps
//! through the surface.
//!
//! So this measures both, over a grid, for bodies of each kind — including ones that understate.
//!
//! Run with: `cargo run --release -p fieldform-field --example mirror_fold_probe`

use fieldform_field::ScalarField;
use fieldform_scene::{
    BooleanOp, Continuity, Curve, CurveId, Document, Fillet, Graph, Node, ParamRef, Place, Turn,
};
use nalgebra::{Point3, Vector3};

fn main() {
    // The mirror plane for every case: through the origin, facing +X, so a body built at
    // positive X lies wholly on one side of it.
    let at = Point3::origin();
    let normal = Vector3::x();

    println!(
        "{:<44} {:>10} {:>12} {:>13} {:>12}",
        "body (all at positive X)", "reads", "worst |diff|", "worst over", "normalized"
    );
    for (label, build) in cases() {
        let (doc, graph) = build();
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 0.5)
            .expect("the body compiles");
        let normalization = format!("{:?}", field.normalization());

        let (lo, hi) = match fieldform_field::part_extent(&doc, &graph) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => panic!("every body here is bounded"),
        };
        // A grid over both halves and well past them, so the fold is asked where it matters:
        // outside, inside, and across the plane it is folding about.
        let span = (hi - lo).norm().max(1.0);
        let centre = nalgebra::center(&lo, &hi);
        let reach = span * 0.9;
        let steps = 48;
        let mut worst_diff: f64 = 0.0;
        let mut worst_over: f64 = 0.0;
        let mut reads = 0usize;
        for i in 0..=steps {
            for j in 0..=steps {
                for k in 0..=steps {
                    let f = |n: usize| (n as f64 / steps as f64 - 0.5) * 2.0 * reach;
                    let p = centre + Vector3::new(f(i), f(j), f(k));
                    let mirrored = reflect(p, at, normal);
                    let both = field.distance(p).min(field.distance(mirrored));
                    // The fold: ask on whichever side the body is, which here is +X.
                    let folded = if (p - at).dot(&normal) < 0.0 { mirrored } else { p };
                    let once = field.distance(folded);
                    reads += 1;
                    worst_diff = worst_diff.max((once - both).abs());
                    // Positive means the fold claims more room than the pair does, which is the
                    // direction that steps a ray through the surface.
                    worst_over = worst_over.max(once - both);
                }
            }
        }
        println!(
            "{label:<44} {reads:>10} {worst_diff:>12.6} {worst_over:>13.6} {normalization:>12}"
        );
    }
    println!(
        "\n`worst over` is the only column that can make this unsafe: it is how much more clear\n\
         room one reading claims than two, and anything above zero is a ray stepping through."
    );
}

fn reflect(p: Point3<f64>, at: Point3<f64>, normal: Vector3<f64>) -> Point3<f64> {
    p - normal * (2.0 * (p - at).dot(&normal))
}

type Build = Box<dyn Fn() -> (Document, Graph)>;

fn cases() -> Vec<(&'static str, Build)> {
    let lit = ParamRef::Literal;
    vec![
        (
            "a sphere",
            Box::new(move || {
                let mut g = Graph::empty();
                g.add(Node::Sphere {
                    centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
                    radius: lit(25.0),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a cuboid, turned",
            Box::new(move || {
                let mut g = Graph::empty();
                g.add(Node::Cuboid { edge: Default::default(),
                    centre: Place::At([lit(70.0), lit(10.0), lit(-5.0)]),
                    size: [lit(40.0), lit(70.0), lit(30.0)],
                    turn: Turn::By([lit(20.0), lit(35.0), lit(0.0)]),
                });
                (Document::new(), g)
            }),
        ),
        (
            "two solids blended at G1",
            Box::new(move || {
                let mut g = Graph::empty();
                let a = g.add(Node::Sphere {
                    centre: Place::At([lit(50.0), lit(0.0), lit(0.0)]),
                    radius: lit(25.0),
                });
                let b = g.add(Node::Cuboid { edge: Default::default(),
                    centre: Place::At([lit(90.0), lit(0.0), lit(0.0)]),
                    size: [lit(40.0); 3],
                    turn: Turn::default(),
                });
                g.add(Node::Blend {
                    op: BooleanOp::Union,
                    a: a.into(),
                    b: b.into(),
                    fillet: Fillet { radius: lit(12.0), continuity: Continuity::G1 },
                });
                (Document::new(), g)
            }),
        ),
        (
            "a profile swept along a bend (understates)",
            Box::new(move || {
                let mut doc = Document::new();
                let profile = CurveId::new();
                doc.curves.insert(
                    profile,
                    Curve::polyline(
                        profile,
                        "section",
                        vec![
                            Point3::new(40.0, -14.0, 0.0),
                            Point3::new(68.0, -14.0, 0.0),
                            Point3::new(68.0, 14.0, 0.0),
                            Point3::new(40.0, 14.0, 0.0),
                        ],
                        true,
                    )
                    .expect("a square"),
                );
                let spine = CurveId::new();
                doc.curves.insert(
                    spine,
                    Curve::new(
                        spine,
                        "spine",
                        3,
                        vec![
                            Point3::new(54.0, 0.0, 0.0),
                            Point3::new(60.0, 0.0, 40.0),
                            Point3::new(90.0, 0.0, 70.0),
                            Point3::new(110.0, 0.0, 95.0),
                        ],
                        vec![1.0; 4],
                        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        false,
                    )
                    .expect("a cubic"),
                );
                let mut g = Graph::empty();
                g.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
                (doc, g)
            }),
        ),
        (
            "a tube infilled with a gyroid (understates)",
            Box::new(move || {
                let mut g = Graph::empty();
                let outer = g.add(Node::Cylinder { edge: Default::default(),
                    ends: fieldform_scene::Ends::At {
                        from: Place::At([lit(70.0), lit(0.0), lit(-40.0)]),
                        to: Place::At([lit(70.0), lit(0.0), lit(40.0)]),
                    },
                    radius: lit(40.0),
                });
                let cell = g.add(Node::Lattice {
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
                    phase: fieldform_scene::LatticePhase::sheet(lit(3.0)),
                    frame: fieldform_scene::Frame::default(),
                });
                g.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![outer.into(), cell.into()] });
                (Document::new(), g)
            }),
        ),
    ]
}
