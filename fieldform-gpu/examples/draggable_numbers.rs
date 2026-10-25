// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which numbers can be dragged without rebuilding the shader?
//!
//! The architecture's rule is that every number in a generated shader is a `params[..]` read, so
//! changing one costs a buffer write rather than a recompile. Where that holds, a number can be
//! scrubbed at sixty frames a second for nothing. Where it does not, the source changes, the
//! driver compiles it again, and today that is measured in seconds
//! (`fieldform-gpu/examples/wing_compile_cost.rs`).
//!
//! So this drags each kind of number a little and asks whether the shader's *source* moved.
//!
//! **It is also the list a generation cache would need.** Skipping `generate` when nothing
//! structural has changed means keying on the numbers that *are* structural, and getting that
//! key wrong is a stale shape with no error to show for it.
//!
//! **Read a row as being about one movement, not about the field.** A row says what happened
//! when *that* number moved *that* far in *that* document -- which is not the same as "this kind
//! of number is safe", and the mirror below is the proof. A mirror's plane is an ordinary
//! literal, and moving it is a buffer write right up until it crosses the body's own extent,
//! where `mirror_one_side` stops returning `None` and the shader folds the point instead of
//! taking a minimum of two readings. Same field, same kind of edit, opposite answers. So a key
//! cannot be built by reading the "a buffer write" column and ignoring those fields; what the
//! column rules out is the *number* reaching the text, not a predicate over it doing so.
//!
//! The first two mirror rows written here both said "a buffer write" and both were wrong to:
//! they moved the plane from one crossing position to another crossing position, so nothing
//! transitioned. **A case for a value-dependent branch has to cross the branch.**
//!
//! Run with: `cargo run --release -p fieldform-gpu --example draggable_numbers`

use fieldform_scene::{Document, Graph, Node, ParamRef, Place, Turn};

fn main() {
    println!("{:<44} {:>12}", "dragging this", "shader");
    for (label, build) in cases() {
        let at = |v: f64| {
            let (doc, graph) = build(v);
            fieldform_gpu::generate(&doc, &graph).map(|p| p.source)
        };
        // Three outcomes, not two: a number can also take a document from generating to being
        // refused, and that is worth telling apart from a rebuild -- it is a message rather than
        // a compile, and the viewport keeps the last good picture. This used to panic.
        let (one, other) = (at(1.0), at(1.7));
        let verdict = match (&one, &other) {
            (Ok(a), Ok(b)) if a == b => "a buffer write".to_string(),
            (Ok(_), Ok(_)) => "REBUILT".to_string(),
            (Ok(_), Err(e)) | (Err(e), Ok(_)) => format!("refused: {e}"),
            (Err(_), Err(_)) => "refused both ways".to_string(),
        };
        println!("{label:<44} {verdict:>12}");
        // `FIELDFORM_DIFF=1` says *where* a REBUILT row's source changed, which is the first
        // thing to ask of a row that was expected to read "a buffer write".
        if let (Ok(a), Ok(b)) = (&one, &other)
            && a != b
            && std::env::var("FIELDFORM_DIFF").is_ok()
        {
            let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
            let differing = la.iter().zip(&lb).filter(|(x, y)| x != y).count();
            println!("    {} lines vs {}, {differing} differ", la.len(), lb.len());
            for (i, (x, y)) in la.iter().zip(&lb).enumerate().filter(|(_, (x, y))| x != y).take(3) {
                println!("    line {}:
      {x}
      {y}", i + 1);
            }
        }
    }
    println!(
        "
A number that rebuilds the shader cannot be scrubbed: every frame of the drag is a
         cold driver compile -- and it is a number any cache over `generate` has to key on."
    );
}

type Build = Box<dyn Fn(f64) -> (Document, Graph)>;

fn cases() -> Vec<(&'static str, Build)> {
    fn bar(g: &mut Graph) -> fieldform_scene::NodeId {
        let lit = ParamRef::Literal;
        g.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(40.0), lit(30.0), lit(120.0)],
            turn: Turn::default(),
        })
    }
    let lit = ParamRef::Literal;
    vec![
        (
            "a cuboid's size",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                g.add(Node::Cuboid { edge: Default::default(),
                    centre: Place::At([lit(0.0); 3]),
                    size: [lit(40.0 * v), lit(30.0), lit(120.0)],
                    turn: Turn::default(),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a sphere's radius",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                g.add(Node::Sphere {
                    centre: Place::At([lit(0.0); 3]),
                    radius: lit(20.0 * v),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a blend's fillet radius",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let a = g.add(Node::Sphere {
                    centre: Place::At([lit(-20.0), lit(0.0), lit(0.0)]),
                    radius: lit(25.0),
                });
                let b = bar(&mut g);
                g.add(Node::Blend {
                    op: fieldform_scene::BooleanOp::Union,
                    a: a.into(),
                    b: b.into(),
                    fillet: fieldform_scene::Fillet {
                        radius: lit(10.0 * v),
                        continuity: fieldform_scene::Continuity::G1,
                        second: None,
                    },
                });
                (Document::new(), g)
            }),
        ),
        (
            "a twist's rate (degrees per mm)",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let body = bar(&mut g);
                g.add(Node::Twist {
                    input: body.into(),
                    degrees_per_mm: lit(0.5 * v),
                    frame: fieldform_scene::Frame::default(),
                });
                (Document::new(), g)
            }),
        ),
        (
            "an offset's distance",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let body = bar(&mut g);
                g.add(Node::Offset { input: body.into(), distance: lit(3.0 * v) });
                (Document::new(), g)
            }),
        ),
        (
            "a lattice's wall thickness",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                g.add(Node::Lattice {
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0) },
                    phase: fieldform_scene::LatticePhase::Sheet { thickness: lit(2.0 * v), offset: lit(0.0) },
                    frame: fieldform_scene::Frame::default(),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a curve's control point (what a drag moves)",
            Box::new(move |v: f64| {
                let mut doc = Document::new();
                let mut g = Graph::empty();
                let id = fieldform_scene::CurveId::new();
                doc.curves.insert(
                    id,
                    fieldform_scene::Curve::polyline(
                        id,
                        "section",
                        vec![
                            nalgebra::Point3::new(0.0, -15.0, 0.0),
                            nalgebra::Point3::new(30.0 * v, -15.0, 0.0),
                            nalgebra::Point3::new(30.0, 15.0, 0.0),
                            nalgebra::Point3::new(0.0, 15.0, 0.0),
                        ],
                        true,
                    )
                    .expect("a rectangle"),
                );
                g.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(id),
                    sweep: fieldform_scene::Sweep::Normal,
                    height: lit(40.0),
                    start: lit(0.0),
                    sides: fieldform_scene::Sides::One,
                    draft: lit(0.0),
                    corners: Default::default(),
                });
                (doc, g)
            }),
        ),
        (
            "a pattern's step (its prune sphere is derived)",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let body = bar(&mut g);
                g.add(Node::Pattern {
                    options: Default::default(),
                    input: body.into(),
                    count: ParamRef::Literal(6.0),
                    kind: fieldform_scene::PatternKind::Linear {
                        step: [lit(60.0 * v), lit(0.0), lit(0.0)],
                    },
                    join: true,
                    fillet: fieldform_scene::Fillet::hard(),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a pattern's count",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let body = bar(&mut g);
                g.add(Node::Pattern {
                    options: Default::default(),
                    input: body.into(),
                    count: ParamRef::Literal((6.0 * v).trunc()),
                    kind: fieldform_scene::PatternKind::Linear {
                        step: [lit(60.0), lit(0.0), lit(0.0)],
                    },
                    join: true,
                    fillet: fieldform_scene::Fillet::hard(),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a morphed operand's size (its divisor is derived)",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let a = g.add(Node::Sphere {
                    centre: Place::At([lit(0.0); 3]),
                    radius: lit(25.0 * v),
                });
                let b = bar(&mut g);
                g.add(Node::Morph { a: a.into(), b: b.into(), t: lit(0.5) });
                (Document::new(), g)
            }),
        ),
        (
            "a lattice's cell size",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                g.add(Node::Lattice {
                    cell: fieldform_scene::LatticeCell::Gyroid,
                    map: fieldform_scene::CellMap::Rectangular { cell: lit(20.0 * v) },
                    phase: fieldform_scene::LatticePhase::Sheet {
                        thickness: lit(2.0),
                        offset: lit(0.0),
                    },
                    frame: fieldform_scene::Frame::default(),
                });
                (Document::new(), g)
            }),
        ),
        (
            "a body's position (its guard box is derived)",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let a = g.add(Node::Pipe {
                    curve: fieldform_scene::CurveId::new(),
                    radius: lit(5.0),
                });
                let b = g.add(Node::Sphere {
                    centre: Place::At([lit(200.0 * v), lit(0.0), lit(0.0)]),
                    radius: lit(25.0),
                });
                g.add(Node::Boolean {
                    op: fieldform_scene::BooleanOp::Union,
                    inputs: vec![a.into(), b.into()],
                });
                (Document::new(), g)
            }),
        ),
        (
            "a swept profile's control point",
            Box::new(move |v: f64| {
                let (mut doc, mut g) = (Document::new(), Graph::empty());
                let profile = rectangle(&mut doc, 30.0 * v, 4);
                let spine = polyline(
                    &mut doc,
                    vec![
                        nalgebra::Point3::new(0.0, 0.0, 0.0),
                        nalgebra::Point3::new(0.0, 0.0, 60.0),
                        nalgebra::Point3::new(20.0, 0.0, 120.0),
                    ],
                    false,
                );
                g.add(Node::Sweep { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(profile),
                    spine,
                    section: fieldform_scene::Section::Skewed,
                    law: Default::default(),
                });
                (doc, g)
            }),
        ),
        (
            "a swept spine's control point",
            Box::new(move |v: f64| {
                let (mut doc, mut g) = (Document::new(), Graph::empty());
                let profile = rectangle(&mut doc, 30.0, 4);
                let spine = polyline(
                    &mut doc,
                    vec![
                        nalgebra::Point3::new(0.0, 0.0, 0.0),
                        nalgebra::Point3::new(0.0, 0.0, 60.0),
                        nalgebra::Point3::new(20.0 * v, 0.0, 120.0),
                    ],
                    false,
                );
                g.add(Node::Sweep { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(profile),
                    spine,
                    section: fieldform_scene::Section::Skewed,
                    law: Default::default(),
                });
                (doc, g)
            }),
        ),
        (
            "a profile's number of control points",
            Box::new(move |v: f64| {
                let (mut doc, mut g) = (Document::new(), Graph::empty());
                let profile = rectangle(&mut doc, 30.0, if v > 1.5 { 7 } else { 4 });
                g.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(profile),
                    sweep: fieldform_scene::Sweep::Normal,
                    height: lit(40.0),
                    start: lit(0.0),
                    sides: fieldform_scene::Sides::One,
                    draft: lit(0.0),
                    corners: Default::default(),
                });
                (doc, g)
            }),
        ),
        // **A held loop's corner, which is the real case the three above stand in for.** Two
        // open curves meeting at both ends, each meeting's two ends sitting on a Point block
        // (`profile-group-spec.md` §4), and an extrude reading the loop. A drag on a held
        // corner moves the block's literals and both ends follow, so all three move here.
        //
        // Built once and *mutated* per value, unlike the cases above. When this row was
        // written the graph had three roots -- two points and the extrude -- emitted in id
        // order, and fresh random ids per build read as REBUILT for a drag that changed no
        // structure. A point is no longer a body (`Node::is_body`), so the extrude is the only
        // root now; the fixed ids stay, because they are what the application has across a drag.
        (
            "a held loop's corner (its point and both ends)",
            {
                let (mut doc, mut g) = (Document::new(), Graph::empty());
                let up = polyline(
                    &mut doc,
                    vec![
                        nalgebra::Point3::new(0.0, -15.0, 0.0),
                        nalgebra::Point3::new(30.0, -15.0, 0.0),
                        nalgebra::Point3::new(30.0, 15.0, 0.0),
                    ],
                    false,
                );
                let back = polyline(
                    &mut doc,
                    vec![
                        nalgebra::Point3::new(30.0, 15.0, 0.0),
                        nalgebra::Point3::new(0.0, 15.0, 0.0),
                        nalgebra::Point3::new(0.0, -15.0, 0.0),
                    ],
                    false,
                );
                let held = g.add(Node::Point { at: [lit(30.0), lit(15.0), lit(0.0)] });
                let dragged = g.add(Node::Point { at: [lit(0.0), lit(-15.0), lit(0.0)] });
                doc.curve_mut(up).expect("up").place_on(2, Some(held));
                doc.curve_mut(back).expect("back").place_on(0, Some(held));
                doc.curve_mut(up).expect("up").place_on(0, Some(dragged));
                doc.curve_mut(back).expect("back").place_on(2, Some(dragged));
                g.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(up),
                    sweep: fieldform_scene::Sweep::Normal,
                    height: lit(40.0),
                    start: lit(0.0),
                    sides: fieldform_scene::Sides::One,
                    draft: lit(0.0),
                    corners: Default::default(),
                });
                Box::new(move |v: f64| {
                    let (mut doc, mut g) = (doc.clone(), g.clone());
                    let to = nalgebra::Point3::new(30.0 * (v - 1.0), -15.0, 0.0);
                    if let Some(Node::Point { at }) = g.node_mut(dragged) {
                        *at = [lit(to.x), lit(to.y), lit(to.z)];
                    }
                    doc.curve_mut(up).expect("up").set_control_point(0, to);
                    doc.curve_mut(back).expect("back").set_control_point(2, to);
                    (doc, g)
                }) as Build
            },
        ),
        // **A profile bent out of its own plane**, which the profile-group session expects to be
        // refused rather than emitted differently -- `Profile::in_plane` is a predicate over
        // numbers, like the mirror above, but the arm it guards returns an error instead of a
        // second form. Worth a row precisely because the expectation is that it reads
        // differently from every other row here.
        (
            "a profile lifted out of plane",
            Box::new(move |v: f64| {
                let (mut doc, mut g) = (Document::new(), Graph::empty());
                let id = fieldform_scene::CurveId::new();
                let lift = (v - 1.0) * 40.0;
                doc.curves.insert(
                    id,
                    fieldform_scene::Curve::polyline(
                        id,
                        "section",
                        vec![
                            nalgebra::Point3::new(0.0, -15.0, 0.0),
                            nalgebra::Point3::new(30.0, -15.0, 0.0),
                            nalgebra::Point3::new(30.0, 15.0, lift),
                            nalgebra::Point3::new(0.0, 15.0, 0.0),
                        ],
                        true,
                    )
                    .expect("a quad"),
                );
                g.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(id),
                    sweep: fieldform_scene::Sweep::Normal,
                    height: lit(40.0),
                    start: lit(0.0),
                    sides: fieldform_scene::Sides::One,
                    draft: lit(0.0),
                    corners: Default::default(),
                });
                (doc, g)
            }),
        ),
        // **A mirror's plane, which is the one that is not a buffer write.** `mirror_one_side`
        // asks whether the whole input lies on one side of the plane, and when it does the
        // shader folds the point instead of taking a minimum of two readings -- a different
        // form, not a different number. So the plane's own origin decides which text is
        // written, and dragging it across the body's extent recompiles.
        //
        // Found by a design panel reading `codegen.rs` after this table already read "a buffer
        // write" all the way down, which it had no business doing: the table had no mirror in
        // it. The bar spans x = 20 to x = 60, so the plane starts clear of it at x = 15 and
        // ends inside it at x = 25.5 -- the case has to CROSS the boundary, which the first
        // pair of cases written here did not, and they reported "a buffer write" for it.
        (
            "a mirror's plane crossing its body",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let body = bar(&mut g);
                g.add(Node::Mirror {
                    input: body.into(),
                    frame: fieldform_scene::Frame::At {
                        origin: [lit(15.0 * v), lit(0.0), lit(0.0)],
                        axis: [lit(1.0), lit(0.0), lit(0.0)],
                    },
                    join: true,
                });
                (Document::new(), g)
            }),
        ),
        // And the same mirror moved where it changes nothing but the number: both values leave
        // the whole bar on one side, so the fold stands and only the plane's position moves.
        (
            "a mirror's plane clear of its body",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let body = bar(&mut g);
                g.add(Node::Mirror {
                    input: body.into(),
                    frame: fieldform_scene::Frame::At {
                        origin: [lit(-100.0 * v), lit(0.0), lit(0.0)],
                        axis: [lit(1.0), lit(0.0), lit(0.0)],
                    },
                    join: true,
                });
                (Document::new(), g)
            }),
        ),
        // **The control.** A change that must rebuild, so a table reading "a buffer write" all
        // the way down is a reading rather than a broken probe. Which boolean a blend is doing
        // picks which of three forms is written, and no slot can stand in for that.
        (
            "CONTROL: a blend's boolean (must rebuild)",
            Box::new(move |v: f64| {
                let mut g = Graph::empty();
                let a = g.add(Node::Sphere {
                    centre: Place::At([lit(-20.0), lit(0.0), lit(0.0)]),
                    radius: lit(25.0),
                });
                let b = bar(&mut g);
                let op = if v > 1.5 {
                    fieldform_scene::BooleanOp::Difference
                } else {
                    fieldform_scene::BooleanOp::Union
                };
                g.add(Node::Blend {
                    op,
                    a: a.into(),
                    b: b.into(),
                    fillet: fieldform_scene::Fillet {
                        radius: lit(10.0),
                        continuity: fieldform_scene::Continuity::G1,
                        second: None,
                    },
                });
                (Document::new(), g)
            }),
        ),
    ]
}

/// A closed loop of `sides` points on a circle of radius `across`, so a case can move a point
/// without changing how many there are -- or change how many without moving one.
fn polyline(
    doc: &mut Document,
    points: Vec<nalgebra::Point3<f64>>,
    closed: bool,
) -> fieldform_scene::CurveId {
    let id = fieldform_scene::CurveId::new();
    doc.curves.insert(
        id,
        fieldform_scene::Curve::polyline(id, "curve", points, closed).expect("a curve"),
    );
    id
}

fn rectangle(doc: &mut Document, across: f64, sides: usize) -> fieldform_scene::CurveId {
    let points = (0..sides)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / sides as f64;
            nalgebra::Point3::new(across * a.cos(), across * a.sin(), 0.0)
        })
        .collect();
    polyline(doc, points, true)
}
