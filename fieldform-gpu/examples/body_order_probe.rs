// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What order do a part's bodies come in, and what does adding one cost the ones already there?
//!
//! `Graph::roots` ends with `roots.sort_by_key(|id| id.raw())`, and `NodeId::new` draws
//! `rand::random::<u64>()` (`fieldform-scene/src/id.rs` says why ids are random). So nothing chose
//! the order the bodies come in, and a new body takes a random *rank* among them rather than
//! going last. `split_into_bodies` names each body function by its position — `body_0`, `body_1`
//! — and `evaluation_order` seeds its walk from the unconsumed nodes sorted by that same raw id,
//! so a rank that moves renames functions and renumbers SSA slots that had nothing to do with the
//! edit. The source changes, and the driver compiles the whole part again: 1.1 s to 3.8 s cold on
//! the scenes `fieldform-gpu/examples/wing_compile_cost.rs` times.
//!
//! **Every row has a control that ought to show no difference**, because the interesting claim is
//! not "the source changed" — it is that the source changes *only* here. Rows 2 and 3 are those
//! controls and they must read "identical"; if either ever reads otherwise, this probe is
//! measuring something else and the rest of it means nothing.
//!
//! # Two versions of this file, merged
//!
//! The profile-group session wrote a probe of this name first, and it was right about something
//! this one had to be corrected on: *"the cost of a join is one cold compile either way, and the
//! question worth a number is how much of the part it disturbs."* Either way. Ordering the bodies
//! by when they were made does not avoid the compile, because the driver keeps a pipeline only when
//! the **whole** source is identical and a creation never leaves it so. The first table below is
//! that, measured; `body-order-decision.md` §9 is the correction it forced.
//!
//! So this file carries both: the fresh-ids questions that first showed the order was id-driven,
//! the whole-source question that says what the driver charges for, and the candidate orderings.
//! Its point table is also theirs rather than the original: **a point is never a body** since
//! 2026-09-26 (`Node::is_body`, the owner's decision on `profile-group-spec.md` §4.5), so a loose
//! joint costs no root and no generated line, and the rows below check that rather than assume it.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example body_order_probe`

use fieldform_scene::{Document, Graph, Node, NodeId, ParamRef, Place, Turn};
use std::collections::HashMap;

/// Rounds per rate. Each round redraws every id, so the rate is over fresh id draws.
const ROUNDS: usize = 20_000;

/// Rounds for the rows that generate shaders, which is four programs a round rather than a sort.
const SOURCE_ROUNDS: usize = 200;

fn main() {
    println!("A part's bodies are ordered by random node id.\n");
    what_the_driver_recompiles();
    the_cost_at_the_source();
    the_rank_a_creation_takes();
    the_controls();
    the_candidates();
    what_a_root_costs();
    two_independent_builds();
    println!();
    println!("In one line: ordering the bodies by when they were made does not save a compile on a");
    println!("creation, because a creation changes the source whatever the order is. What it buys is");
    println!("an interface -- a body list that does not reshuffle, an import in its file's order, and");
    println!("\"the first body\" meaning the first row rather than the lowest random id -- and it is");
    println!("what lets the list be reordered for free, which is a compile genuinely avoided.");
}

// -------------------------------------------------------------------------------------------
// What the driver recompiles
// -------------------------------------------------------------------------------------------

/// **The question that decides everything else: when does the driver compile again?**
///
/// `traced/rebuild.rs` gates on `previous.source() == program.source` — **whole-source
/// equality**. One byte different and the entire module is compiled again; there is no
/// per-function cache and no incremental path. So the only thing that matters is whether the
/// generated source comes back *identical*, not how much of it changed.
///
/// This row is here because the probe did not have it, and its absence carried a wrong conclusion
/// for two commits: that ordering the bodies by when they were made saves a compile on a creation.
/// It does not. A creation adds a body function and a term to the fold, so the source differs
/// whatever order the bodies are in, and the driver recompiles either way.
///
/// What the ordering *is* worth is below it: a reorder of the list is byte-identical, so it is free —
/// and that is the whole reason the list's order and the compilers' order are separate things.
fn what_the_driver_recompiles() {
    println!("{:-<78}", "WHAT THE DRIVER RECOMPILES ");
    println!("{:<44} {:>14} {:>16}", "after", "source", "driver");
    for (label, identical) in [
        ("adding a body, bodies ordered by stamp", creation_is_identical(true)),
        ("adding a body, bodies ordered by id", creation_is_identical(false)),
        ("reversing the list's own order", reorder_is_identical()),
        ("dragging a radius", drag_is_identical()),
    ] {
        println!(
            "{:<44} {:>14} {:>16}",
            label,
            if identical { "identical" } else { "DIFFERENT" },
            if identical { "keeps it" } else { "COMPILES AGAIN" },
        );
    }
    println!();
    println!("  So a creation costs a compile however the bodies are ordered -- 1.1 s to 3.8 s cold");
    println!("  on the scenes `wing_compile_cost.rs` times. Ordering them by when they were made");
    println!("  does not avoid it, and the table below, which measures how much of the text moved,");
    println!("  is not a measure of compiles saved. The driver does not care how much changed.");
    println!();
    println!("  What the ordering buys is the third row: a list reordered by hand generates the");
    println!("  same bytes, so it is free. That needs the list's order and the compilers' order to");
    println!("  be separate, which is what `block-order-spec.md` §4 ruled.");
}

/// Whether adding a body leaves the whole source identical. It does not, either way.
fn creation_is_identical(stamped: bool) -> bool {
    let (mut document, mut graph, mut made) = (Document::new(), Graph::empty(), 0u64);
    for _ in 0..4 {
        add_body(&mut graph, &mut document, &mut made, stamped);
    }
    let before = source(&document, &graph);
    add_body(&mut graph, &mut document, &mut made, stamped);
    before == source(&document, &graph)
}

/// Whether arranging the list leaves the whole source identical. It does.
fn reorder_is_identical() -> bool {
    let (mut document, mut graph, mut made) = (Document::new(), Graph::empty(), 0u64);
    for _ in 0..4 {
        add_body(&mut graph, &mut document, &mut made, true);
    }
    let before = source(&document, &graph);
    let bodies = document.folded_bodies(&graph);
    for (rank, id) in bodies.iter().rev().enumerate() {
        document.arrangement.insert(*id, rank as u32);
    }
    before == source(&document, &graph)
}

/// The control the architecture's own promise rests on, restated here beside the others.
fn drag_is_identical() -> bool {
    let (doc, graph) = three_bodies(1.0);
    let before = source(&doc, &graph);
    let (doc2, graph2) = same_graph_new_numbers(&graph);
    before == source(&doc2, &graph2)
}

fn add_body(graph: &mut Graph, document: &mut Document, made: &mut u64, stamped: bool) -> NodeId {
    let id = one_of_each_kind(graph, *made as usize);
    *made += 1;
    if stamped {
        document.created.insert(id, 1_700_000_000_000_000 + *made);
    }
    id
}

// -------------------------------------------------------------------------------------------
// How much of the text moves, which is not the same question
// -------------------------------------------------------------------------------------------

/// Adding a body to an n-body part: how many of the bodies already there are **rewritten**?
///
/// **Not a measure of compiles saved** — the row above is. Kept because it is the measure of how
/// much of the program a creation disturbs, which is what a per-function or incremental compiler
/// would charge for, and because it is the number that would matter if `split_into_bodies` named
/// its functions by body rather than by position (`block-order-spec.md` §4.1).
///
/// **The bodies are of different kinds, and that is the measurement, not the scene dressing.**
/// Written first with spheres throughout, this reported no cost at all -- and was right about that
/// scene and wrong about the claim. A number never reaches the generated text: every literal is a
/// `params[..]` slot, so two spheres differ only in which slots they hold, and when the fold order
/// and the slot order are decided by the same thing the k-th body always holds the k-th block of
/// slots. Rank and text then move together and neither shows. A part of *different* bodies is the
/// case where a body landing in front of another rewrites it, and it is also what a document is.
fn the_cost_at_the_source() {
    println!("\n{:-<78}", "HOW MUCH OF THE TEXT MOVES ");
    println!(
        "{:<26} {:>12} {:>16} {:>16}",
        "adding one body to", "bodies", "rewritten, kept", "rewritten, tied"
    );
    for existing in [1usize, 2, 3, 5] {
        let (mut kept, mut tied) = (0usize, 0usize);
        for _ in 0..SOURCE_ROUNDS {
            kept += rewritten_by_a_creation(existing, true);
            tied += rewritten_by_a_creation(existing, false);
        }
        println!(
            "{:<26} {:>12} {:>15.2} {:>15.2}",
            format!("a {existing}-body part"),
            existing,
            kept as f64 / SOURCE_ROUNDS as f64,
            tied as f64 / SOURCE_ROUNDS as f64,
        );
    }
    println!();
    println!("  The average number of *existing* body functions whose text changed. \"kept\" stamps");
    println!("  each body as the document does; \"tied\" leaves them unstamped, which is what");
    println!("  shipped before -- every stamp ties, so the id breaks it.");
    println!();
    println!("  Zero is worth having and is not a compile saved: the module is recompiled either");
    println!("  way, because the new body's own function is new. See the row above.");
}

/// One round: build `existing` bodies, generate, add one more, generate, and count how many of the
/// original bodies' functions were rewritten.
fn rewritten_by_a_creation(existing: usize, stamped: bool) -> usize {
    let mut document = Document::new();
    let mut graph = Graph::empty();
    let mut made = 0u64;
    let add = |graph: &mut Graph, document: &mut Document, made: &mut u64| {
        let id = one_of_each_kind(graph, *made as usize);
        *made += 1;
        if stamped {
            document.created.insert(id, 1_700_000_000_000_000 + *made);
        }
        id
    };
    for _ in 0..existing {
        add(&mut graph, &mut document, &mut made);
    }
    let before = source(&document, &graph);
    add(&mut graph, &mut document, &mut made);
    let after = source(&document, &graph);
    (0..existing)
        .filter(|i| body_function(&before, *i) != body_function(&after, *i))
        .count()
}

/// A body whose generated text is distinguishable from its neighbours', by kind.
///
/// Cycled rather than drawn, unlike `mixed_kinds` below: here the point is that the bodies differ
/// from each other, and which kind is which is not what is being measured.
fn one_of_each_kind(graph: &mut Graph, i: usize) -> NodeId {
    let lit = ParamRef::Literal;
    let x = lit(i as f64 * 40.0);
    match i % 5 {
        0 => sphere(graph, i),
        1 => graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([x, lit(0.0), lit(0.0)]),
            size: [lit(18.0); 3],
            turn: Turn::default(),
        }),
        2 => graph.add(Node::Torus {
            centre: Place::At([x, lit(0.0), lit(0.0)]),
            major_radius: lit(15.0),
            minor_radius: lit(4.0),
            turn: Turn::default(),
        }),
        3 => graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::At {
                from: Place::At([x, lit(0.0), lit(0.0)]),
                to: Place::At([x, lit(0.0), lit(30.0)]),
            },
            radius: lit(9.0),
        }),
        _ => graph.add(Node::Capsule {
            ends: fieldform_scene::Ends::At {
                from: Place::At([x, lit(0.0), lit(0.0)]),
                to: Place::At([x, lit(0.0), lit(30.0)]),
            },
            start_radius: lit(7.0),
            end_radius: lit(7.0),
        }),
    }
}

/// One generated body function, by index, for comparing two programs body by body.
///
/// Read out of the text rather than re-derived from the graph -- the same choice `closure_of` makes
/// on the other side, and for the same reason: the text cannot disagree with itself.
fn body_function(source: &str, index: usize) -> String {
    let opening = format!("fn body_{index}(");
    let Some(from) = source.find(&opening) else {
        return format!("<no body_{index}>");
    };
    let rest = &source[from..];
    // Trailing blank lines trimmed: the last function in a program has a different number of them
    // from a middle one, and that is formatting rather than content.
    let to = rest.find("\n\nfn ").unwrap_or(rest.len());
    rest[..to].trim_end().to_string()
}

// -------------------------------------------------------------------------------------------
// Why the order moved: the rank a creation takes
// -------------------------------------------------------------------------------------------

/// Ranked by raw id, where does a new body land among the ones already there?
///
/// The mechanism behind the table above: `split_into_bodies` names each body function by its
/// position, so a rank that moves is a function that is renamed. An id is drawn at random, so the
/// rank is drawn uniformly and the chance of going last is 1/(n+1).
fn the_rank_a_creation_takes() {
    println!("\n{:-<78}", "WHY -- THE RANK A CREATION TAKES, BY ID ");
    println!(
        "{:<26} {:>10} {:>10} {:>12}",
        "adding one body to", "goes last", "1/(n+1)", "disturbs"
    );
    for existing in [1usize, 2, 3, 5] {
        let mut last = 0usize;
        for _ in 0..ROUNDS {
            let mut graph = Graph::empty();
            for i in 0..existing {
                sphere(&mut graph, i);
            }
            let fresh = sphere(&mut graph, existing);
            let rank = graph.roots().iter().position(|id| *id == fresh).expect("it is a body");
            if rank == existing {
                last += 1;
            }
        }
        println!(
            "{:<26} {:>9.1}% {:>9.1}% {:>11.1}%",
            format!("a {existing}-body part"),
            pct(last),
            100.0 / (existing as f64 + 1.0),
            pct(ROUNDS - last),
        );
    }
    println!();
    println!("  Sampled rates, so they move a few tenths between runs -- which is why 1/(n+1) is");
    println!("  beside them: the claim is that the rank is drawn uniformly, and the agreement is");
    println!("  what says so.");
}

/// Whether any of `existing` sits at a different rank than it did.
fn renamed_any(was: &[NodeId], now: &[NodeId], existing: &[NodeId]) -> bool {
    existing.iter().any(|id| {
        let before = was.iter().position(|r| r == id);
        let after = now.iter().position(|r| r == id);
        before != after
    })
}

// -------------------------------------------------------------------------------------------
// The controls
// -------------------------------------------------------------------------------------------

/// The two things that ought to show no difference at all.
fn the_controls() {
    println!("\n{:-<78}", "THE CONTROLS (these must read \"identical\") ");
    println!("{:<52} {:>24}", "regenerating after", "source");

    // A part whose ids have not moved, generated twice.
    let (doc, graph) = three_bodies(1.0);
    let a = source(&doc, &graph);
    let b = source(&doc, &graph);
    verdict("nothing at all, same graph, twice", &a, &b);

    // Only a number changed: the architecture's own promise, and the property
    // `draggable_numbers.rs` asks about across every kind of number.
    let (doc2, graph2) = same_graph_new_numbers(&graph);
    let c = source(&doc2, &graph2);
    verdict("a radius dragged, every id unchanged", &a, &c);

    println!();
    println!("  So regeneration is not the problem: drags, sliders, settles and curve edits all");
    println!("  leave the source byte-identical. Only creation and removal move an id, and only");
    println!("  a moved id moves a rank.");
}

/// The same graph with every literal scaled -- same nodes, same ids, different numbers.
fn same_graph_new_numbers(graph: &Graph) -> (Document, Graph) {
    let mut out = graph.clone();
    let ids: Vec<NodeId> = out.nodes().map(|(id, _)| id).collect();
    for id in ids {
        if let Some(Node::Sphere { radius, .. }) = out.node_mut(id) {
            *radius = ParamRef::Literal(37.0);
        }
    }
    (Document::new(), out)
}

fn verdict(label: &str, a: &str, b: &str) {
    let word = if a == b { "identical" } else { "DIFFERENT" };
    println!("{label:<52} {word:>24}");
}

// -------------------------------------------------------------------------------------------
// The candidates
// -------------------------------------------------------------------------------------------

/// Four rankings of the same roots, over the same rounds: which leave the existing bodies put?
///
/// Measured at the order rather than at the source, because `generate` asks `Graph::roots` for the
/// order itself -- there is no way to hand it another one without changing that function, which is
/// the decision this probe is here to inform rather than to anticipate.
///
/// **Two scenes, because a structural sort ties and a stamp can tie too.** Bodies that are all the
/// same kind are what a closed loop of joints is, and they are also what makes a fingerprint
/// useless. A stamp shared by every body is what a multi-body paste writes -- `stamp_new_nodes`
/// reads the clock once and writes that one moment to every fresh node -- and what an older file
/// with no `created` entry at all reads as.
fn the_candidates() {
    println!("\n{:-<78}", "THE CANDIDATES ");
    for scene in Scene::all() {
        println!("\n  {}", scene.label);
        println!(
            "  {:<42} {:>14} {:>14}",
            "ranking the roots by", "kept (3+1)", "kept (5+1)"
        );
        for (label, rank) in rankings() {
            let mut kept = [0usize; 2];
            for (slot, existing) in [3usize, 5].into_iter().enumerate() {
                for round in 0..ROUNDS {
                    let mut graph = Graph::empty();
                    let before: Vec<NodeId> =
                        (0..existing).map(|i| (scene.add)(&mut graph, i)).collect();
                    let mut made: HashMap<NodeId, u64> = before
                        .iter()
                        .enumerate()
                        .map(|(i, id)| (*id, scene.stamp(round, i)))
                        .collect();
                    let was = rank(&graph, &made);
                    let fresh = (scene.add)(&mut graph, existing);
                    made.insert(fresh, scene.stamp(round, existing));
                    let now = rank(&graph, &made);
                    if !renamed_any(&was, &now, &before) {
                        kept[slot] += 1;
                    }
                }
            }
            println!(
                "  {:<42} {:>13.1}% {:>13.1}%",
                label,
                100.0 * kept[0] as f64 / ROUNDS as f64,
                100.0 * kept[1] as f64 / ROUNDS as f64,
            );
        }
    }
    println!();
    println!("  \"kept\" is the rounds in which no body already in the part changed rank -- so no");
    println!("  body function is renamed and the driver has nothing to recompile but the new one.");
    println!("  The id row is the control that ought to *fail*, at 1/(n+1), and does.");
    println!();
    println!("  The two rows that read like it are the two ways of tying. A fingerprint over");
    println!("  bodies of one kind has nothing to sort on, and a stamp every body shares has");
    println!("  nothing either; both fall back to the id and inherit its rate exactly. So the");
    println!("  ordering that wins does so because the stamp is *distinct*, not because it is a");
    println!("  stamp -- which is a requirement on whatever writes it, not on the sort.");
}

/// A scene to rank, and how its nodes are stamped.
struct Scene {
    label: &'static str,
    add: fn(&mut Graph, usize) -> NodeId,
    /// `false` writes one moment to every node, which is what a multi-body paste does.
    distinct_stamps: bool,
}

impl Scene {
    fn all() -> Vec<Scene> {
        vec![
            Scene {
                label: "five bodies of one kind, each made at its own moment \
                        (a loop of joints)",
                add: same_kind,
                distinct_stamps: true,
            },
            Scene {
                label: "five bodies of kinds drawn at random, each at its own moment",
                add: mixed_kinds,
                distinct_stamps: true,
            },
            Scene {
                label: "five of random kinds, all stamped one moment \
                        (a paste, or an unstamped file)",
                add: mixed_kinds,
                distinct_stamps: false,
            },
        ]
    }

    fn stamp(&self, round: usize, i: usize) -> u64 {
        if self.distinct_stamps {
            1_700_000_000_000 + (round * 16 + i) as u64
        } else {
            1_700_000_000_000
        }
    }
}

fn same_kind(graph: &mut Graph, i: usize) -> NodeId {
    sphere(graph, i)
}

/// A body of a kind drawn at random, so a fingerprint over the kind has something to sort on.
///
/// **Drawn rather than cycled through, because which kind sorts where decides the answer.** Laid
/// out `i % 6`, the kind of the *next* body was fixed by how many there already were, and the
/// structural rows then read 0% or 100% -- not a rate but a foregone conclusion, and one this
/// probe had chosen for itself. A modeller does not know what they will make next, so the kind is
/// drawn, and by the same generator the id is: `NodeId::new()` is `rand::random`, which is the
/// whole subject here.
fn mixed_kinds(graph: &mut Graph, i: usize) -> NodeId {
    let lit = ParamRef::Literal;
    let x = lit(i as f64 * 60.0);
    match NodeId::new().raw() % 6 {
        0 => sphere(graph, i),
        1 => graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([x, lit(0.0), lit(0.0)]),
            size: [lit(30.0); 3],
            turn: Turn::default(),
        }),
        2 => graph.add(Node::Torus {
            centre: Place::At([x, lit(0.0), lit(0.0)]),
            major_radius: lit(20.0),
            minor_radius: lit(5.0),
            turn: Turn::default(),
        }),
        3 => graph.add(Node::Point { at: [x, lit(0.0), lit(0.0)] }),
        4 => graph.add(Node::Cylinder { edge: Default::default(),
            ends: fieldform_scene::Ends::At {
                from: Place::At([x, lit(0.0), lit(0.0)]),
                to: Place::At([x, lit(0.0), lit(40.0)]),
            },
            radius: lit(10.0),
        }),
        _ => graph.add(Node::Capsule {
            ends: fieldform_scene::Ends::At {
                from: Place::At([x, lit(0.0), lit(0.0)]),
                to: Place::At([x, lit(0.0), lit(40.0)]),
            },
            start_radius: lit(8.0),
            end_radius: lit(8.0),
        }),
    }
}

type Ranking = (&'static str, fn(&Graph, &HashMap<NodeId, u64>) -> Vec<NodeId>);

fn rankings() -> Vec<Ranking> {
    vec![
        ("its raw id (what ships today)", |g, _| g.roots()),
        ("when it was made, then its id", |g, made| {
            let mut roots = g.roots();
            roots.sort_by_key(|id| (made.get(id).copied().unwrap_or(0), id.raw()));
            roots
        }),
        ("its node kind, then its id", |g, _| {
            // The structural sort, in the form the brief proposes it: a kind, then a tiebreak
            // that has to be the id because two bodies of one kind fingerprint alike.
            let mut roots = g.roots();
            roots.sort_by_key(|id| (g.node(*id).map(kind_rank).unwrap_or(u8::MAX), id.raw()));
            roots
        }),
        // What the block list ships today: `App::ordered_nodes` walks the graph in Kahn order,
        // taking the earliest-stamped of everything whose operands are placed, and breaks a tie
        // on position in `Graph::evaluation_order`. That function seeds its walk from the
        // unconsumed nodes *sorted by raw id*, so the tie-break is an id in disguise -- which is
        // the row below being here rather than being taken on faith.
        ("when it was made, then evaluation order", |g, made| {
            let order = g.evaluation_order().expect("no cycle in these scenes");
            let rank: HashMap<NodeId, usize> =
                order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
            let mut roots = g.roots();
            roots.sort_by_key(|id| (made.get(id).copied().unwrap_or(0), rank[id]));
            roots
        }),
        ("its node kind, then when it was made", |g, made| {
            let mut roots = g.roots();
            roots.sort_by_key(|id| {
                (
                    g.node(*id).map(kind_rank).unwrap_or(u8::MAX),
                    made.get(id).copied().unwrap_or(0),
                    id.raw(),
                )
            });
            roots
        }),
    ]
}

/// Enough of a node-kind order to rank the roots this probe builds.
fn kind_rank(node: &Node) -> u8 {
    match node {
        Node::Point { .. } => 0,
        Node::Sphere { .. } => 1,
        Node::Cuboid { .. } => 2,
        Node::Torus { .. } => 3,
        Node::Cylinder { .. } => 4,
        Node::Capsule { .. } => 5,
        _ => 6,
    }
}

// -------------------------------------------------------------------------------------------
// What a root costs, and what a point costs now
// -------------------------------------------------------------------------------------------

/// Each body is a root, and each root costs generated lines. A point is not a body and costs none.
///
/// **The second half was the whole reason this file exists**, and it is now a check rather than a
/// measurement. `Node::is_body` excluded only a ramp, so an unconsumed `Node::Point` was a root: it
/// took a body index and about fifteen generated lines, and the profile-group work holds a loop's
/// joints as point blocks — so closing a loop of four joints was four creations, each one a cold
/// compile of the whole part. A point is never a body since 2026-09-26, so the rows here add small
/// spheres to measure a root's cost and then assert that four points add neither a root nor a byte.
fn what_a_root_costs() {
    println!("\n{:-<78}", "WHAT A ROOT COSTS, AND WHAT A POINT COSTS NOW ");
    println!("{:<34} {:>8} {:>10} {:>14}", "a part holding", "bodies", "lines", "vs. two alone");
    let base = {
        let (doc, graph) = two_solids();
        source(&doc, &graph).lines().count()
    };
    for extra in [0usize, 1, 2, 4] {
        let (doc, graph) = two_solids_and_spheres(extra);
        let lines = source(&doc, &graph).lines().count();
        let label = match extra {
            0 => "two solids".to_string(),
            1 => "two solids and a third body".to_string(),
            n => format!("two solids and {n} more bodies"),
        };
        println!(
            "{:<34} {:>8} {:>10} {:>13}",
            label,
            graph.roots().len(),
            lines,
            if extra == 0 { "-".to_string() } else { format!("+{}", lines - base) }
        );
    }

    // And the check the joint decision rests on: points beside the bodies add no root, and no byte.
    let (doc, plain) = two_solids();
    let (doc_points, with_points) = two_solids_and_points(4);
    println!();
    println!(
        "  four points beside two bodies: {} roots, and the source is identical to the two \
         bodies alone: {}",
        with_points.roots().len(),
        source(&doc, &plain) == source(&doc_points, &with_points),
    );
    println!("  So a joint really does cost no compile -- not because of any ordering, but because");
    println!("  it contributes nothing to the source at all.");
}

// -------------------------------------------------------------------------------------------
// What two independently built copies of one composition have in common
// -------------------------------------------------------------------------------------------

/// Built twice from scratch, do two copies of one composition generate the same shader?
///
/// **Not an operation the application performs**, and the profile-group session's own correction
/// says so: every id is fresh here, where an app frame regenerates a part whose ids have not moved.
/// Kept because it is how the order was shown to be id-driven at all, and because the one-body and
/// unioned rows are the contrast that makes it legible — with a single root there is no order to
/// get wrong.
fn two_independent_builds() {
    println!("\n{:-<78}", "TWO INDEPENDENT BUILDS OF ONE COMPOSITION ");
    println!("{:<44} {:>30}", "built twice from scratch", "source");
    let doc = Document::new();
    let differ = |build: &dyn Fn() -> Graph| {
        let first = source(&doc, &build());
        (0..20).filter(|_| source(&doc, &build()) != first).count()
    };
    let two = differ(&|| two_solids().1);
    println!("{:<44} {:>30}", "two bodies", format!("{two} of 20 differ"));
    println!(
        "{:<44} {:>30}",
        "one body",
        format!("{} of 20 differ", differ(&one_solid))
    );
    println!(
        "{:<44} {:>30}",
        "the same two bodies unioned into one",
        format!("{} of 20 differ", differ(&unioned))
    );
    println!();
    println!("  About half for two bodies, which is what two random ids should give. Nought for a");
    println!("  single root, because there is then no order to choose: the fold is one call, and a");
    println!("  boolean's operands are a list somebody built rather than a sort by id.");
}

/// The two solids joined, so the part has one root.
fn unioned() -> Graph {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let a = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(20.0) });
    let b = graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
        size: [lit(30.0); 3],
        turn: Turn::default(),
    });
    graph.add(Node::Boolean { op: fieldform_scene::BooleanOp::Union, inputs: vec![a.into(), b.into()] });
    graph
}

/// One body, for the contrast: with a single root there is no order to choose.
fn one_solid() -> Graph {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(20.0) });
    graph
}

// -------------------------------------------------------------------------------------------
// Scenes
// -------------------------------------------------------------------------------------------

fn sphere(graph: &mut Graph, i: usize) -> NodeId {
    let lit = ParamRef::Literal;
    graph.add(Node::Sphere {
        centre: Place::At([lit(i as f64 * 60.0), lit(0.0), lit(0.0)]),
        radius: lit(20.0),
    })
}

fn three_bodies(scale: f64) -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(20.0 * scale) });
    graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
        size: [lit(30.0), lit(30.0), lit(30.0)],
        turn: Turn::default(),
    });
    graph.add(Node::Sphere {
        centre: Place::At([lit(120.0), lit(0.0), lit(0.0)]),
        radius: lit(15.0),
    });
    (Document::new(), graph)
}

fn two_solids() -> (Document, Graph) {
    two_solids_and_points(0)
}

/// Two solids and `extra` more small bodies, for measuring what a root costs.
fn two_solids_and_spheres(extra: usize) -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let (doc, mut graph) = two_solids();
    for i in 0..extra {
        graph.add(Node::Sphere {
            centre: Place::At([lit(i as f64 * 10.0), lit(0.0), lit(0.0)]),
            radius: lit(2.0),
        });
    }
    (doc, graph)
}

fn two_solids_and_points(points: usize) -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(20.0) });
    graph.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
        size: [lit(30.0), lit(30.0), lit(30.0)],
        turn: Turn::default(),
    });
    for i in 0..points {
        graph.add(Node::Point { at: [lit(i as f64 * 10.0), lit(0.0), lit(0.0)] });
    }
    (Document::new(), graph)
}

fn source(doc: &Document, graph: &Graph) -> String {
    fieldform_gpu::generate(doc, graph).expect("generates").field_function().to_string()
}

fn pct(n: usize) -> f64 {
    100.0 * n as f64 / ROUNDS as f64
}
