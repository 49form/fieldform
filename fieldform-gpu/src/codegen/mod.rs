// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Turning a composition graph into a shader that evaluates it.
//!
//! # Why generate a shader rather than interpret a tape
//!
//! Fidget can already evaluate a closed-form tree on the GPU, and `examples/gpu_lattice.rs`
//! shows a lattice rendering that way at interactive rates. It cannot evaluate a *cage*: a cage
//! is a baked 3D texture, and Fidget's opcode set is a closed vocabulary of arithmetic with no
//! texture fetch and no extension point. Adding one means editing files under `vendor/fidget`,
//! where MPL-2.0 is file-level copyleft — which D6 forbids as a business rule.
//!
//! Emitting WGSL directly sidesteps that entirely, and is less code than plumbing a new opcode
//! through someone else's tape interpreter, three evaluators, bytecode serialiser and
//! simplifier would have been. The generated function is a plain transcription of the graph;
//! everything it calls is hand-written in `prelude.wgsl`.
//!
//! # What is generated and what is not
//!
//! Only the graph's *shape* is compiled in. Every number — a pipe's radius, a blend's radius —
//! becomes an index into a buffer, **including literals**:
//!
//! ```text
//!   let n1 = pipe_distance(0u, p, params[0]);   // not  pipe_distance(0u, p, 5.0)
//! ```
//!
//! That is the point of the exercise. Changing a radius writes four bytes and redraws: no
//! cage is retessellated, no field is recompiled, no shader is rebuilt. Inlining literals
//! would have made dragging a slider on an unnamed value a recompile, which is precisely the
//! interaction the parametric layer exists to make cheap. Only editing the graph's structure
//! produces a new program.

use fieldform_scene::{
    BooleanOp, CageId, CurveId, Graph, LatticePhase, Node, NodeId, Operand, ParamRef,
};
use std::collections::{HashMap, HashSet};

/// The hand-written runtime every generated program is built on: bindings, and the helpers
/// the generated function calls.
pub const PRELUDE: &str = include_str!("../prelude.wgsl");

/// The start and end of the prelude's profile-solid region, every `fn` of which is instantiated
/// again as a `_bake` copy for the normal pass. The marker lines say why.
const BAKE_REGION_START: &str = "// ---- profile solids: instantiated again for the normal pass ----";
const BAKE_REGION_END: &str = "// ---- end of profile solids ----";

/// The prelude as the shader gets it: the file, and after it the `_bake` copy of every function
/// in the profile-solid region, renamed and reading sections through `profile_distance_bake`.
///
/// **Generated rather than written twice**, because two copies of forty functions kept by hand
/// are the drift D9 is about: the copy is the original with its names changed and nothing else,
/// made once per process. `bake_copy_names` is what the normal pass renames its calls by.
pub fn prelude() -> &'static str {
    static INSTANTIATED: std::sync::OnceLock<(String, Vec<String>)> = std::sync::OnceLock::new();
    &INSTANTIATED.get_or_init(instantiate_bake_copies).0
}

/// The names of the functions the normal pass reads through their `_bake` copies.
pub fn bake_copy_names() -> &'static [String] {
    static INSTANTIATED: std::sync::OnceLock<(String, Vec<String>)> = std::sync::OnceLock::new();
    &INSTANTIATED.get_or_init(instantiate_bake_copies).1
}

fn instantiate_bake_copies() -> (String, Vec<String>) {
    let start = PRELUDE.find(BAKE_REGION_START).expect("the profile-solid region starts");
    let end = PRELUDE[start..].find(BAKE_REGION_END).expect("and ends") + start;
    let region = &PRELUDE[start..end];
    // Every function block: from a line starting `fn ` to the first line that is `}` alone.
    // Structs, constants and comments between them are not copied.
    let mut names = Vec::new();
    let mut blocks = Vec::new();
    let mut rest = region;
    while let Some(at) = rest.find("\nfn ") {
        let from = at + 1;
        let close = rest[from..].find("\n}").expect("a function closes") + from + 2;
        let block = &rest[from..close];
        let name = block[3..].split('(').next().expect("a name").trim().to_string();
        names.push(name);
        blocks.push(block.to_string());
        rest = &rest[close..];
    }
    let mut copies = String::from(
        "\n// ---- the profile solids again, for the normal pass: generated from the region above \
         by `instantiate_bake_copies`, reading every section through `profile_distance_bake` ----\n",
    );
    for block in &blocks {
        copies.push_str(&to_bake_names(block, &names));
        copies.push_str("\n\n");
    }
    (format!("{PRELUDE}{copies}"), names)
}

/// `text` with every call of, and definition of, a function in `names` renamed to its `_bake`
/// copy, and every section read through `profile_distance_bake`. A name is matched whole: the
/// character before it is not part of an identifier, so `extrude_distance` leaves
/// `wall_extrude_distance` to its own entry in the list.
pub fn to_bake_names(text: &str, names: &[String]) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    let bytes = text.as_bytes();
    let mut i = 0;
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    'scan: while i < bytes.len() {
        if i == 0 || !ident(bytes[i - 1]) {
            let here = &text[i..];
            for name in names.iter().map(String::as_str).chain(["profile_distance"]) {
                if here.starts_with(name) && here[name.len()..].starts_with('(') {
                    out.push_str(name);
                    out.push_str("_bake(");
                    i += name.len() + 1;
                    continue 'scan;
                }
            }
        }
        let c = text[i..].chars().next().expect("a char");
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// The sphere tracer, appended after the generated function because WGSL requires a function
/// to be declared before it is called.
#[cfg(not(feature = "march-stats"))]
pub(crate) const MARCH: &str = concat!(include_str!("../camera.wgsl"), include_str!("../march.wgsl"));
/// With the counting copy of the primary march after it — a second entry point in the same
/// module, which the frame's own pipeline never names. See `march_stats.wgsl`.
#[cfg(feature = "march-stats")]
pub(crate) const MARCH: &str = concat!(
    include_str!("../camera.wgsl"),
    include_str!("../march.wgsl"),
    include_str!("../march_stats.wgsl")
);

/// A lattice that follows a surface, appended only to programs with one in them.
///
/// Out of `PRELUDE` because a compiler cannot remove a branch on a buffer value, so while it was
/// in there every lattice shader built the surface walk: 2.3 to 5.3 seconds a pipeline, cold,
/// for a gyroid with no surface in the document. See the header of `conformal.wgsl`.
pub(crate) const CONFORMAL: &str = include_str!("../conformal.wgsl");

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum CodegenError {
    #[error("the graph has a cycle, so it has no evaluation order")]
    Cyclic,
    /// A block whose slot is empty, named so the refusal says which block to fill.
    ///
    /// **The one refusal a hole gets**, wherever the generator meets one: a body that cannot be
    /// written, or a number driven by a ramp that cannot. It replaces `RampHasNoInput`, which
    /// answered for one kind of block when an empty slot is a state every slot can be in
    /// (`notebook-spec.md` §7.2) — and it is the same refusal under the same name as
    /// `CompileError::EmptySlot`, so a part the field cannot evaluate is a part the shader
    /// declines to write, rather than two differently worded complaints about one hole.
    #[error(
        "a block has not been told what to read \u{2014} node {0:?} has an empty slot, so there \
         is nothing to generate for it. Put a block in it, or take out what reads it."
    )]
    EmptySlot(NodeId),
    /// A body that is, or reads, a cage the document holds open -- the same refusal
    /// `CompileError::OpenCage` is, for the same reason `EmptySlot` is shared (R12).
    #[error("cage {0} is not a solid, so there is nothing to generate for it")]
    OpenCage(fieldform_scene::CageId),
    /// The field refuses this by name and so does the shader: an aimed lattice is asked in its
    /// own frame, and a ramp driving its phase would be read there rather than in the world.
    #[error(
        "a lattice whose wall, density or strut follows a ramp cannot also be aimed by its own \
         frame yet \u{2014} leave the lattice's frame as it is and move the finished body with \
         a placement, which carries both"
    )]
    GradedLatticeCannotBeAimed,
    /// A prism is measured in millimetres across a flat period, and only a rectangular map is
    /// one. The field refuses the rest by name, so the shader does.
    #[error(
        "a honeycomb cannot be laid over that map yet \u{2014} use a rectangular map, and turn \
         or place the lattice to aim it"
    )]
    CellCannotTakeThatMap,
    /// A jittered seed grid searches the cells around a point, and a conformal map gives each
    /// face its own coordinates -- so the search would stop at every face edge. The field
    /// refuses it by name, so the shader does.
    #[error(
        "a Voronoi cell cannot follow a surface yet \u{2014} use a rectangular, cylindrical or \
         spherical map"
    )]
    CellCannotTakeAConformalMap,
    /// A loft whose sections or centerline cannot be built into one -- too few sections, a
    /// section out of order along the centerline, a centerline folding through itself. The
    /// compile refuses the same loft with the reason (`CompileError::BadLoft`); this only says
    /// which block, since the shader has nothing it could draw in its place.
    #[error("loft {0:?} cannot be built from its sections and centerline")]
    LoftCannotBeBuilt(NodeId),
    #[error("the part has no bodies, so there is nothing to draw")]
    Empty,
    #[error("boolean {0:?} has no operands, so there is nothing to combine")]
    NothingToCombine(NodeId),
    #[error("the graph references node {0:?}, which it does not contain")]
    DanglingNode(NodeId),
    #[error(
        "the graph contains {0}, so there is nothing to generate for it — the document still \
         opens and saves, and the node is written back unchanged"
    )]
    UnreadableNode(String),
    #[error(
        "{0} came out as infinity or not-a-number, so there is no shader to write — a ramp          whose input range is empty is the way this happens, and the CPU refuses that graph too"
    )]
    NotANumber(&'static str),
    #[error(
        "the generated shader did not compile, which is this build's fault rather than the          document's: {0}"
    )]
    Invalid(String),
}

/// A generated shader, plus everything the caller has to bind for it to run.
///
/// The lists are in binding order: `baked[i]` is the leaf the shader calls `sample_baked(i, …)`
/// on, and `params[i]` is where slot `i` of the parameter buffer takes its value from. Handing
/// those back rather than resolving them here keeps this module free of the document, and free
/// of any need for a GPU — which is what lets it be tested by parsing its own output.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderProgram {
    /// Complete WGSL: the prelude followed by the generated `field` function.
    pub source: String,
    /// The leaves that reach the GPU as a baked narrow-band texture rather than as an
    /// expression, in binding order.
    ///
    /// A cage and an imported mesh are both here, and the shader cannot tell them apart —
    /// correctly, because there is nothing to tell: neither has a closed form, both answer by
    /// closest-point search against triangles, and the texture is what that search becomes on
    /// the way to a GPU. The renderer never knew it was baking a *cage*; only the name did.
    pub baked: Vec<BakedLeaf>,
    /// Polylines the shader needs uploaded, in binding order.
    ///
    /// A pipe's spine and an extrude's profile share one buffer because they are the same thing
    /// to the shader — a list of points to walk. They differ only in what the caller has to
    /// compute, which is why [`CurveUse`] says which.
    pub curves: Vec<CurveUse>,
    /// Surfaces a conformal lattice follows, in binding order — the caller packs these and
    /// uploads them, and a `params` slot carries each one's position in this list.
    ///
    /// **Ids rather than packed data**, because generating a shader must not depend on any
    /// number in the document: a program is keyed on structure so that changing a value costs a
    /// buffer write instead of a recompile, and a tessellation is the largest value there is.
    ///
    /// Paired with the lattice's cells-per-face, which is what `surface_levels` derives the
    /// tessellation depth from — so a caller holding only the id could not arrive at the same
    /// surface the field compiled. Two lattices over one mesh at different counts are two
    /// entries here, correctly: they need different tessellations.
    pub surfaces: Vec<(fieldform_scene::QuadMeshId, u32)>,
    /// Where each parameter slot's value comes from.
    pub params: Vec<ParamRef>,
    /// The node behind each operand mask, indexed by its bit -- see [`OperandMasks`].
    ///
    /// Reported rather than inferred from the source, because the question a bit answers
    /// is "did masking *this operand* ever fire", and a count of `masked_operand_N` in the
    /// text cannot say which block it came from.
    pub masked: Vec<fieldform_scene::NodeId>,
    /// How big this program is, for the perf line and for the message a refused pipeline shows.
    pub size: ShaderSize,
    /// Whether this program settles a hit onto its surface: the `SETTLES` it was generated with,
    /// for `cpu_march`'s replica of the march to run the same loop.
    pub settles: bool,
    /// Where each body's guard box starts in [`ShaderProgram::params`], in the order
    /// `Graph::roots` gives, and `None` for a body that has no box.
    ///
    /// **Reported because two callers need the same answer and neither can derive it.** The slots
    /// are pushed after every node's own, so their indices are not something a caller could work
    /// out from the graph; and a body has a box only when it is worth guarding (`GUARD_WEIGHT`) and
    /// reports a true distance, which is a decision made here. `box_guard` reads six consecutive
    /// slots from one of these: `lo.xyz` then `hi.xyz`.
    ///
    /// `tile_cull_oracle.rs` is the first caller -- it blanks a body's box to make the existing
    /// guard skip it, which is how the per-tile mask's prize is bounded without building the mask.
    /// A cull pass would be the second, and for the same reason.
    pub body_boxes: Vec<Option<usize>>,
    /// Where the per-tile cull's extents start in [`ShaderProgram::params`], or `None` where this
    /// program does not cull -- one body, more than `CULL_CAP`, or the guard switched off.
    ///
    /// Six slots per body, `lo.xyz` then `hi.xyz`, contiguous and in `body_boxes`' order, one for
    /// **every** body rather than only the guarded ones: a cull extent is a promise about where a
    /// body can be, so there is no such thing as a body without one. An unbounded body's slots
    /// hold ±1e30 and an empty one's are inverted, both of which the cull reads as written.
    ///
    /// Reported for the reason `body_boxes` is: the slots are pushed after every node's own, so
    /// their index is not something a caller could derive, and whether there are any at all is a
    /// decision this module makes. `Some(_)` is also the one honest answer to "does this program
    /// cull" -- the source's `CULL_BASE` says the same thing, and parsing it back would be reading
    /// our own output to learn what we decided.
    pub cull_base: Option<usize>,
}

/// What a generated program measures, reported on every generate.
///
/// **Because nothing warned before a driver refused one** (plan step 13). MPR's hard ceiling was a
/// per-thread register and tape budget; here it is the driver's compile of a generated shader, and
/// `union-compile-8fa06b7.md` measured that cost at a power of about 2 to 2.5 in the size of one
/// entry point *after inlining* — sixteen unioned pipes reached 123 seconds. So the wall is not
/// hypothetical and was not found on a customer's graph: it was found on sixteen pipes.
///
/// **These four are the cheap proxies, not the predictor.** The quantity that predicts the driver's
/// cost is the inlined size of `march`, which needs a call-graph walk over the whole source
/// including the prelude — too much to do on every rebuild, and a number a drag would pay for.
/// `fieldform-gpu/examples/compile_where_the_time_goes.rs` computes the real one; these are what a
/// generate can afford, and they move with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShaderSize {
    /// The whole source, prelude included, in bytes — which is what the driver is handed.
    pub bytes: usize,
    /// Bodies, which is `Graph::roots().len()`: one `body_k` function and one `body_normal_k` each.
    pub bodies: usize,
    /// Nodes in the graph, before any lowering.
    pub nodes: usize,
    /// The longest chain of declared points, `p0` included.
    ///
    /// **The multiplier a node count cannot show.** Every block that changes the point makes its
    /// whole subtree be written again at the new point, and that compounds down a chain — a mirror
    /// over a twist over an extrude writes the extrude once per point either declares. So a graph
    /// of ten nodes and a chain of six is a much larger shader than a graph of ten and a chain of
    /// one. This is the cheap structural form of that; `shader_composition.rs` shows the whole of it.
    pub deepest_point_chain: usize,
}

impl std::fmt::Display for ShaderSize {
    /// As the perf line wants it: `12 KB, 3 bodies, 41 nodes, points 2 deep`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} KB, {} bodies, {} nodes, points {} deep",
            self.bytes / 1024,
            self.bodies,
            self.nodes,
            self.deepest_point_chain,
        )
    }
}

impl ShaderProgram {
    /// The generated part alone, without the prelude — for tests and for diagnostics that want
    /// to show what a graph compiled to without a page of runtime around it.
    pub fn field_function(&self) -> &str {
        // **Everything generated, which is what the sentence above already promised.** It used
        // to start at `fn field(` and so returned only the fold, which was the whole of the
        // generated composition right up until each body became a function of its own. The
        // narrower version quietly excluded the part the tests were looking for.
        self.source.strip_prefix(prelude()).unwrap_or(&self.source)
    }
}

/// Rewrites an aimed lattice as an unaimed one under a [`Node::Transform`].
///
/// **The same move the CPU makes, in the form this side already has.** `to_cell_coords` reads a
/// pattern laid out about the world's own axes, so aiming one is a matter of putting the point
/// into its frame first — and this crate already lowers a point into a frame, in `to_local`,
/// driven by the point context a `Transform` sets up. Writing that rotation again inside the
/// lattice function would be a second implementation of it, and the two could disagree in ways
/// only a picture would show.
///
/// A frame that says nothing is left alone, so the ordinary lattice generates exactly the shader
/// it generated before.
fn expand_frames(document: &fieldform_scene::Document, graph: &Graph) -> Graph {
    let aimed: Vec<NodeId> = graph
        .nodes()
        .filter(|(_, n)| match n {
            Node::Lattice { frame, .. } | Node::Twist { frame, .. } => !frame.is_default(),
            _ => false,
        })
        .map(|(id, _)| id)
        .collect();
    if aimed.is_empty() {
        return graph.clone();
    }

    let mut out = graph.clone();
    for id in aimed {
        // A leaf takes one placement in front of it; a node that *wraps a body* takes a
        // sandwich.
        //
        // A lattice tiles space and has no input, so putting the query point into its frame is
        // the whole of aiming it. A twist wraps a body, so a single placement would also hand
        // that body a point in the frame's local space — and the body lives in world space. The
        // result is a twist about the right axis applied to a body that has been moved, which
        // the CPU/GPU residual measured at 248 mm before this was noticed.
        //
        // What is wanted is `R . twist . R⁻¹` on the point: into the frame, wind about its own
        // Z, back out, then look the body up. `to_local` evaluates its input at `R⁻¹(p − t)`, so
        // the inverse of that map is the placement with rotation `R⁻¹` and translation `−R⁻¹t`.
        let (frame, inner) = match out.node(id).cloned() {
            Some(Node::Lattice { cell, map, phase, frame }) => (
                frame,
                Node::Lattice { cell, map, phase, frame: fieldform_scene::Frame::default() },
            ),
            Some(Node::Twist { input, degrees_per_mm, frame }) => {
                let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
                let Ok((translation, rotation)) =
                    fieldform_field::resolve_frame(document, &out, &frame, &numbers)
                else {
                    continue;
                };
                let back = rotation.inverse();
                let (bx, by, bz) = back.euler_angles();
                let shift = -(back * translation);
                let undone = out.add(Node::Transform {
                    input,
                    translation: [
                        ParamRef::Literal(shift.x),
                        ParamRef::Literal(shift.y),
                        ParamRef::Literal(shift.z),
                    ],
                    rotation: [
                        ParamRef::Literal(bx.to_degrees()),
                        ParamRef::Literal(by.to_degrees()),
                        ParamRef::Literal(bz.to_degrees()),
                    ],
                    scale: ParamRef::Literal(1.0),
                    about: fieldform_scene::Place::origin(),
                });
                (
                    frame,
                    Node::Twist {
                        input: undone.into(),
                        degrees_per_mm,
                        frame: fieldform_scene::Frame::default(),
                    },
                )
            }
            _ => continue,
        };
        // Through the one resolver, so this side accepts exactly what the kernel does — the
        // whole point of the frame being a shared type rather than two fields (D9).
        let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
        let Ok((translation, rotation)) = fieldform_field::resolve_frame(document, graph, &frame, &numbers)
        else {
            // Refused by the kernel too, and with a better message than this side could give.
            // Left alone rather than half-lowered, so the refusal is what a caller meets.
            continue;
        };
        // Back to the Euler triple a `Transform` carries. `euler_angles` is the exact inverse of
        // the `from_euler_angles` the transform is rebuilt with, so nothing is lost going round.
        let (rx, ry, rz) = rotation.euler_angles();

        // The node itself, now facing the way it always did in its own frame.
        let pattern = out.add(inner);
        // The lattice node *becomes* the transform, rather than a copy of it being written into
        // one. Everything referring to the lattice still refers to it, and no spare node is
        // left over as a root — the same reason `expand_shells` ends the way it does.
        if let Some(slot) = out.node_mut(id) {
            *slot = Node::Transform {
                input: pattern.into(),
                translation: [
                    ParamRef::Literal(translation.x),
                    ParamRef::Literal(translation.y),
                    ParamRef::Literal(translation.z),
                ],
                rotation: [
                    ParamRef::Literal(rx.to_degrees()),
                    ParamRef::Literal(ry.to_degrees()),
                    ParamRef::Literal(rz.to_degrees()),
                ],
                scale: ParamRef::Literal(1.0),
                about: fieldform_scene::Place::origin(),
            };
        }
    }
    out
}

/// A lowered graph with anything that became a body *because of the lowering* taken back out.
///
/// **Lowering may not change what the bodies are.** A lattice aimed at a plane is rewritten by
/// `expand_frames` into a placement over a lattice facing its own way — and the frame it drops
/// on the way was the *only* thing consuming that plane. So the plane came out of the lowering
/// unconsumed, which makes it a root, which makes it a body, which puts a shaded half-space
/// across the whole viewport. Reported as exactly that: "when you create a plane in the gyroid
/// tool it creates a physical shaded plane", and "only the yellow plane should be visible, not
/// an actual shaded geometry plane".
///
/// Stated as the invariant rather than patched at the one site, because the same thing is true
/// of every lowering pass there will ever be: a rewrite is supposed to preserve what is drawn,
/// and a node that was an *ingredient* before it ran has not become a body by being rewritten
/// around. `Graph::with_turns_expanded_using` keeps its reference alive for the same reason and
/// says so; this catches the passes that cannot.
///
/// Cascaded, since a plane can itself be sitting on a point which is then orphaned in its turn.
fn only_the_same_bodies(before: &Graph, mut lowered: Graph) -> Graph {
    let bodies = before.roots();
    loop {
        let strays: Vec<NodeId> = lowered
            .roots()
            .into_iter()
            .filter(|id| !bodies.contains(id))
            .collect();
        if strays.is_empty() {
            return lowered;
        }
        for id in strays {
            lowered.remove(id);
        }
    }
}

/// Rewrites every `Shell` into the body-minus-its-own-inward-offset it stands for.
///
/// The same division of labour as `expand_frames`, and for a plainer reason: a shell *is* that
/// composition, on both paths — the CPU builds the two ops directly in its compile arm. This is
/// where the shader does the same thing, by rewriting the graph rather than by growing an arm
/// that would have to repeat the offset's own emission.
///
/// The body is **shared**, not copied: one node read twice at the same point, which is a single
/// SSA slot in the generated source. A copy would emit the whole body twice and double the
/// shader for a wall.
fn expand_shells(graph: &Graph) -> Graph {
    let shells: Vec<NodeId> = graph
        .nodes()
        .filter(|(_, n)| matches!(n, Node::Shell { .. }))
        .map(|(id, _)| id)
        .collect();
    if shells.is_empty() {
        return graph.clone();
    }

    let mut out = graph.clone();
    for id in shells {
        let Some(Node::Shell { input, thickness }) = out.node(id).cloned() else { continue };
        // Negated, because the cavity is the body shrunk and `Offset` subtracts. A named
        // thickness keeps its name here, since nothing multiplies it — unlike a pattern's step,
        // this is one number used once.
        let cavity = out.add(Node::Offset {
            input,
            distance: match thickness {
                ParamRef::Literal(v) => ParamRef::Literal(-v),
                // A named or driven thickness cannot be negated without resolving it, and both
                // are refused before a shader is generated — a shell's thickness is not
                // driveable, and a named one resolves through the parameter buffer where this
                // sign flip cannot reach. Left as-is so the refusal is what a caller meets.
                other => other,
            },
        });
        // The shell node *becomes* the difference, so everything referring to it still does and
        // no spare node is left over as a second body.
        if let Some(slot) = out.node_mut(id) {
            *slot = Node::Boolean {
                op: fieldform_scene::BooleanOp::Difference,
                inputs: vec![input, cavity.into()],
            };
        }
    }
    out
}




/// Parses and validates generated WGSL, so a shader that cannot compile is an error here.
///
/// wgpu treats a shader compilation failure as **fatal** — it panics the process — so anything
/// this module gets wrong takes the whole application down with a message about the generated
/// source rather than about the document that produced it. That happened: an infinite divisor
/// printed as `inf`, which WGSL reads as an undefined identifier.
///
/// `naga` is the same front end wgpu uses, so a module that passes here is one the driver will
/// accept. It costs a parse of the generated source, on the structural edits that regenerate it
/// rather than per frame — sub-millisecond for an ordinary graph, a few for a 512-copy pattern.
/// That is a cheap price for never crashing on our own output.
///
/// **Paid once per source, not once per edit.** "Sub-millisecond" above was written when the
/// prelude was a fraction of what it is; measured now, by `edit_step_cost.rs`, the parse is 5 to
/// 6 ms of a 120 KB source and was the whole of what generating cost -- the generator's own
/// work is about 0.15 ms -- and it ran on every step of every drag, on the interface's thread,
/// over a source that had not changed by a byte, because every number in it is a buffer read.
/// Whether a source validates depends on nothing but the source, so the ones that have passed
/// are remembered and compared whole. Only passes: a failure is rare, and is wanted in full
/// each time.
pub(crate) fn validated(source: &str) -> Result<(), CodegenError> {
    // A handful is plenty. A drag alternates between at most two sources -- a fold that goes
    // from windowed to whole and back -- and everything older is another document.
    const REMEMBERED: usize = 8;
    static PASSED: std::sync::Mutex<std::collections::VecDeque<std::sync::Arc<str>>> =
        std::sync::Mutex::new(std::collections::VecDeque::new());
    // A poisoned lock means a panic elsewhere while it was held; what is in it is still a list
    // of sources that passed, so it is still true.
    let seen = |list: &std::collections::VecDeque<std::sync::Arc<str>>| {
        list.iter().any(|passed| **passed == *source)
    };
    if seen(&PASSED.lock().unwrap_or_else(|e| e.into_inner())) {
        return Ok(());
    }
    validate_now(source)?;
    let mut passed = PASSED.lock().unwrap_or_else(|e| e.into_inner());
    if !seen(&passed) {
        if passed.len() == REMEMBERED {
            passed.pop_front();
        }
        passed.push_back(std::sync::Arc::from(source));
    }
    Ok(())
}

/// The check itself, every time it is asked.
fn validate_now(source: &str) -> Result<(), CodegenError> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| CodegenError::Invalid(e.emit_to_string(source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| CodegenError::Invalid(format!("{e:?}")))?;
    Ok(())
}

/// A number on its way into the generated source, refused if it is not one.
///
/// `{:?}` on an infinite `f64` writes `inf`, which is not a WGSL literal — it is parsed as an
/// identifier, and there is no `inf` in scope. The shader then fails to compile, and wgpu
/// treats a shader compilation failure as fatal, so the application panics with a message
/// about an unknown identifier rather than about the number that produced it.
///
/// It got there through a driven offset's divisor: a ramp whose input range is dragged to zero
/// has unbounded slope, `graph_lipschitz` says so honestly with infinity, and that arrived here
/// as a divisor. The CPU refuses such a graph at compile time; this is the shader saying the
/// same thing instead of emitting something that cannot parse.
fn finite(value: f64, what: &'static str) -> Result<f64, CodegenError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CodegenError::NotANumber(what))
    }
}

/// A ramp's five numbers, resolved, in the order the shader helper takes them.
///
/// Constants rather than parameter slots — see `NodeEmitter::ramp` for why. The shape rides
/// along as a float because WGSL has no enums and it is a constant either way.
fn ramp_constants(
    document: &fieldform_scene::Document,
    node: &Node,
) -> (f64, f64, f64, f64, f64) {
    let Node::Ramp { in_min, in_max, out_min, out_max, shape, .. } = node else {
        return (0.0, 1.0, 0.0, 0.0, 0.0);
    };
    let (lo, hi) = (resolved(document, in_min), resolved(document, in_max));
    let (out_lo, out_hi) = (resolved(document, out_min), resolved(document, out_max));
    // An empty span is refused on the CPU before a shader is generated; one here would divide
    // by zero, so it falls back to a span of one rather than filling the viewport with NaN.
    let span = if (hi - lo).abs() < f64::EPSILON { 1.0 } else { hi - lo };
    let code = match shape {
        fieldform_scene::RampShape::Linear => 0.0,
        fieldform_scene::RampShape::Smooth => 1.0,
        fieldform_scene::RampShape::Smoother => 2.0,
    };
    (lo, span, out_lo, out_hi - out_lo, code)
}

/// A parameter's value at generation time, or zero.
///
/// Zero rather than an error because this runs before the walk that reports unresolvable
/// parameters, and that walk gives the better message. A pattern whose step is a driven
/// expression stacks its copies in one place, which is visibly wrong rather than quietly so.
fn resolved(document: &fieldform_scene::Document, r: &ParamRef) -> f64 {
    document.resolve(r).unwrap_or(0.0)
}

/// Compiles a graph into a shader.
///
/// Depends only on the graph's structure, so two graphs that differ solely in their numbers
/// produce byte-identical source and the second needs no recompilation. **Nearly**, and the two
/// exceptions are both measured, because this sentence is the one anything caching a program
/// would rest on:
///
/// - **A number a *predicate* reads still reaches the source.** `mirror_one_side` asks whether
///   the whole input lies on one side of the plane, and when it does the shader folds the point
///   instead of taking a minimum of two readings. So dragging a mirror's plane across its body
///   changes the text; dragging it while it stays clear does not.
///   `fieldform-gpu/examples/draggable_numbers.rs` has the row, and fifteen others that hold.
/// - **The order the bodies come in is the ids'.** `Graph::roots` sorts by raw `NodeId` and
///   `NodeId::new` is random, so making or removing a node re-ranks the rest and renames them
///   through the source. Regenerating an unchanged graph is byte-identical;
///   `fieldform-gpu/examples/body_order_probe.rs` measures both halves.
///
/// **And this is not the authority on whether a document is one.** `GraphField::compile` refuses
/// things generation does not look at -- a profile that strays out of its plane is the measured
/// case (`fieldform-gpu/examples/planarity_probe.rs`), since a profile reaches the shader as an
/// uploaded buffer and how its points sit is the caller's business. The application compiles the
/// field first and draws nothing when that fails, deliberately; a caller that reaches straight
/// for this function gets a shader for a document the rest of the codebase refuses.
pub fn generate(
    document: &fieldform_scene::Document,
    graph: &Graph,
) -> Result<ShaderProgram, CodegenError> {
    generate_with(document, graph, &[])
}

/// The same, told what each driven morph was divided by.
///
/// **Because the shader has to divide by the number the CPU divided by, and cannot work it out.**
/// `morph-spec.md` §4.3 bounds a driven morph by `sup|B - A|` read from the two operands'
/// *intervals*, and an interval is evaluated over a compiled op list -- which exists on the CPU
/// and not here. Given nothing, this falls back to bounding the same quantity by the diagonal of
/// the box enclosing both operands, which is sound and much looser: on the reported scene it
/// prices at x42.5 where the intervals say x24.5, and every step is that much shorter.
///
/// So the caller that has a compiled field hands its `morph_divisors` along and the two agree by
/// construction rather than by both deriving the same thing twice. A caller that has none gets
/// the box bound on both sides, which is still one number -- what must never happen is one path
/// taking the interval bound and the other the box bound, which is one surface in two places.
pub fn generate_with(
    document: &fieldform_scene::Document,
    graph: &Graph,
    morph_divisors: &[(NodeId, f64)],
) -> Result<ShaderProgram, CodegenError> {
    // `FIELDFORM_NO_PATTERN_PRUNE` leaves the instance prune out, so the application can be
    // measured both ways on one document; nothing else reads it.
    let prune = std::env::var_os("FIELDFORM_NO_PATTERN_PRUNE").is_none();
    generate_inner(
        document,
        graph,
        morph_divisors,
        true,
        cull_wanted(),
        prune,
        masks_wanted(),
        run_floor_wanted(),
    )
}

/// [`generate`] with each exact skip switched by hand, so an example can hold the pruned field
/// to the unpruned one and measure what each is worth. Not for drawing.
#[doc(hidden)]
pub fn generate_with_options(
    document: &fieldform_scene::Document,
    graph: &Graph,
    guard_bodies: bool,
    prune_patterns: bool,
) -> Result<ShaderProgram, CodegenError> {
    generate_inner(
        document,
        graph,
        &[],
        guard_bodies,
        cull_wanted(),
        prune_patterns,
        masks_wanted(),
        run_floor_wanted(),
    )
}

/// Whether [`OperandMasks`] is wanted. `FIELDFORM_NO_OPERAND_MASKS` is a measurement's off
/// switch, as the pattern prune and the spine index have one: each masked operand is a function
/// more for the driver, and what that costs can only be read against the same graph without.
fn masks_wanted() -> bool {
    std::env::var_os("FIELDFORM_NO_OPERAND_MASKS").is_none()
}

/// Whether the per-tile body cull is wanted, with `FIELDFORM_NO_TILE_CULL` to turn it off.
///
/// **On, after a day off, and the day is worth knowing about** (`tile-mask-design-01941c5.md`
/// §1.7a–c). Within an hour of shipping, the owner photographed a bar of background through a
/// 2 mm shell. The cause was written up as "the relaxed step overshoots a thin wall" and the cull
/// was switched off; then masked rays were made to march plainly, which was sound and turned the
/// feature into a trade (0.86× on dense scenes). Both were treating a symptom. The kept fold is an
/// exact distance to everything a masked ray can hit, so Keinert's vouch on it is a sound
/// argument that a long step never crosses a kept surface -- and the march was not running that
/// argument: a hop that followed a relaxed step zeroed `reach` whenever the landing cell was
/// clear, so the step's tail was vouched by nothing. Unmasked, a reading is only taken in a cell
/// the grid could not clear, which bounds the tail at under a cell; masked, the reading is the
/// distance to something far away and the tail swallowed a whole shell. `march.wgsl` now vouches
/// a long step before the grid may hop from its landing, for every ray, and the masked relaxed
/// march is bit-identical to the unmasked one over 768 cameras of that document.
///
/// What it buys, least of three, 720 px: 2.67× on fifty spread thin shells, 1.36× on fifty spread
/// solids, 1.19× orthographic, **1.08× on fifty touching solids** -- the scene that lost 14 % under
/// the plain march -- and nothing below 1.0 anywhere measured. The unmasked column did not move.
///
/// **Its own switch, not the guard's.** The two rule out different things, and a measurement that
/// moved both at once could not say which paid; the arm that answers "was the mask worth
/// building" is the cull against *the guard alone*, which `masked_matches_unmasked` is.
fn cull_wanted() -> bool {
    std::env::var_os("FIELDFORM_NO_TILE_CULL").is_none()
}

/// Whether a union's members may be walked as a loop: [`RUN_FLOOR`] unless switched off.
///
/// `FIELDFORM_NO_UNION_LOOP` is a measurement's off switch, as the pattern prune and the operand
/// masks each have one — and this one earns its keep in `gpu_matches_cpu`, which can then check
/// **the same scene both ways**. That is the only control that tells a run reading the wrong spine
/// from a crease the union has whether it loops or not: a union of pipes has a seam where two of
/// them meet, the normal is genuinely discontinuous there, and a worst-case normal figure taken
/// once says nothing on its own.
///
/// **It has now paid for itself twice**, the second time by failing. The blade fan read 228 mm off
/// with the loop switched off and nothing at all with it on, which was the *harness* pairing
/// profile bindings by position — a fault no single-arm run could have shown, because the arm
/// that was wrong was the one nothing had exercised.
fn run_floor_wanted() -> usize {
    if std::env::var_os("FIELDFORM_NO_UNION_LOOP").is_some() {
        usize::MAX
    } else {
        RUN_FLOOR
    }
}

/// [`generate`] with the operand masks forced on or off, so a test can hold the two compositions
/// against each other without an environment variable that the next thread in the process would
/// see too. Not for drawing.
#[doc(hidden)]
pub fn generate_with_masks(
    document: &fieldform_scene::Document,
    graph: &Graph,
    mask_operands: bool,
) -> Result<ShaderProgram, CodegenError> {
    generate_inner(document, graph, &[], true, cull_wanted(), true, mask_operands, run_floor_wanted())
}

/// [`generate`] with the union loop's floor set by hand, so an example can hold a union walked as
/// a loop against the same union written out as a call an operand.
///
/// `usize::MAX` never loops, which is the arm this crate shipped before the loop existed. Not for
/// drawing: `RUN_FLOOR` is the measured number and the only one the application should compile.
#[doc(hidden)]
pub fn generate_with_run_floor(
    document: &fieldform_scene::Document,
    graph: &Graph,
    run_floor: usize,
) -> Result<ShaderProgram, CodegenError> {
    generate_inner(document, graph, &[], true, cull_wanted(), true, masks_wanted(), run_floor)
}

/// [`generate`] with the per-body box guard left out, so an example can measure what the guard
/// is worth against the same scene. Not for drawing: the guard is exact, and a shader without
/// it is only slower.
#[doc(hidden)]
pub fn generate_for_measurement(
    document: &fieldform_scene::Document,
    graph: &Graph,
    guard_bodies: bool,
    cull_bodies: bool,
) -> Result<ShaderProgram, CodegenError> {
    generate_inner(
        document,
        graph,
        &[],
        guard_bodies,
        cull_bodies,
        true,
        masks_wanted(),
        run_floor_wanted(),
    )
}

/// How far a body's guard box is grown past its extent, as a fraction of the box's diagonal.
///
/// The guard is exact when the box distance is a lower bound on the body's value, and for a
/// body that reports true distance it is — mathematically. In `f32` the two are computed by
/// different expressions, and along an axis a sphere's `length(p - c) - r` and its box's
/// `|p.x - hi.x|` can land an ulp apart in either order. A tenth of a percent of padding puts
/// the guard that far under the body's value everywhere outside the box, so a rounding error
/// can never skip a body that the fold would have taken, and `gpu_matches_cpu` stays bit for
/// bit what it was.
const BOX_GUARD_PAD: f64 = 1e-3;

/// The most bodies a per-tile cull will carry, from `tile-mask-design-01941c5.md` §1.5.
///
/// Eight mask words, which is 3.2 MB of mask for a 2560² frame. Past it the cull is switched off
/// rather than truncated: a mask that covered some bodies and not others would need every reader
/// to know which, and the whole arrangement rests on one rule -- a set bit rules a body out -- that
/// a partial mask would quietly break.
const CULL_CAP: usize = 256;

/// How far a cull extent is grown past the body's own box, before it is tested against a tile.
///
/// **The box bounds the mathematical surface; the cull has to bound the drawn one** (§1.6, review
/// point 2). They are not the same shape, and every difference is a place a correct box culls a
/// tile that has material in it:
///
/// - a baked profile answers `bilinear − margin`, so its surface sits outside its own boundary;
/// - a pipe's box is built from `tessellate(0.1)` while the shader walks `tessellate_to_angle`;
/// - a cage is drawn from a bake with 0.21 mm of error against a hull box with no slack;
/// - a hit is accepted anywhere inside `epsilon × divisor` of the surface.
///
/// The extent sweep measured all of those together as **R7**, at 0.16 mm worst over 153 cases
/// (§1.8). A thousandth of the diagonal covers that on anything larger than 160 mm, and the
/// absolute floor covers it on anything smaller -- a 20 mm part gets the floor, not 0.02 mm.
///
/// This is the *world-space* pad. The cull's own one-pixel screen pad is a different thing and
/// stays: that one is for jitter, this one is for the gap between the surface and its box.
const CULL_PAD_FRACTION: f64 = 1e-3;
const CULL_PAD_FLOOR_MM: f64 = 0.5;

/// The six slots a body's *cull* extent takes, padded for the drawn surface.
///
/// **Every root gets one, where a guard box is only for some.** A guard box is an optimisation and
/// skips a body it cannot help; a cull extent is a promise about where a body can be, so a body
/// without one would have to be treated as everywhere -- which is what `Unbounded` already writes,
/// and what `Empty` inverts. So this never returns `None` for a root that has a box at all:
///
/// - **Unbounded** writes ±1e30, which every frustum touches, so such a body is never culled.
/// - **Empty** writes an *inverted* box, `lo = +1e30` and `hi = −1e30`, which the cull rejects
///   before any plane test (§1.6, review point 7). Four side planes and no far plane would
///   otherwise accept a far box straight down the view.
fn cull_extent_slots(
    document: &fieldform_scene::Document,
    graph: &Graph,
    root: NodeId,
    params: &mut Vec<ParamRef>,
) -> usize {
    let first = params.len();
    let six = match fieldform_field::graph_extent(document, graph, root) {
        fieldform_field::Extent::Unbounded => [-1e30, -1e30, -1e30, 1e30, 1e30, 1e30],
        fieldform_field::Extent::Empty => [1e30, 1e30, 1e30, -1e30, -1e30, -1e30],
        fieldform_field::Extent::Bounded { lo, hi } => {
            let pad = ((hi - lo).norm() * CULL_PAD_FRACTION).max(CULL_PAD_FLOOR_MM);
            let grown = [
                lo.x - pad,
                lo.y - pad,
                lo.z - pad,
                hi.x + pad,
                hi.y + pad,
                hi.z + pad,
            ];
            // A box with a number that is not finite says nothing about where the body is, so it
            // is written as the one that rules nothing out rather than as a box.
            if grown.iter().all(|v| v.is_finite()) {
                grown
            } else {
                [-1e30, -1e30, -1e30, 1e30, 1e30, 1e30]
            }
        }
    };
    for v in six {
        params.push(ParamRef::Literal(v));
    }
    first
}

/// How dear a body has to be before its box is asked ahead of it, in units of one primitive's
/// evaluation.
///
/// **Measured, not chosen** — `fieldform-gpu/examples/spread_bodies_cost.rs` on the RTX A500,
/// bodies three radii apart at 720 × 405, guarded against not:
///
/// | bodies | one sphere each | four spheres each | eight spheres each |
/// |---|---|---|---|
/// | 10 | 0.9 → 2.6 ms | 17.1 → 10.5 ms | 34.0 → 3.9 ms |
/// | 100 | 17.4 → 61.5 ms | 340 → 111 ms | 718 → 127 ms |
/// | 500 | 543 → 734 ms | — | — |
///
/// A box test is six buffer reads and a little arithmetic, which is about what a sphere costs,
/// so a scene of single primitives pays for the guard and gets nothing back — and a scene of
/// bodies with several nodes each gets its frame back three to six times over. The weight is
/// the count of nodes under the body's root, with a curve-walking node counting for several,
/// since a pipe visits every segment of its spine per reading. Four is the first weight
/// measured at which the guard wins, and it wins there by a third to a factor of three; with
/// the policy in place the single-sphere rows read the same guarded as not.
const GUARD_WEIGHT: usize = 4;

/// What one node costs a reading, roughly, in primitives.
///
/// `walking_lattices` counts a lattice and a pattern as a walk as well. The box guard does not
/// need that reading -- a lattice is never a true distance, so the guard never asks about one --
/// and the operand masks do; see `exclusive_weight`.
fn node_weight(node: &Node, walking_lattices: bool) -> usize {
    match node {
        // A walk of every segment or sample per reading, against a primitive's few
        // operations: `sweep_cost` puts the walk alone at four times an extrude.
        Node::Pipe { .. }
        | Node::Sweep { .. }
        | Node::Loft { .. }
        | Node::Extrude { .. }
        | Node::Revolve { .. }
        | Node::Attract { .. } => 8,
        Node::Lattice { .. } | Node::Pattern { .. } if walking_lattices => 8,
        _ => 1,
    }
}

/// What a body costs to evaluate, roughly, in primitives.
fn body_weight(graph: &Graph, root: NodeId) -> usize {
    graph
        .subtree_of(root)
        .iter()
        .filter_map(|id| graph.node(*id))
        .map(|n| node_weight(n, false))
        .sum()
}

/// Six parameter slots holding the box a body's surface lies in, when a box may stand in for
/// the body — or `None`, and the body is evaluated at every reading as before.
///
/// **The guard is `box_distance(p, lo, hi) < best`, and it is exact only for a body whose
/// value is a true distance.** Outside its box such a body reads at least the distance to the
/// box, so a box no nearer than the running minimum cannot hold the nearest body and skipping
/// it leaves `min` unchanged. A lower-bound body — a lattice, a divided sweep, a driven morph —
/// may read *less* than the distance to its box, and then the guard would change the value the
/// fold returns: still a valid bound, but not the same number, and the loop spec holds the
/// shader to the same number. So those bodies are not guarded, decided by the same walk the
/// panel uses to say whether a blend is exact. An unbounded body has no box to guard with.
fn body_box_slots(
    document: &fieldform_scene::Document,
    graph: &Graph,
    root: NodeId,
    params: &mut Vec<ParamRef>,
) -> Option<usize> {
    if !fieldform_field::carries_true_distance(document, graph, root) {
        return None;
    }
    // Nor a body cheaper than the test -- see `GUARD_WEIGHT`.
    if body_weight(graph, root) < GUARD_WEIGHT {
        return None;
    }
    // Nor a body with a baked leaf in it. Outside its bake box a baked cage or mesh already
    // answers with the distance to that box -- a box five percent wider than its extent --
    // so the guard would gain nothing there and, being the tighter box, would read *larger*
    // than the body and skip it where the fold would have taken it: a different number in
    // the far field, and `gpu_matches_cpu` moved on the cage-and-pipe scene by exactly that.
    if graph
        .subtree_of(root)
        .iter()
        .any(|id| matches!(graph.node(*id), Some(Node::Cage(_) | Node::Mesh(_))))
    {
        return None;
    }
    let (lo, hi) = fieldform_field::graph_extent(document, graph, root).as_box()?;
    let pad = (hi - lo).norm() * BOX_GUARD_PAD;
    if !pad.is_finite() {
        return None;
    }
    let first = params.len();
    for v in [lo.x - pad, lo.y - pad, lo.z - pad, hi.x + pad, hi.y + pad, hi.z + pad] {
        if !v.is_finite() {
            params.truncate(first);
            return None;
        }
        params.push(ParamRef::Literal(v));
    }
    Some(first)
}

fn generate_inner(
    document: &fieldform_scene::Document,
    graph: &Graph,
    morph_divisors: &[(NodeId, f64)],
    guard_bodies: bool,
    cull_bodies: bool,
    prune_patterns: bool,
    mask_operands: bool,
    // How many pipes a union needs before it is walked rather than written out: `RUN_FLOOR`,
    // which is what every caller but a measurement passes.
    run_floor: usize,
) -> Result<ShaderProgram, CodegenError> {
    // **The rewrite the field reads first, read first here too** (backlog item 113), so a joined
    // mirror of a section touching its plane is one extrude in both and they cannot disagree
    // about which body is drawn. The caller resolves this program's sections against the same
    // rewrite (`TracedView::rebuild`), since a merged mirror's section is a curve only the
    // rewritten document holds.
    if let Some((merged, merged_graph)) = fieldform_field::with_shared_faces_merged(document, graph)
    {
        return generate_inner(
            &merged,
            &merged_graph,
            morph_divisors,
            guard_bodies,
            cull_bodies,
            prune_patterns,
            mask_operands,
            run_floor,
        );
    }
    // **A pattern is no longer lowered away.** It used to be rewritten into one placement per
    // instance, so the shader carried the body once per copy: a hundred copies emitted it a
    // hundred times, and this codebase has already met the end of that road, where two thousand
    // bodies simulated to a twenty-two-million-character expression. It is emitted directly now
    // and folds, like the lattice beside it and like the field it has to agree with -- see
    // `window_of`. Frames and shells are still lowered, because those are rewrites into things
    // that already exist rather than a cost that grows with a number somebody typed.
    // The same lowering the field runs, from the same function, so a turned block reaches both
    // paths as the identical composition -- including a block aimed at a plane or a line, which
    // needs the document to resolve and is the reason this is not the graph's own method.
    if graph.nodes().any(|(_, n)| {
        matches!(n, Node::Lattice { cell, map, .. }
            if cell.is_prism() && !matches!(map, fieldform_scene::CellMap::Rectangular { .. }))
    }) {
        return Err(CodegenError::CellCannotTakeThatMap);
    }
    if graph.nodes().any(|(_, n)| {
        matches!(n, Node::Lattice { cell, map, .. }
            if cell.is_stochastic() && matches!(map, fieldform_scene::CellMap::Conformal { .. }))
    }) {
        return Err(CodegenError::CellCannotTakeAConformalMap);
    }
    // Before any lowering, because `expand_frames` moves an aimed lattice's frame into a
    // placement and `NodeEmitter::lattice` could no longer tell.
    if graph.nodes().any(|(_, n)| {
        matches!(n, Node::Lattice { phase, frame, .. }
            if phase.param().driver().is_some() && !frame.is_default())
    }) {
        return Err(CodegenError::GradedLatticeCannotBeAimed);
    }
    let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
    let turned = fieldform_field::with_turns_flattened(document, graph, &numbers);
    let lowered = only_the_same_bodies(&turned, expand_shells(&expand_frames(document, &turned)));
    let graph = &lowered;
    // Same refusal as the CPU path, for the same reason: a node this build could not read
    // has no honest lowering, and emitting a shader that quietly leaves it out would put the
    // wrong shape on screen with nothing to notice. Checked over the whole graph, since an
    // unknown node's operands are exactly what could not be parsed.
    if let Some((_, node)) = graph.nodes().find(|(_, n)| matches!(n, Node::Unknown(_))) {
        let Node::Unknown(unknown) = node else { unreachable!() };
        return Err(CodegenError::UnreadableNode(unknown.to_string()));
    }
    // Seeded in the document's order, not the ids': see `Document::evaluation_order_of`. The
    // fold order below settles the body functions' names and this settles their contents.
    let order = document.evaluation_order_of(graph).ok_or(CodegenError::Cyclic)?;

    let mut cages: Vec<BakedLeaf> = Vec::new();
    let mut curves: Vec<CurveUse> = Vec::new();
    // **A looped run's bindings are made first, in the run's own order.** Both run loops read
    // binding `first + k`, so a run's bindings have to be one consecutive stretch — and left to
    // the emission order they are not even in the right order: the evaluation order reached a
    // two-pipe union's operands back to front, so the run's spines came out as bindings 1 and 0
    // and `run_first_curve` refused every run there was. Interning them here, before any node is
    // emitted, is what makes the stretch consecutive by construction rather than by luck — for a
    // fan of blades as much as for a union of pipes, since a profile is keyed by its extrude's
    // node and so cannot have been taken by anything else.
    //
    // In `order` rather than over `graph.nodes()`, because that is a map and D7 wants the same
    // graph to compile to the same bytes.
    for id in &order {
        if let Some(Node::Boolean { op, inputs }) = graph.node(*id) {
            if let Some(run) = looped_run(*op, graph, inputs, run_floor) {
                for binding in &run.curves {
                    intern(&mut curves, *binding);
                }
            }
        }
    }
    let mut surfaces: Vec<(fieldform_scene::QuadMeshId, u32)> = Vec::new();
    let mut params: Vec<ParamRef> = Vec::new();
    // `(node, point context)` to its SSA name index. Keyed by the pair rather than the node,
    // because a node reached through two different points is two expressions in the generated
    // code — which is what lets a body be unioned with its own reflection.
    let mut slot: HashMap<(NodeId, usize), usize> = HashMap::new();

    // A transform changes the point its subtree is sampled at, so the point is threaded
    // through the generated code as a chain of variables rather than being the single `p` the
    // function was called with. The declarations are all emitted up front: a node's expression
    // comes after its operands' in evaluation order, so a transform's own line arrives *after*
    // the subtree that needs its point, and the point could not be declared there. Each one
    // depends on nothing but its parent and some parameters, so the chain hoists whole.
    let plan = point_assignment(document, graph, &order)?;
    // **A part with a hole in it has no shader, and the refusal names the block to fill.**
    //
    // An empty slot is a legal state -- a block dragged out of one, or placed from the Create
    // menu before it has been told what to read (`notebook-spec.md` §7.2) -- and there is no
    // number a missing input could supply that keeps `distance` from overestimating. So nothing
    // is invented for one and nothing is quietly left out either: a body drawn without the block
    // it is built from is the wrong shape with nothing to notice.
    //
    // **Asked of the plan rather than of the graph**, which is what makes this the same line the
    // CPU compile draws: `incomplete` is closed downstream, so a hole under a body makes that
    // body incomplete too, and the plan holds exactly the nodes the program would be written
    // from -- its bodies and their subtrees, and the critical surfaces. A holed block *nothing*
    // reads, the ramp placed and not yet wired, is in neither, and generates nothing, as an
    // unwired ramp always did.
    //
    // **One check, before any line is written**, rather than a guard at each of the thirty-seven
    // operand reads: having got past this, every node the two passes walk has every operand, so
    // the reads stay the plain lookups they were and whoever adds the next kind of block has
    // nothing to remember. `source_of` says the same thing where it would otherwise be missed.
    //
    // Of the document rather than the graph, so a cage block whose cage is open is in it too
    // (R12): a red cage has no field to bake, exactly as an empty slot has no value.
    let incomplete = document.incomplete_in(graph);
    if let Some((met, _)) = plan.emit.iter().find(|(id, _)| incomplete.contains(id)) {
        if let Some(cage) = red_cage_behind(document, graph, *met) {
            return Err(CodegenError::OpenCage(cage));
        }
        return Err(empty_slot(graph, *met));
    }
    // The point chain, shared by every body: one line per context, each depending only on its
    // parent, so it hoists whole and is cheap to repeat wherever it is needed.
    let mut body = String::from("    let p0 = p;\n");
    // **Each node's line, kept apart rather than appended to one function.** They used to go
    // straight into `field`, which made the whole composition a single function -- and a driver
    // optimiser's cost is superlinear in the values live at once, so a scene of three or four
    // bodies took ten minutes to compile where one took half a second. See backlog item 34 and
    // `fieldform-gpu/examples/rebuild_cost.rs`; they are split into a function per body below.
    let mut node_lines: Vec<String> = Vec::new();
    // A pattern's subtree, emitted once as a function.
    let mut subtrees: Vec<String> = Vec::new();
    // Statements that have to run before a node's own line, flushed in front of it.
    //
    // Two kinds need this, for the same reason: what they compute is a *sequence* rather than
    // an expression. A pattern's window is a loop; a boolean's operands are folded one at a
    // time, so a list of any length costs a statement each instead of a nest of calls. Both
    // must land beside the node's line in the body function, because that is the only scope
    // where the operands they name exist -- the shared preamble above holds the point chain
    // and nothing else.
    let mut blocks_before: Vec<String> = Vec::new();
    // Keyed by the *context* a changer creates rather than by the changer itself, since one
    // node can create several — the same transform reached through two parents has two points
    // and two sets of parameters.
    let mut placement_slots: HashMap<usize, (String, usize)> = HashMap::new();
    // A mirror's plane normal, per declared point, since the normal pass has to reflect a
    // direction back out with the same plane the point was reflected in.
    let mut mirror_normals: HashMap<usize, String> = HashMap::new();
    // An attractor's strength slot, per declared point. The value pass divides by it, and it
    // has to be the *same* slot the point pass warped with -- warping by one number and
    // dividing by another is a field that overstates its distance.
    let mut attract_slots: HashMap<usize, (usize, usize)> = HashMap::new();
    // One frame per pattern occurrence, shared by every step of its window -- see the radial
    // arm of `PointEmitter::pattern` for why that matters to the shader compiler rather than
    // only to tidiness.
    let mut shared_frame: HashMap<(NodeId, usize), (String, String)> = HashMap::new();
    // What each pattern's loop needs, by the context standing for its current step.
    let mut pattern_loops: HashMap<usize, PatternLoop> = HashMap::new();
    // One pattern a program has its copies masked: see `COPY_MASK`.
    let mut mask_taken = false;
    // And up to thirty-two operands of its joins: see `OperandMasks`.
    let mut operand_masks = OperandMasks { on: mask_operands, used: 0, of: Vec::new() };
    for (index, (parent, node_id, offset)) in plan.contexts.iter().enumerate() {
        let point = PointEmitter {
            document,
            graph,
            prune_patterns,
            params: &mut params,
            curves: &mut curves,
            body: &mut body,
            placement_slots: &mut placement_slots,
            mirror_normals: &mut mirror_normals,
            attract_slots: &mut attract_slots,
            shared_frame: &mut shared_frame,
            pattern_loops: &mut pattern_loops,
            mask_taken: &mut mask_taken,
            index,
            parent,
            node_id,
            offset,
        };
        match graph.node(*node_id) {
            Some(Node::Transform { translation, rotation, scale, about, .. }) => {
                point.transform(translation, rotation, scale, about)
            }
            Some(Node::Twist { degrees_per_mm, .. }) => point.twist(degrees_per_mm),
            Some(Node::Attract { toward, strength, falloff, .. }) => {
                point.attract(toward, strength, falloff)
            }
            Some(Node::Mirror { frame, .. }) => point.mirror(frame),
            Some(Node::Pattern { input, kind, fillet, .. }) => point.pattern(input, kind, fillet),
            _ => unreachable!("point_assignment records the nodes that move a point"),
        }
    }

    for (id, ctx) in &plan.emit {
        let (id, ctx) = (*id, *ctx);
        let node = graph.node(id).ok_or(CodegenError::DanglingNode(id))?;
        let n = slot.len();
        let p = format!("p{ctx}");
        // Where this node's operands were emitted: the context it hands down if it changes the
        // point, otherwise its own.
        let inner = plan.inputs_of(id, ctx);
        let emit = NodeEmitter {
            document,
            graph,
            morph_divisors,
            run_floor,
            cages: &mut cages,
            curves: &mut curves,
            surfaces: &mut surfaces,
            params: &mut params,
            slot: &slot,
            body: &body,
            node_lines: &node_lines,
            subtrees: &mut subtrees,
            blocks_before: &mut blocks_before,
            placement_slots: &placement_slots,
            attract_slots: &attract_slots,
            pattern_loops: &pattern_loops,
            operand_masks: &mut operand_masks,
            node,
            id,
            ctx,
            n,
            p: &p,
            inner,
        };

        let expr = match node {
            Node::Cage(_)
            | Node::Mesh(_)
            | Node::Sphere { .. }
            | Node::Cuboid { .. }
            | Node::SkewedBox { .. }
            | Node::Cylinder { .. }
            | Node::Cone { .. }
            | Node::Torus { .. }
            | Node::Plane { .. }
            | Node::Point { .. }
            | Node::Capsule { .. } => emit.leaf(),
            Node::Lattice { cell, map, phase, frame: _ } => emit.lattice(cell, map, phase)?,
            Node::Extrude { outline, sweep, start, height, sides, draft, corners, rims } => {
                emit.extrude(outline, sweep, start, height, sides, draft, corners, rims)
            }
            Node::Revolve { outline, frame, angle, sides, rims } => {
                emit.revolve(outline, frame, angle, sides, rims)
            }
            Node::Pipe { curve, radius } => emit.pipe(curve, radius)?,
            Node::Morph { a, b, t } => emit.morph(a, b, t)?,
            Node::Boolean { op, inputs } => emit.boolean(op, inputs)?,
            Node::Blend { op, a, b, fillet } => emit.blend(op, a, b, fillet)?,
            Node::Offset { input, distance } => emit.offset(input, distance)?,
            Node::Ramp { input, .. } => emit.ramp(input)?,
            Node::Pattern { input, .. } => emit.pattern(input)?,
            Node::Mirror { input, join, .. } => emit.mirror(input, join)?,
            Node::Twist { input, .. } => emit.twist(input)?,
            Node::Attract { input, .. } => emit.attract(input)?,
            Node::Transform { input, .. } => emit.transform(input)?,
            Node::Shell { .. } => {
                unreachable!("lowered into a difference before any point was assigned")
            }
            Node::Sweep { outline, spine, section, law, rims } => {
                emit.sweep(outline, spine, section, law, rims)?
            }
            Node::Loft { sections, spine, smooth, rims } => emit.loft(sections, spine, smooth, rims)?,
            Node::Unknown(_) => unreachable!("refused in generate, before any lowering"),
        };

        for block in blocks_before.drain(..) {
            node_lines.push(block);
        }
        node_lines.push(format!("    let n{n} = {expr};\n"));
        slot.insert((id, ctx), n);
    }

    // A part's field is the union of its bodies, matching how `GraphField` folds them: one
    // body is itself, several are nested `min`s.
    //
    // **In the order the document says, not the order the ids fall in.** `split_into_bodies`
    // names each body function by its position here, so a body that changes rank is a body the
    // driver compiles again — and ranking by `NodeId` ranked by nothing, since ids are drawn at
    // random. A creation then recompiled bodies it had not touched, 67% to 83% of the time
    // (`fieldform-gpu/examples/body_order_probe.rs`). Asking the document instead means a body
    // already in the part keeps its function, so a graph that has not changed *and* a graph that
    // has only gained a body both leave the old bodies byte-identical.
    let bodies = document.folded_bodies(graph);
    let roots: Vec<String> = bodies
        .iter()
        .map(|id| operand(&slot, (*id, 0)))
        .collect::<Result<_, _>>()?;

    // **Bodies that cannot be nearest are not evaluated** (performance-plan-of-action.md step
    // 4). Every ray used to evaluate every body at every step, and on an assembly — or any
    // scene of parts apart — most of those evaluations were of bodies that could not be the
    // nearest. Each body with a finite box and a true-distance value gets six slots for the box,
    // and the fold below asks the box before the body. Pushed after every node's own slots, so
    // the normal pass, which counts slots by walking the nodes, is unaffected. A buffer value,
    // so a body that moves rewrites six numbers and never the shader — and a rebuild is what
    // every change to the document goes through, so the box is never staler than the values.
    //
    // Not for a lone body: the fold of one is the body itself, and there is no minimum for its
    // box to be compared against. Step 4's other half, the ray's start, is what a lone body
    // gets.
    // Read from `bodies` above rather than asked again: this list is indexed *alongside* the
    // fold, so the two being one list is the invariant, not a coincidence.
    let guard_bodies = guard_bodies && bodies.len() > 1;
    let boxes: Vec<Option<usize>> = bodies
        .iter()
        .map(|id| guard_bodies.then(|| body_box_slots(document, graph, *id, &mut params)).flatten())
        .collect();

    // **The cull's extents, after the guard boxes and contiguous** (§1.5). They ride the parameter
    // buffer for the reason the guard boxes do: an extent depends on the document's numbers, and
    // `set_params` changes numbers without a rebuild, so a cull reading extents uploaded once
    // would miss a body the user just grew -- which is missing material, not a slower frame.
    //
    // **One body is not worth culling and the cap is where the mask stops being cheap**, so
    // culling is on for 2 to `CULL_CAP` roots and off otherwise (§1.6, review point 8). Off, no
    // slot is pushed and the fold below is emitted byte-identical to what it was before any of
    // this existed, which is what `culling_off_generates_what_it_always_did` holds.
    let culling = cull_bodies && (2..=CULL_CAP).contains(&bodies.len());
    let cull_base = culling.then(|| {
        let first = params.len();
        for id in &bodies {
            cull_extent_slots(document, graph, *id, &mut params);
        }
        first
    });

    // **The gap to the critical surfaces, as a function of the world point.** Each marked node
    // was planned at context 0 above, so its value there is already a line; this is the closure
    // of those lines, less each mark's clearance. The clearance slots are pushed last of all --
    // after the boxes -- for the boxes' own reason: the normal pass counts slots by walking the
    // nodes, and it never reads these, since it calls the function this makes.
    let gap_function = if tapers(graph) {
        let terms: Vec<String> = graph
            .critical()
            .iter()
            .map(|mark| {
                let value = operand(&slot, (mark.node, 0))?;
                let c = params.len();
                params.push(mark.clearance);
                Ok(format!("abs({value}) - max(params[{c}], 0.0)"))
            })
            .collect::<Result<_, CodegenError>>()?;
        let mut folded = String::from("3.4e38");
        for term in &terms {
            folded = format!("min({folded}, {term})");
        }
        let inner = closure_of(&body, &node_lines, &folded, "");
        format!("fn critical_gap(p: vec3<f32>) -> f32 {{\n{inner}    return {folded};\n}}\n\n")
    } else {
        String::new()
    };
    // Reset first, so a critical surface that is itself a blended body is measured at its full
    // radii -- which is how the field compiles it, from its own subtree with no marks.
    let set_gap = if gap_function.is_empty() {
        ""
    } else {
        "    crit_gap = 3.4e38;\n    crit_gap = critical_gap(p);\n"
    };

    // **One function per body.** See `split_into_bodies` for why, and for what it costs.
    let (bodies_source, calls) = split_into_bodies(&body, &node_lines, &roots, "body", "f32");
    // Subtree functions first: WGSL wants a function declared before it is called.
    let functions = format!("{}{gap_function}{bodies_source}", subtrees.concat());
    let mut body = String::from(set_gap);
    let out = fold_union(&mut body, &calls, &boxes, cull_base).ok_or(CodegenError::Empty)?;
    let normal_body = normal_function(
        document, graph, &order, &cages, &curves, &boxes, cull_base, prune_patterns, run_floor,
    )?;
    // Only when something follows a surface. A program without one is built without a line of
    // the walk, which is the difference between a pipeline in hundreds of milliseconds and one in
    // seconds -- see `CONFORMAL`.
    let conformal = if surfaces.is_empty() { "" } else { CONFORMAL };
    // How many times a pattern's sub-field was called since it was last zeroed, for the
    // counting march. Declared only in a measuring build.
    let stat_counter = if cfg!(feature = "march-stats") {
        "var<private> stat_sub_calls: u32;\n"
    } else {
        ""
    };
    // **The per-tile cull's own declarations** (`tile-mask-design-01941c5.md` §1.5). Generated
    // rather than written in the prelude for the reason the fold is: the number of bodies is a
    // fact about *this* graph, and `tile_mask` has to be exactly as long as the bodies it
    // describes. `march.wgsl` is concatenated after this and reads these constants, which is how
    // one fixed cull pass serves every program.
    //
    // **The state is private and per-invocation, not a buffer read inside the fold.** A pixel's
    // tile is settled before it marches, so its mask is loaded once; asking the buffer inside
    // `field` would put a storage read in the innermost loop of the whole renderer to answer a
    // question that cannot change while the ray advances.
    //
    // `tile_live` false is the arrangement switched off, and nothing here sets it: a program
    // emitted today reads the same bodies it always did, because every `tile_ruled_out` answers
    // false. That is what makes the fold's new conjunct provably inert until the pass that fills
    // the mask exists.
    //
    // **Declared whether or not this program culls**, with `CULL_BODIES` nought when it does not
    // (review point 8). `march.wgsl` is one fixed text for every program, and the prologue that
    // loads a mask sits behind `if (CULL_BODIES > 0u)` -- a constant, so the driver removes the
    // whole branch rather than testing it per pixel. Leaving the constants out instead would mean
    // a second copy of the march for the programs that do not cull.
    //
    // What stays out when culling is off is the *fold's* conjunct, which is the thing a frame
    // pays for: `culling_off_generates_what_it_always_did` holds that. `MASK_WORDS` is one rather
    // than nought there, because WGSL has no zero-length array and naga strips the function that
    // would have read it.
    // **Whether a hit is settled onto its surface** (`Settle` in `march.wgsl`, backlog item 114):
    // only where a block can divide its reading. A constant, so a scene with none compiles without
    // it and pays nothing -- the settling state alone cost a hundred patterned revolves 6 percent
    // of their frame. Decided on structure (`may_divide`), not on whether anything divides now,
    // because that is a number, a drag moves it, and a drag must not recompile the shader.
    let settles = fieldform_field::may_divide(graph);
    // Whether any body reads a profile or a wall, which is the only thing a ray's exact finish
    // (`exact_reads`, backlog item 115) can land on: a program with none compiles without it.
    let has_outlines =
        curves.iter().any(|c| matches!(c, CurveUse::Profile(_) | CurveUse::Wall(..)));
    let cull_decls = [
        format!("const SETTLES: bool = {settles};"),
        format!("const HAS_OUTLINES: bool = {has_outlines};"),
        format!("const CULL_BODIES: u32 = {}u;", if culling { bodies.len() } else { 0 }),
        format!("const CULL_BASE: u32 = {}u;", cull_base.unwrap_or(0)),
        format!(
            "const MASK_WORDS: u32 = {}u;",
            if culling { bodies.len().div_ceil(32) } else { 1 }
        ),
        "var<private> tile_mask: array<u32, MASK_WORDS>;".to_string(),
        "var<private> tile_live: bool = false;".to_string(),
        "fn tile_ruled_out(i: u32) -> bool {".to_string(),
        "    if (!tile_live) { return false; }".to_string(),
        "    return (tile_mask[i / 32u] & (1u << (i % 32u))) != 0u;".to_string(),
        "}\n".to_string(),
    ]
    .join("\n");
    let prelude = prelude();
    let source = format!(
        "{prelude}\n\
         // ---- generated from the composition graph ----\n\
         //\n\
         // Structure only. Every number is a `params[..]` read, so changing one costs a buffer\n\
         // write rather than a recompile of this function.\n\
         // Which body was nearest at the last `field` call, for highlighting a selection.\n\
         // A module variable rather than a second function, because the alternative is\n\
         // emitting the whole composition twice: the answer is a by-product of the fold that\n\
         // has already run, and the marcher reads it right after the call that set it.\n\
         var<private> owner: u32;\n\
         {cull_decls}\
         {stat_counter}\
         \n\
         {functions}\n\
         fn field(p: vec3<f32>) -> f32 {{\n\
         {body}    return {out};\n\
         }}\n\
         \n\
         {normal_body}\n\
         {MARCH}\n\
         {conformal}"
    );

    // Before it leaves this module, because wgpu's answer to a shader it cannot compile is to
    // panic the process — so an error returned here is the difference between a message in the
    // status bar and the application disappearing.
    validated(&source)?;
    Ok(ShaderProgram {
        settles,
        size: ShaderSize {
            bytes: source.len(),
            bodies: roots.len(),
            nodes: graph.nodes().count(),
            // `p0` is the world's own point and is declared by every program, so a graph that
            // changes no point reads 1 rather than 0.
            deepest_point_chain: 1 + plan
                .contexts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    // Walk up the parents. Bounded by the list's own length, because a context's
                    // parent is always one declared before it.
                    let (mut at, mut depth) = (i, 1usize);
                    while let Some((parent, _, _)) = plan.contexts.get(at) {
                        if *parent == 0 {
                            break;
                        }
                        at = parent - 1;
                        depth += 1;
                    }
                    depth
                })
                .max()
                .unwrap_or(0),
        },
        source,
        baked: cages,
        curves,
        surfaces,
        params,
        masked: operand_masks.of,
        body_boxes: boxes,
        cull_base,
    })
}

/// What one point declaration of [`generate_inner`] is made from: the state its first loop
/// keeps, lent for one context, and where that context stands.
///
/// **Lent for the one call rather than held for the loop**, so the loop's state stays the loop's
/// between one context and the next, and each emitter reaches only what it is handed. `parent`
/// and `node_id` are the loop's own references to its plan.
struct PointEmitter<'a> {
    document: &'a fieldform_scene::Document,
    graph: &'a Graph,
    prune_patterns: bool,
    params: &'a mut Vec<ParamRef>,
    curves: &'a mut Vec<CurveUse>,
    body: &'a mut String,
    placement_slots: &'a mut HashMap<usize, (String, usize)>,
    mirror_normals: &'a mut HashMap<usize, String>,
    attract_slots: &'a mut HashMap<usize, (usize, usize)>,
    shared_frame: &'a mut HashMap<(NodeId, usize), (String, String)>,
    pattern_loops: &'a mut HashMap<usize, PatternLoop>,
    mask_taken: &'a mut bool,
    /// Where this context is in `plan.contexts`: it declares `p{index + 1}`.
    index: usize,
    /// The context it is derived from, the node that derives it, and which step of that node's
    /// window it is.
    parent: &'a usize,
    node_id: &'a NodeId,
    offset: &'a isize,
}

// `needless_borrow` is allowed because these bodies were the loop's arms, written against the
// owned locals the view now lends, and were moved unchanged so the move could be read line for
// line: `&mut params` borrows a reference again, which the compiler undoes.
#[allow(clippy::needless_borrow)]
impl PointEmitter<'_> {
    fn transform(
        self,
        translation: &[ParamRef; 3],
        rotation: &[ParamRef; 3],
        scale: &ParamRef,
        about: &fieldform_scene::Place,
    ) {
        let PointEmitter {
            document, graph, mut params, body, placement_slots, index, parent, node_id, ..
        } = self;
        let t = push_point(&mut params, translation);
        let euler = push_point(&mut params, rotation);
        let s = params.len();
        params.push(*scale);
        // Three more, counted in the order the slot pass counts them.
        let pivot = pivot_slots(&mut params, graph, document, *node_id, about);
        body.push_str(&format!(
            "    let p{} = to_local_about(p{parent}, {t}, {euler}, params[{s}], \
             {pivot});\n",
            index + 1,
        ));
        placement_slots.insert(index + 1, (euler, s));
    }

    // The frame is gone by here: `expand_frames` turned an aimed twist into an
    // unaimed one under a placement, so this winds about local Z through the local
    // origin and the placement carries it wherever the frame pointed.
    fn twist(self, degrees_per_mm: &ParamRef) {
        let PointEmitter { mut params, body, index, parent, .. } = self;
        let at = push_point(&mut params, &[ParamRef::Literal(0.0); 3]);
        let ax = params.len();
        params.push(ParamRef::Literal(2.0));
        let r = params.len();
        params.push(*degrees_per_mm);
        // Degrees on the wire and radians in the shader, converted here so the value
        // the user typed is what the parameter buffer carries — a buffer write should
        // not need a unit conversion to be read back.
        body.push_str(&format!(
            "    let p{} = unwind_point(p{parent}, radians(params[{r}]), \
             params[{ax}], {at});\n",
            index + 1,
        ));
    }

    // The second warp, and the simpler of the two to place: an attractor has no frame
    // to be aimed by, because the curve it names already says where it is and which way
    // it runs. So there is none of the sandwich a twist needs -- the curve is in world
    // space and so is the point that is about to be pushed away from it.
    fn attract(
        self,
        toward: &fieldform_scene::Attractor,
        strength: &ParamRef,
        falloff: &ParamRef,
    ) {
        let PointEmitter { params, mut curves, body, attract_slots, index, parent, .. } = self;
        let c = intern(&mut curves, attractor_use(*toward));
        let st = params.len();
        params.push(*strength);
        let fo = params.len();
        params.push(*falloff);
        attract_slots.insert(index + 1, (st, fo));
        body.push_str(&format!(
            "    let p{} = attract_point({c}u, p{parent}, params[{st}], \
             params[{fo}]);\n",
            index + 1,
        ));
    }

    fn mirror(self, frame: &fieldform_scene::Frame) {
        let PointEmitter {
            document, graph, mut params, body, mirror_normals, index, parent, node_id, ..
        } = self;
        let (at, normal) = frame_params(&mut params, document, graph, frame);
        let line = match fieldform_field::mirror_one_side(document, graph, *node_id) {
            // Folded: the body is all on one side, so this point and the one the `min`
            // would have compared it with are the same point.
            Some(side) => format!(
                "    let p{} = mirror_fold(p{parent}, {at}, {normal}, {side:?});\n",
                index + 1,
            ),
            None => format!(
                "    let p{} = mirror_point(p{parent}, {at}, {normal});\n",
                index + 1,
            ),
        };
        body.push_str(&line);
        mirror_normals.insert(index + 1, normal);
    }

    // One step of a pattern's window. The fold finds which instance a point belongs
    // to by arithmetic rather than by trying them all, and the offset walks the few
    // cells either side that could still hold the nearest material -- see `window_of`.
    // **Nothing is emitted here.** The step point is built inside the loop that walks
    // the window, and the subtree below is a function taking it as its parameter -- so
    // this records the ingredients and `NodeEmitter::pattern` assembles them.
    fn pattern(
        self,
        input: &Operand,
        kind: &fieldform_scene::PatternKind,
        fillet: &fieldform_scene::Fillet,
    ) {
        let PointEmitter {
            document,
            graph,
            prune_patterns,
            mut params,
            shared_frame,
            pattern_loops,
            mask_taken,
            index,
            parent,
            node_id,
            offset,
            ..
        } = self;
        // **Every number the options change, resolved once** from the pattern's layout --
        // where the copies are, which of them exist -- and handed to the loop as slots, so
        // the field and both passes of the shader read one answer.
        let numbers = pattern_numbers(document, graph, *node_id);
        let n = f64::from(numbers.layout.total);
        let _ = *offset;
        // Pushed before the window's own slots and read by both passes, so the two
        // count parameters in step with each other whether or not there is a fillet.
        let rounding = (!fillet.is_plainly_hard()).then(|| {
            let at = params.len();
            params.push(fillet.radius);
            at
        });
        // The prune's five, pushed here by both passes so they count in step. Literals
        // rewritten on every rebuild, as the body guard's box is: a body that moves
        // writes five numbers and never the shader.
        let prune = if prune_patterns {
            pattern_prune_floor(document, graph, input)
        } else {
            Err("switched off for a measurement".to_string())
        }
        .map(|(c, rho, d)| {
            let at = params.len();
            for v in [c[0], c[1], c[2], rho, d] {
                params.push(ParamRef::Literal(v));
            }
            at
        });
        // The mask's one slot, pushed here by both passes so they count in step.
        let mask = (masks_copies(rounding.is_some(), *parent) && !*mask_taken).then(|| {
            *mask_taken = true;
            let at = params.len();
            // Never below one: every leaf here reports one, and a bound that is too
            // small is a copy masked out of a cell it reaches.
            let rate = fieldform_field::operand_lipschitz(document, graph, input).max(1.0);
            params.push(ParamRef::Literal(if rate.is_finite() { rate } else { 1e30 }));
            at
        });
        let exponent = fillet.exponent();
        let span = window_of(document, graph, *node_id, &numbers.layout)
            .end()
            .max(&0)
            .to_owned() as usize;
        let whole = 2 * span + 1 >= n as usize;
        let reach = params.len();
        params.push(ParamRef::Literal(span as f64));
        match kind {
            fieldform_scene::PatternKind::Linear { .. } if numbers.grid.is_some() => {
                let g = numbers.grid.as_ref().expect("guarded");
                let vec3 = |params: &mut Vec<ParamRef>, v: &[f64; 3]| {
                    push_point(
                        params,
                        &[ParamRef::Literal(v[0]), ParamRef::Literal(v[1]), ParamRef::Literal(v[2])],
                    )
                };
                let anchor = vec3(&mut params, &g.anchor);
                let disp0 = vec3(&mut params, &g.disp0);
                let e1 = vec3(&mut params, &g.e1);
                let e2 = vec3(&mut params, &g.e2);
                let d1 = vec3(&mut params, &g.d1);
                let d2 = vec3(&mut params, &g.d2);
                let cn = params.len();
                params.push(ParamRef::Literal(n));
                let n1 = params.len();
                params.push(ParamRef::Literal(g.columns));
                let n2 = params.len();
                params.push(ParamRef::Literal(g.rows));
                let r1 = params.len();
                params.push(ParamRef::Literal(g.reach[0] as f64));
                let r2 = params.len();
                params.push(ParamRef::Literal(g.reach[1] as f64));
                let fi = params.len();
                params.push(ParamRef::Literal(numbers.first));
                let extras = push_pattern_extras(&mut params, &numbers);
                pattern_loops.insert(
                    index + 1,
                    PatternLoop {
                        parent: *parent,
                        reach,
                        whole,
                        rounding,
                        exponent,
                        prune: prune.clone(),
                        mask,
                        first: fi,
                        left_out: extras.left_out,
                        skip: extras.skip,
                        point: PatternPoint::Grid {
                            anchor,
                            disp0,
                            e1,
                            e2,
                            d1,
                            d2,
                            count: cn,
                            columns: n1,
                            rows: n2,
                            reach: [r1, r2],
                            whole: [
                                2 * g.reach[0] + 1 >= g.columns as usize,
                                2 * g.reach[1] + 1 >= g.rows as usize,
                            ],
                        },
                    },
                );
            }
            fieldform_scene::PatternKind::Linear { .. } => {
                let u = push_point(
                    &mut params,
                    &[
                        ParamRef::Literal(numbers.unit[0]),
                        ParamRef::Literal(numbers.unit[1]),
                        ParamRef::Literal(numbers.unit[2]),
                    ],
                );
                let pi = params.len();
                params.push(ParamRef::Literal(numbers.pitch));
                let cn = params.len();
                params.push(ParamRef::Literal(n));
                let og = params.len();
                params.push(ParamRef::Literal(numbers.origin));
                let fi = params.len();
                params.push(ParamRef::Literal(numbers.first));
                let extras = push_pattern_extras(&mut params, &numbers);
                pattern_loops.insert(
                    index + 1,
                    PatternLoop {
                        parent: *parent,
                        reach,
                        whole,
                        rounding,
                        exponent,
                        prune: prune.clone(),
                        mask,
                        first: fi,
                        left_out: extras.left_out,
                        skip: extras.skip,
                        point: PatternPoint::Linear {
                            unit: u,
                            pitch: pi,
                            count: cn,
                            origin: og,
                            bias: extras.bias,
                        },
                    },
                );
            }
            fieldform_scene::PatternKind::Radial { frame } => {
                // **The same slots for every step of one pattern's window.** Each step
                // used to push its own copy of the frame, so the identical work at the
                // top of `pattern_point_radial` -- building a rotation from three Euler
                // angles -- read different buffer entries in each and the shader
                // compiler could not prove them equal. Written once, the expressions
                // are textually identical and it can hoist them, which turns a window
                // of five frame decodes into one.
                let (at, euler) = shared_frame
                    .entry((*node_id, *parent))
                    .or_insert_with(|| {
                        frame_placement(&mut params, graph, document, frame)
                    })
                    .clone();
                let cn = params.len();
                params.push(ParamRef::Literal(n));
                let og = params.len();
                params.push(ParamRef::Literal(numbers.origin));
                let fi = params.len();
                params.push(ParamRef::Literal(numbers.first));
                let extras = push_pattern_extras(&mut params, &numbers);
                pattern_loops.insert(
                    index + 1,
                    PatternLoop {
                        parent: *parent,
                        reach,
                        whole,
                        rounding,
                        exponent,
                        prune: prune.clone(),
                        mask,
                        first: fi,
                        left_out: extras.left_out,
                        skip: extras.skip,
                        point: PatternPoint::Radial {
                            at,
                            euler,
                            count: cn,
                            step: extras.step,
                            origin: og,
                            bias: extras.bias,
                            arc: extras.arc,
                            is_arc: numbers.arc,
                        },
                    },
                );
            }
        }
    }
}

/// What one node's line in [`generate_inner`] is made from: the state its second loop keeps,
/// lent for one node, and where that node stands.
///
/// **Lent for the one call rather than held for the walk**, because the walk reads some of that
/// state itself between nodes -- how many lines there are, to name the next one, and the blocks
/// and lines it flushes after each -- and those reads stay in the walk, where they were.
struct NodeEmitter<'a> {
    document: &'a fieldform_scene::Document,
    graph: &'a Graph,
    morph_divisors: &'a [(NodeId, f64)],
    run_floor: usize,
    cages: &'a mut Vec<BakedLeaf>,
    curves: &'a mut Vec<CurveUse>,
    surfaces: &'a mut Vec<(fieldform_scene::QuadMeshId, u32)>,
    params: &'a mut Vec<ParamRef>,
    slot: &'a HashMap<(NodeId, usize), usize>,
    body: &'a str,
    node_lines: &'a [String],
    subtrees: &'a mut Vec<String>,
    blocks_before: &'a mut Vec<String>,
    placement_slots: &'a HashMap<usize, (String, usize)>,
    attract_slots: &'a HashMap<usize, (usize, usize)>,
    pattern_loops: &'a HashMap<usize, PatternLoop>,
    operand_masks: &'a mut OperandMasks,
    /// The node, and the context it is emitted in.
    node: &'a Node,
    id: NodeId,
    ctx: usize,
    /// Its line's number: the line declares `n{n}`.
    n: usize,
    /// The point it is read at: `p{ctx}`.
    p: &'a str,
    /// Where its operands were emitted: see [`PointPlan::inputs_of`].
    inner: usize,
}

// `needless_borrow` is allowed for the reason it is on `PointEmitter`'s methods.
#[allow(clippy::needless_borrow)]
impl NodeEmitter<'_> {
    // The leaves, which read their own numbers and never an operand: small enough that a
    // method each would be mostly signature, so they share one and match the node again.
    fn leaf(self) -> String {
        let NodeEmitter { document, graph, mut cages, mut params, node, id, p, .. } = self;
        match node {
            Node::Cage(cage) => {
                let i = intern(&mut cages, BakedLeaf::Cage(*cage));
                format!("sample_baked({i}u, {p})")
            }
            Node::Mesh(mesh) => {
                let i = intern(&mut cages, BakedLeaf::Mesh(*mesh));
                format!("sample_baked({i}u, {p})")
            }
            Node::Sphere { centre, radius } => {
                let c = place_slots(&mut params, graph, document, id, centre);
                let r = params.len();
                params.push(*radius);
                format!("sphere_distance({p}, {c}, params[{r}])")
            }
            // The size is three slots read as a vec3, the same shape as a centre — which is
            // what `push_point` already is, once its name is read as what it does.
            Node::Cuboid { centre, size, edge, .. } => {
                let c = place_slots(&mut params, graph, document, id, centre);
                let s = push_point(&mut params, size);
                // **A hard edge is the call it always was**; a cut one is its `_edge_` form with
                // the reach one slot more, last, and the shape's exponent a literal, as a rim's.
                match edge_vector(edge, &mut params) {
                    None => format!("cuboid_distance({p}, {c}, {s})"),
                    Some(e) => format!("cuboid_edge_distance({p}, {c}, {s}, {e})"),
                }
            }
            // Twelve slots: the centre as a place, then each edge as a vec3. The edges are always
            // literal-or-parameter numbers and never a place, so the normal pass counts them
            // without asking.
            Node::SkewedBox { centre, edges } => {
                let c = place_slots(&mut params, graph, document, id, centre);
                let e0 = push_point(&mut params, &edges[0]);
                let e1 = push_point(&mut params, &edges[1]);
                let e2 = push_point(&mut params, &edges[2]);
                format!("skewed_box_distance({p}, {c}, {e0}, {e1}, {e2})")
            }
            Node::Cylinder { ends, radius, edge } => {
                let (a, b) = push_ends(&mut params, document, graph, id, ends);
                let r = params.len();
                params.push(*radius);
                match edge_vector(edge, &mut params) {
                    None => format!("cylinder_distance({p}, {a}, {b}, params[{r}])"),
                    Some(e) => format!("cylinder_edge_distance({p}, {a}, {b}, params[{r}], {e})"),
                }
            }
            Node::Cone { ends, base_radius, top_radius, edge } => {
                let (a, b) = push_ends(&mut params, document, graph, id, ends);
                let r = params.len();
                params.push(*base_radius);
                params.push(*top_radius);
                let args = format!("{p}, {a}, {b}, params[{r}], params[{}]", r + 1);
                match edge_vector(edge, &mut params) {
                    None => format!("cone_distance({args})"),
                    Some(e) => format!("cone_edge_distance({args}, {e})"),
                }
            }
            Node::Torus { centre, major_radius, minor_radius, .. } => {
                let c = place_slots(&mut params, graph, document, id, centre);
                let r = params.len();
                params.push(*major_radius);
                params.push(*minor_radius);
                format!("torus_distance({p}, {c}, params[{r}], params[{}])", r + 1)
            }
            Node::Plane { origin, flipped, .. } => {
                let o = place_slots(&mut params, graph, document, id, origin);
                let f = params.len();
                params.push(ParamRef::Literal(if *flipped { -1.0 } else { 1.0 }));
                format!("plane_distance({p}, {o}, params[{f}])")
            }
            Node::Point { at } => {
                let c = push_point(&mut params, at);
                format!("point_distance({p}, {c})")
            }
            Node::Capsule { ends, start_radius, end_radius } => {
                let (a, b) = push_ends(&mut params, document, graph, id, ends);
                let r = params.len();
                params.push(*start_radius);
                params.push(*end_radius);
                format!("capsule_distance({p}, {a}, {b}, params[{r}], params[{}])", r + 1)
            }
            _ => unreachable!("the walk sends a leaf here and nothing else"),
        }
    }

    // Twelve slots: the cell, the map tag, that map's three parameters, the phase, and
    // the one number the phase is sized by. The three in the middle mean different
    // things under each map — one cell size, or a radial size, an axial size and a
    // count — which is the asymmetry the cell/map split exists to allow, and the last
    // one means different things under each phase for the reason `LatticePhase` gives.
    //
    // **The sizing number goes across as what the user typed, not as an isovalue.** A
    // density could be converted here and pushed as a literal, which would keep the
    // per-cell table out of the shader entirely — and would freeze it: a density driven
    // by a named parameter would then need a recompile to change, where every other
    // number in this list needs a buffer write. So the table is transcribed into WGSL
    // instead, and `the_shader_reads_a_density_through_the_same_table_the_cpu_does`
    // holds the two copies together.
    //
    // The frame is gone by here: `expand_lattice_frames` turned an aimed lattice
    // into an unaimed one under a transform, so nothing below has to know about it.
    fn lattice(
        self,
        cell: &fieldform_scene::LatticeCell,
        map: &fieldform_scene::CellMap,
        phase: &LatticePhase,
    ) -> Result<String, CodegenError> {
        let NodeEmitter { document, graph, mut surfaces, params, slot, id, ctx, p, .. } = self;
        Ok({
            let c = params.len();
            params.push(ParamRef::Literal(lattice_cell_index(*cell)));
            match map {
                fieldform_scene::CellMap::Rectangular { cell } => {
                    params.push(ParamRef::Literal(0.0));
                    params.push(*cell);
                    params.push(ParamRef::Literal(0.0));
                    params.push(ParamRef::Literal(0.0));
                }
                fieldform_scene::CellMap::Cylindrical {
                    cell_radius,
                    cell_height,
                    arc_count,
                } => {
                    params.push(ParamRef::Literal(1.0));
                    params.push(*cell_radius);
                    params.push(*cell_height);
                    params.push(ParamRef::Literal(*arc_count as f64));
                }
                fieldform_scene::CellMap::Spherical {
                    cell_radius,
                    azimuth_count,
                    polar_count,
                } => {
                    params.push(ParamRef::Literal(2.0));
                    params.push(*cell_radius);
                    params.push(ParamRef::Literal(*azimuth_count as f64));
                    params.push(ParamRef::Literal(*polar_count as f64));
                }
                // **Which surface travels as a number in a slot, like everything else.**
                // That is what keeps the normal pass counting seven parameters without
                // knowing the map, and it means a conformal lattice recompiles for the same
                // reasons every other node does and no others. The depth is a real
                // `ParamRef`, so it can be driven by a named parameter; the other two
                // describe the tiling's structure and are literals.
                fieldform_scene::CellMap::Conformal {
                    surface,
                    cells_per_face,
                    depth,
                    thickness: _,
                } => {
                    let index = intern(&mut surfaces, (*surface, *cells_per_face));
                    params.push(ParamRef::Literal(3.0));
                    params.push(ParamRef::Literal(index as f64));
                    params.push(ParamRef::Literal(*cells_per_face as f64));
                    params.push(*depth);
                }
            }
            params.push(ParamRef::Literal(lattice_phase_index(phase)));
            // **A driven phase is an expression, not a slot.** The slot stays and holds
            // nothing, so the normal pass still counts eight without asking which; the
            // number itself is the ramp's value at the point, written where the slot was
            // read, which is all a graded lattice is on this side -- `lattice_distance`
            // always took its sizing as an argument.
            let driver = phase.param().driver();
            params.push(match driver {
                Some(_) => ParamRef::Literal(0.0),
                None => *phase.param(),
            });
            let sizing = match driver {
                Some(driver) => format!("({})", operand(&slot, (driver, ctx))?),
                None => format!("params[{}]", c + 6),
            };
            // The field's own divisor, asked of the same graph-level function.
            let divisor = finite(
                fieldform_field::driven_lattice_divisor(document, graph, id),
                "a graded lattice's bound",
            )?;
            // **The shell's thickness, or a negative for no shell.** A shader has no
            // `Option`, and a thickness is a length — so a number that cannot be one says
            // "not confined" without a second slot to carry the question.
            //
            // The *whole* thickness, as typed, halved by the shader at the point of use.
            // The field halves it once at compile time instead, which is cheaper there and
            // would be two rules here: a literal could be halved in this slot and a named
            // parameter could not, and a slot that means half a number sometimes is the
            // kind of thing that is only ever found by drawing it wrong.
            params.push(match map {
                fieldform_scene::CellMap::Conformal { thickness: Some(t), .. } => *t,
                _ => ParamRef::Literal(-1.0),
            });
            // A sheet's mid-surface offset, and nought for a phase that has none: the ninth
            // slot, last so that nothing before it moved.
            params.push(phase.offset().copied().unwrap_or(ParamRef::Literal(0.0)));
            // **The cell's own two numbers**, and the first slots any cell has had of its
            // own. Pushed for every cell rather than only the two that read them, so both
            // passes count twelve without asking which cell it is -- the arrangement the
            // conformal arm's comment already relies on.
            //
            // The seed is halved into two sixteen-bit slots because a slot is an `f32` and
            // cannot carry 32 bits of integer exactly. `voronoi_seed` folds the node's
            // stored `u64` down to those 32 bits, once, on this side.
            params.push(cell.jitter().copied().unwrap_or(ParamRef::Literal(0.0)));
            let folded = cell.seed().map(fieldform_field::voronoi_seed).unwrap_or(0);
            params.push(ParamRef::Literal((folded & 0xFFFF) as f64));
            params.push(ParamRef::Literal((folded >> 16) as f64));
            // **By name, for a conformal map.** Routing it through `lattice_distance` meant
            // a runtime branch on the map tag, which put the surface walk into every lattice
            // shader in every document — see `CONFORMAL`. The slots are the same eight
            // either way, so the normal pass counts them without asking which.
            let value = if matches!(map, fieldform_scene::CellMap::Conformal { .. }) {
                format!(
                    "conformal_lattice_distance({p}, params[{c}], u32(params[{}]),                          params[{}], params[{}], params[{}], {sizing}, params[{}], params[{}])",
                    c + 2,
                    c + 3,
                    c + 4,
                    c + 5,
                    c + 7,
                    c + 8,
                )
            } else {
                format!(
                    "lattice_distance({p}, params[{c}], params[{}], params[{}], params[{}],                          params[{}], params[{}], {sizing}, params[{}], params[{}], params[{}],                          params[{}])",
                    c + 1,
                    c + 2,
                    c + 3,
                    c + 4,
                    c + 5,
                    c + 8,
                    c + 9,
                    c + 10,
                    c + 11,
                )
            };
            match driver {
                Some(_) => format!("({value}) / {divisor:?}"),
                None => value,
            }
        })
    }

    // The frame is applied at the call site with `to_local`, not by a placement above
    // the node. The profile's points are uploaded already in that frame, so the two
    // have to be put in the same coordinates *here* — and doing it inline keeps the
    // node's own id, which is what the uploaded polyline is keyed by.
    #[allow(clippy::too_many_arguments)]
    fn extrude(
        self,
        outline: &fieldform_scene::Outline,
        sweep: &fieldform_scene::Sweep,
        start: &ParamRef,
        height: &ParamRef,
        sides: &fieldform_scene::Sides,
        draft: &ParamRef,
        corners: &fieldform_scene::Corners,
        rims: &fieldform_scene::Rims,
    ) -> String {
        let NodeEmitter { document, graph, mut curves, mut params, id, p, .. } = self;
        // What is read in the plane: a profile's one polyline, or a wall's three. Bound
        // before anything is pushed, as a profile's always was.
        let bound: Vec<usize> = match outline {
            fieldform_scene::Outline::Closed { .. } => {
                vec![intern(&mut curves, CurveUse::Profile(id))]
            }
            fieldform_scene::Outline::Wall { .. } => WALL_PARTS
                .iter()
                .map(|part| intern(&mut curves, CurveUse::Wall(id, *part)))
                .collect(),
        };
        let c = bound[0];
        // **Bound before the placement's slots**, so the normal pass can find it by
        // asking the same question rather than by counting.
        let mitres = miters_binding(
            &mut curves,
            id,
            *corners,
            extrude_is_upright(document, draft, sides),
        );
        let (at, euler, dir) = extrude_placement(&mut params, document, outline, sweep);
        let depth = depth_slots(&mut params, document, start, height, draft, sides);
        // **A hard rim is the call it always was, byte for byte**; a rim is the `_rim_` call
        // with one more argument, its reach a slot so dragging it refills a buffer and its shape
        // a literal, as a blend's is. Pushed last of all the extrude's slots, after a wall's four
        // where there are four, which is the order the normal pass counts them in.
        let square = extrude_rims_are_square(document, sweep, draft, sides);
        let rim = |params: &mut Vec<ParamRef>| -> Option<String> {
            (!rims.is_plainly_hard()).then(|| {
                let r = params.len();
                params.push(rims.radius.clone());
                params.extend(rims.fillet().second_leg().cloned());
                rim_vector(r, rims, square)
            })
        };
        match (outline, mitres) {
            (fieldform_scene::Outline::Closed { .. }, None) => match rim(&mut params) {
                None => format!(
                    "extrude_distance({c}u, to_local({p}, {at}, {euler}, 1.0), {dir}, {depth})"
                ),
                Some(rim) => format!(
                    "extrude_rim_distance({c}u, to_local({p}, {at}, {euler}, 1.0), {dir}, \
                     {depth}, {rim})"
                ),
            },
            (fieldform_scene::Outline::Closed { .. }, Some(m)) => match rim(&mut params) {
                None => format!(
                    "extrude_sharp_distance({c}u, {m}u, to_local({p}, {at}, {euler}, 1.0), \
                     {dir}, {depth})"
                ),
                Some(rim) => format!(
                    "extrude_sharp_rim_distance({c}u, {m}u, to_local({p}, {at}, {euler}, \
                     1.0), {dir}, {depth}, {rim})"
                ),
            },
            // **Four more slots, and every one of them a number rather than a branch**:
            // the thickness, which way the band lies, whether the ends are round and
            // whether the chain closes. So dragging a thickness, flipping a side or
            // dragging two ends together refills a buffer and rebuilds nothing — the
            // last of those is not even an edit to this block.
            (fieldform_scene::Outline::Wall { .. }, mitres) => {
                let t = wall_slots(&mut params, document, graph, id, outline);
                let rim = rim(&mut params);
                // The rimmed call is the plain one renamed, with the rim appended.
                let rimmed = |call: String, name: &str| match &rim {
                    None => call,
                    Some(rim) => format!(
                        "{name}_rim_distance{}, {rim})",
                        call.strip_prefix(&format!("{name}_distance"))
                            .and_then(|args| args.strip_suffix(')'))
                            .expect("the call is the name and its arguments")
                    ),
                };
                match mitres {
                    // A wall round a closed loop, mitred: the loop's own displaced
                    // reading, folded about the band. An open wall never gets here --
                    // a draft on one has no inside to push out from and is refused.
                    Some(m) => rimmed(
                        format!(
                            "wall_extrude_sharp_distance({c}u, {m}u, to_local({p}, {at}, \
                             {euler}, 1.0), {dir}, {depth}, vec4<f32>(params[{t}], \
                             params[{}], params[{}], params[{}]))",
                            t + 1,
                            t + 2,
                            t + 3,
                        ),
                        "wall_extrude_sharp",
                    ),
                    None => rimmed(
                        format!(
                            "wall_extrude_distance({c}u, {}u, {}u, to_local({p}, {at}, \
                             {euler}, 1.0), {dir}, {depth}, params[{t}], params[{}], \
                             params[{}], params[{}])",
                            bound[1],
                            bound[2],
                            t + 1,
                            t + 2,
                            t + 3,
                        ),
                        "wall_extrude",
                    ),
                }
            }
        }
    }

    // **A full turn is the text it always was; anything less is the wedge's call** with
    // two more slots, where the wedge begins and how far it turns, both in radians. A
    // partial revolve whose angle is dragged refills a buffer; dragging it through a
    // full turn rebuilds, which is the price of keeping every existing revolve's shader
    // byte for byte.
    fn revolve(
        self,
        outline: &fieldform_scene::Outline,
        frame: &fieldform_scene::Frame,
        angle: &ParamRef,
        sides: &fieldform_scene::AngleSides,
        rims: &fieldform_scene::Rims,
    ) -> String {
        let NodeEmitter { document, graph, mut curves, mut params, id, p, morph_divisors, .. } = self;
        // **The rim's slots after the wedge's two**, and only on a wedge: a full revolve has no
        // ends, and the compile refuses rims on one, so the plain call stands for it. The
        // second slot is the square-corner scale the compile worked out (`Rim::cut`).
        let scale = morph_divisors.iter().find(|(node, _)| *node == id).map(|(_, s)| *s);
        match outline {
            fieldform_scene::Outline::Closed { .. } => {
                let c = intern(&mut curves, CurveUse::Profile(id));
                let (at, euler) = frame_placement(&mut params, graph, document, frame);
                match wedge_slots(&mut params, document, graph, id, angle, sides) {
                    None => format!("revolve_distance({c}u, to_local({p}, {at}, {euler}, 1.0))"),
                    Some(w) => rimmed_revolve_call(
                        format!(
                            "revolve_wedge_distance({c}u, to_local({p}, {at}, {euler}, 1.0), \
                             params[{w}], params[{}])",
                            w + 1
                        ),
                        rims,
                        scale,
                        &mut params,
                    ),
                }
            }
            // A thin revolve: the wall's three polylines where the profile's one was, its
            // four numbers after the frame's six, and one more saying whether the band
            // reaches across the axis — a number like the rest, since which side of the
            // axis a thickness leaves a band on changes as the thickness is dragged.
            fieldform_scene::Outline::Wall { .. } => {
                let bound: Vec<usize> = WALL_PARTS
                    .iter()
                    .map(|part| intern(&mut curves, CurveUse::Wall(id, *part)))
                    .collect();
                let (at, euler) = frame_placement(&mut params, graph, document, frame);
                let t = wall_slots(&mut params, document, graph, id, outline);
                let mirrored = params.len();
                params.push(ParamRef::Literal(wall_crosses_the_axis(document, graph, id)));
                match wedge_slots(&mut params, document, graph, id, angle, sides) {
                    None => format!(
                        "wall_revolve_distance({}u, {}u, {}u, to_local({p}, {at}, {euler}, \
                         1.0), {}, params[{mirrored}])",
                        bound[0],
                        bound[1],
                        bound[2],
                        wall_numbers_as_a_vector(t),
                    ),
                    Some(w) => rimmed_revolve_call(
                        format!(
                            "wall_revolve_wedge_distance({}u, {}u, {}u, to_local({p}, {at}, \
                             {euler}, 1.0), {}, params[{mirrored}], params[{w}], params[{}])",
                            bound[0],
                            bound[1],
                            bound[2],
                            wall_numbers_as_a_vector(t),
                            w + 1,
                        ),
                        rims,
                        scale,
                        &mut params,
                    ),
                }
            }
        }
    }

    fn pipe(self, curve: &CurveId, radius: &ParamRef) -> Result<String, CodegenError> {
        let NodeEmitter { document, graph, mut curves, params, slot, id, ctx, p, .. } = self;
        Ok({
            let c = intern(&mut curves, CurveUse::Spine(*curve));
            match radius.driver() {
                None => {
                    let r = params.len();
                    params.push(*radius);
                    format!("pipe_distance({c}u, {p}, params[{r}])")
                }
                // **The offset it is.** A pipe of a varying radius is the distance to its
                // own spine less that radius, so this is the driven offset's expression
                // with `pipe_distance(..., 0.0)` as the body — the same arithmetic and the
                // same divisor, read from the same graph-level function the field reads.
                //
                // The radius is an expression now, not a number in the buffer; the one
                // parameter pushed is the divisor, exactly as a driven offset pushes its
                // own, and the normal pass counts it the same way.
                Some(driver) => {
                    let amount = operand(&slot, (driver, ctx))?;
                    let divisor = finite(
                        fieldform_field::driven_pipe_divisor(document, graph, id),
                        "a varying pipe radius's bound",
                    )?;
                    // **A slot, not a literal.** The bound depends on the curve's shape, so
                    // written into the source it made every step of dragging a spine's point
                    // a different shader: 1.1 to 1.2 s of pipeline each, measured in the
                    // application, against 6 ms when the source is one the driver has seen.
                    let d = params.len();
                    params.push(ParamRef::Literal(divisor));
                    format!(
                        "((pipe_distance({c}u, {p}, 0.0)) - ({amount})) / params[{d}]"
                    )
                }
            }
        })
    }

    // **No point context.** Both operands are read at the point the morph was asked
    // at, so nothing enters `plan_points` and nothing hands a point down — which is
    // what makes this the simplest node added in a while.
    //
    // `t` is a parameter, so dragging it writes four bytes and redraws. The divisor is
    // 1 while `t` is constant, which is the whole of phase 1.
    fn morph(self, a: &Operand, b: &Operand, t: &ParamRef) -> Result<String, CodegenError> {
        let NodeEmitter {
            document, graph, morph_divisors, params, slot, id, ctx, inner, ..
        } = self;
        Ok({
            let (va, vb) = (operand(&slot, (*a, inner))?, operand(&slot, (*b, inner))?);
            // Either a parameter or the ramp's own expression, exactly as a driven offset
            // lowers its distance. A constant `t` stays a buffer write; a driven one is a
            // ramp, and a ramp's numbers are constants here, so editing it rebuilds.
            let amount = match t.driver() {
                None => {
                    let i = params.len();
                    params.push(*t);
                    format!("params[{i}]")
                }
                Some(driver) => operand(&slot, (driver, ctx))?,
            };
            // The same number the field divides by, from the same function — the two
            // computing it separately is a surface in two places.
            let divisor = finite(
                morph_divisors
                    .iter()
                    .find(|(node, _)| *node == id)
                    .map(|(_, d)| *d)
                    .unwrap_or_else(|| {
                        fieldform_field::driven_morph_divisor(document, graph, id)
                    }),
                "a driven morph's bound",
            )?;
            format!("(mix({va}, {vb}, clamp({amount}, 0.0, 1.0)) / {divisor:?})")
        })
    }

    fn boolean(self, op: &BooleanOp, inputs: &[Operand]) -> Result<String, CodegenError> {
        let NodeEmitter {
            document,
            graph,
            run_floor,
            curves,
            mut params,
            slot,
            body,
            node_lines,
            mut subtrees,
            mut blocks_before,
            operand_masks,
            id,
            n,
            p,
            inner,
            ..
        } = self;
        Ok({
            // **A run of one kind of block is one loop, not a call each** — see `looped_run`
            // for which operands qualify and `pipe_run_distance` for why it is worth it. The
            // run stands in the list where its first member stood, so the fold's order is
            // still the order the graph holds, and the operands around it are untouched.
            let run = looped_run(*op, graph, inputs, run_floor);
            // Each entry is the operand's name and the node behind it — `None` for the run,
            // which has no node of its own and is **not masked**. A mask is a function per
            // operand and a bit per operand, and a run has collapsed exactly that distinction:
            // its members are one call now, so there is nothing per-operand left to skip. The
            // two are alternatives, and `union_loop_cost.rs` measures what each is worth.
            let mut named: Vec<(String, Option<NodeId>)> = Vec::new();
            if let Some((run, first)) = run.as_ref().zip(run_first_curve(&curves, run.as_ref()))
            {
                for input in inputs.iter().take(run.at) {
                    named.push((operand(&slot, (*input, inner))?, Some(source_of(*input))));
                }
                // **The run's own slots, not the ones its members pushed.** Each member is
                // still emitted in its own right — the emission plan drives that, and nothing
                // here can tell it not to — so each has already taken its slots. Its *line*
                // then goes unreferenced and `closure_of` drops it, which is what shrinks the
                // source; its slots stay, so a run takes `2n + 1` slots where the calls took
                // `n` for pipes, `24n + 1` against `12n` for extrudes and `32n + 1` against
                // `16n` for thin ones. Reusing the members' slots would mean requiring those
                // to be contiguous as well, for a few dozen bytes each on a list that is tens
                // long. The source was the thing that cost 123 seconds, not the buffer.
                let base = params.len();
                match &run.kind {
                    RunKind::Pipes(radii) => {
                        for r in radii {
                            params.push(r.clone());
                        }
                    }
                    // Twelve a member, in the order the single-extrude arm pushes them —
                    // `extrude_placement`'s nine, then the height, the draft's *tangent* and
                    // the symmetric flag — so the loop steps by twelve and reads each at the
                    // offset that arm reads it at. Through `extrude_placement` itself rather
                    // than beside it: the plane is worked out from the profile's points, and
                    // two functions working it out is a surface in two places.
                    //
                    // A thin extrude is those twelve and then its wall's four, through the
                    // function the single arm pushes them with.
                    RunKind::Extrudes(members) | RunKind::Walls(members) => {
                        for member in members {
                            let Some(Node::Extrude { outline, sweep, start, height, sides, draft, corners: _, rims: _ }) =
                                graph.node(*member)
                            else {
                                unreachable!("`looped_run` admits only extrudes here")
                            };
                            extrude_placement(&mut params, document, outline, sweep);
                            depth_slots(&mut params, document, start, height, draft, sides);
                            if outline.is_wall() {
                                wall_slots(&mut params, document, graph, *member, outline);
                            }
                        }
                    }
                }
                let count = params.len();
                params.push(ParamRef::Literal(run.len() as f64));
                // The one thing the two passes cannot check against each other at run time,
                // asserted where it is still cheap: the normal pass advances by `slots` and
                // reads nothing, so a kind that pushed a different number here would shift
                // every parameter after this boolean and nothing would say so.
                debug_assert_eq!(
                    params.len() - base,
                    run.slots(),
                    "the value pass pushed slots the normal pass will not count"
                );
                // **Bound to a name, not pasted in.** Each masked operand beside the run
                // tests whether it is dominated by every *other* operand, and the run is one
                // of those — so written as its call the whole walk reappears inside every
                // neighbour's guard, which is the inlining a run exists to remove. Three
                // masked pipes beside a fan of blades put four copies of the loop in one
                // body; named, there is one. The normal pass has always done this, as
                // `pr{n}`, and the value pass was the half that had not.
                let (walk, _) = run.kind.walkers();
                blocks_before.push(format!(
                    "    let r{n} = {walk}({first}u, {base}u, u32(params[{count}]), {p});\n"
                ));
                named.push((format!("r{n}"), None));
                for input in inputs.iter().skip(run.at + run.len()) {
                    named.push((operand(&slot, (*input, inner))?, Some(source_of(*input))));
                }
            } else {
                for input in inputs.iter() {
                    named.push((operand(&slot, (*input, inner))?, Some(source_of(*input))));
                }
            }
            let width = named.len();
            // Each operand as the fold should read it: a lone operand is the fold itself
            // and has nothing to be dominated by.
            let masked: Vec<JoinOperand> = named
                .into_iter()
                .enumerate()
                .map(|(i, (name, input))| {
                    if width < 2 {
                        return JoinOperand { name, rate: 1.0, bit: None };
                    }
                    // A run: already one call, so there is no per-operand skip left to take.
                    // A rate of one is what `record_operands` needs from it to price the
                    // operands beside it, and it is the true figure for either kind — every
                    // leaf in this kernel reports 1.0, which `graph_lipschitz` states and a
                    // pipe and an extrude both are.
                    let Some(input) = input else {
                        return JoinOperand { name, rate: 1.0, bit: None };
                    };
                    // What the fold ignores: a `min` the very large, a `max` the very
                    // small -- and a difference negates everything after its first.
                    let sentinel = match (op, i) {
                        (BooleanOp::Union, _) => "1e30",
                        (BooleanOp::Intersection, _) | (BooleanOp::Difference, 0) => "-1e30",
                        (BooleanOp::Difference, _) => "1e30",
                    };
                    operand_masks.operand(
                        document, graph, id, input, name, inner, sentinel, &body,
                        &node_lines, &mut subtrees, &mut blocks_before,
                    )
                })
                .collect();
            let terms: Vec<(String, f64, Option<u32>)> = masked
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    let term = match (op, i) {
                        (BooleanOp::Difference, i) if i > 0 => format!("-({})", o.name),
                        _ => o.name.clone(),
                    };
                    (term, o.rate, o.bit)
                })
                .collect();
            record_operands(n, &mut blocks_before, &terms, matches!(op, BooleanOp::Union), None);
            let operands: Vec<String> = masked.into_iter().map(|o| o.name).collect();
            let mut block = String::new();
            let folded = fold_boolean(&mut block, *op, &format!("acc{n}"), &operands)
                .ok_or(CodegenError::NothingToCombine(id))?;
            if !block.is_empty() {
                blocks_before.push(block);
            }
            folded
        })
    }

    fn blend(
        self,
        op: &BooleanOp,
        a: &Operand,
        b: &Operand,
        fillet: &fieldform_scene::Fillet,
    ) -> Result<String, CodegenError> {
        let NodeEmitter {
            document,
            graph,
            params,
            slot,
            body,
            node_lines,
            mut subtrees,
            mut blocks_before,
            operand_masks,
            id,
            n,
            inner,
            ctx,
            ..
        } = self;
        Ok({
            let (name_a, name_b) = (operand(&slot, (*a, inner))?, operand(&slot, (*b, inner))?);
            // **The radius: a slot, or the ramp that drives it** (`round-chamfer-spec.md` §9.1),
            // lowered exactly as a driven offset's distance is -- the ramp's own expression read
            // at this point, and the one slot pushed is the divisor both paths get from
            // `driven_blend_divisor`. One slot either way, at the same position, so the normal
            // pass counts it the same. The mask's floor wants a number that holds over a whole
            // cell, which for a ramp is the most it can give; a ramp's numbers are generated
            // constants already, so a literal here changes nothing about when a rebuild happens.
            let mut second_slot = None;
            let (k, k_most, divided_by) = match fillet.radius.driver() {
                None => {
                    let r = params.len();
                    params.push(fillet.radius.clone());
                    // A chamfer's second leg, a slot after the radius's (`round-chamfer-spec.md`
                    // §9.2): read by scaling the operands below, the most either leg reaches
                    // for the mask's floor.
                    match fillet.second_leg() {
                        Some(second) => {
                            params.push(second.clone());
                            second_slot = Some(r + 1);
                            (
                                format!("params[{r}]"),
                                format!("max(params[{r}], params[{}])", r + 1),
                                None,
                            )
                        }
                        None => (format!("params[{r}]"), format!("params[{r}]"), None),
                    }
                }
                Some(driver) => {
                    let amount = operand(&slot, (driver, ctx))?;
                    let divisor = finite(
                        fieldform_field::driven_blend_divisor(document, graph, id),
                        "a varying fillet radius's bound",
                    )?;
                    let most = graph
                        .node(driver)
                        .map(|ramp| {
                            let (_, _, out_lo, out_span, _) = ramp_constants(document, ramp);
                            out_lo.max(out_lo + out_span).max(0.0)
                        })
                        .unwrap_or(0.0);
                    let d = params.len();
                    params.push(ParamRef::Literal(divisor));
                    (format!("({amount})"), format!("{most:?}"), Some(d))
                }
            };
            // Every form below is a smooth minimum of `x` and `y`, the operands negated or
            // not; what it ignores is a very large `x` or `y`, so an operand that goes in
            // negated stands in as very small.
            let (negate_a, negate_b) = match op {
                BooleanOp::Union => (false, false),
                BooleanOp::Difference => (true, false),
                BooleanOp::Intersection => (true, true),
            };
            let sentinel = |negated: bool| if negated { "-1e30" } else { "1e30" };
            let oa = operand_masks.operand(
                document, graph, id, source_of(*a), name_a, inner, sentinel(negate_a), &body,
                &node_lines, &mut subtrees, &mut blocks_before,
            );
            let ob = operand_masks.operand(
                document, graph, id, source_of(*b), name_b, inner, sentinel(negate_b), &body,
                &node_lines, &mut subtrees, &mut blocks_before,
            );
            let term = |o: &JoinOperand, negated: bool| {
                (if negated { format!("-({})", o.name) } else { o.name.clone() }, o.rate, o.bit)
            };
            record_operands(
                n,
                &mut blocks_before,
                &[term(&oa, negate_a), term(&ob, negate_b)],
                true,
                Some(&k_most),
            );
            let (a, b) = (oa.name, ob.name);
            let n = fillet.exponent();
            // A chamfer of two legs is the equal-legged one over scaled operands, as the field
            // reads it (`two_legs` in the compile): the slot after the radius's is the second
            // leg, and `k` becomes the lesser of the two.
            let (a, b, k) = match second_slot {
                Some(sb) => {
                    let m = format!("min({k}, params[{sb}])");
                    (
                        format!("(({a}) * {m} / params[{sb}])"),
                        format!("(({b}) * {m} / {k})"),
                        m,
                    )
                }
                None => (a, b, k),
            };
            // Each blended boolean is the smooth minimum of possibly-negated operands,
            // exactly as the CPU path composes it -- and divided as the CPU path divides
            // it, so the march never reads more room than there is over a shared face.
            // See `blend_divisor` in the prelude.
            let raw = match op {
                BooleanOp::Union => format!("smooth_min({a}, {b}, {k}, {n:?})"),
                BooleanOp::Difference => {
                    format!("-smooth_min(-({a}), {b}, {k}, {n:?})")
                }
                BooleanOp::Intersection => {
                    format!("-smooth_min(-({a}), -({b}), {k}, {n:?})")
                }
            };
            let bounded = if tapers(graph) {
                // The same three forms over the tapering minimum, divided by what tapering
                // costs. The ratio is asked of the field crate so both paths hold one number.
                let rho = finite(
                    fieldform_field::taper_ratio(document, graph, id),
                    "a tapering blend's bound",
                )?;
                let raw = raw.replacen("smooth_min(", "smooth_min_tapering(", 1);
                format!("({raw} / blend_taper_divisor({k}, {n:?}, {rho:?}))")
            } else {
                // Bounded, not divided: `blend_bound` in the prelude, and beside its
                // namesake in fieldform-field the reason. Nothing to compound down a chain.
                raw.replacen("smooth_min(", "blend_bound(", 1)
            };
            // And over the driven divisor where the radius varies, as the field divides.
            match divided_by {
                Some(d) => format!("(({bounded}) / params[{d}])"),
                None => bounded,
            }
        })
    }

    // One subtraction, and the same one the CPU path performs — there is no shader
    // helper because there is nothing to help with. See [`Node::Offset`].
    //
    // Unless the distance varies, in which case the ramp driving it is another node in
    // this same shader, already emitted, and the result is divided by the bound both
    // paths get from `driven_offset_divisor` — the number that keeps this a lower
    // bound rather than a claim of more room than there is.
    fn offset(self, input: &Operand, distance: &ParamRef) -> Result<String, CodegenError> {
        let NodeEmitter { document, graph, params, slot, id, ctx, inner, .. } = self;
        Ok({
            let body = operand(&slot, (*input, inner))?;
            match distance.driver() {
                None => {
                    let d = params.len();
                    params.push(*distance);
                    format!("{body} - params[{d}]")
                }
                Some(driver) => {
                    let amount = operand(&slot, (driver, ctx))?;
                    let divisor = finite(
                        fieldform_field::driven_offset_divisor(document, graph, id),
                        "a driven offset's bound",
                    )?;
                    // A slot for the reason a driven pipe's is: see there.
                    let d = params.len();
                    params.push(ParamRef::Literal(divisor));
                    format!("(({body}) - ({amount})) / params[{d}]")
                }
            }
        })
    }

    // Its input is sampled at this node's own point, since a ramp does not move
    // anything — it reads.
    // Its four numbers are generated constants rather than buffer parameters, and
    // that is deliberate: a driven offset's divisor is computed from them at generation
    // time, so a range that could change without a rebuild would leave the divisor
    // stale — and a divisor that is too small is a tracer stepping through a wall.
    // Tying them together means editing a ramp recompiles, which is correct.
    fn ramp(self, input: &Operand) -> Result<String, CodegenError> {
        let NodeEmitter { document, slot, node, inner, .. } = self;
        Ok({
            let (lo, span, out_lo, out_span, shape) = ramp_constants(document, node);
            format!(
                "ramp({}, {lo:?}, {span:?}, {out_lo:?}, {out_span:?}, {shape:?})",
                // Read like any other operand: a part holding a ramp with an empty slot is
                // refused by name before any line is written, so this arm cannot meet one. A
                // hole is no longer this block's business -- every slot can be one now, and
                // one refusal upstream answers them all.
                operand(&slot, (*input, inner))?,
            )
        })
    }

    // The nearest instance in the window, which is the union of the copies that could
    // be nearest -- the same `min` the field folds, over the same window.
    //
    // **Walked at run time rather than written out.** Each step used to be its own
    // context, so the whole subtree below was emitted once per step: the code multiplied
    // by the window while the shape did not, and one patterned body took a minute to
    // compile. The subtree is now a function and the window is a loop that calls it, so
    // the window costs one copy of the code and N calls. Measured in backlog item 34.
    fn pattern(self, input: &Operand) -> Result<String, CodegenError> {
        let NodeEmitter {
            document, graph, slot, body, node_lines, subtrees, blocks_before, pattern_loops, id,
            n, inner, ..
        } = self;
        Ok({
            let info = pattern_loops.get(&inner).expect("a pattern declares its own context");
            let root = operand(&slot, (*input, inner))?;
            // The subtree, once, against the step point the caller hands in. Its own
            // context becomes the parameter, so every line below it works unchanged.
            let inner_body =
                closure_of(&body, &node_lines, &root, &format!("    let p{inner} = p;\n"));
            subtrees.push(format!(
                "fn pattern_sub_{n}(p: vec3<f32>) -> f32 {{\n{inner_body}    return {root};\n}}\n\n"
            ));
            match info.rounding {
                // The hard fold, unchanged: one running minimum and nothing else.
                None => {
                    let walk = info.walk(&n.to_string(), Some(&format!("pv{n}")), |q| {
                        format!("        pv{n} = min(pv{n}, pattern_sub_{n}({q}));\n")
                    });
                    // While the skip grid is being filled, say which copies this cell
                    // keeps: every one within `2·L·h` of the minimum just found. Every
                    // copy is asked, by index, whatever the loop above walked -- a window
                    // is the copies that matter at one point, and a cell is many points.
                    let record = match info.mask {
                        Some(rate) => format!(
                            "    if (mask_mode == 1u) {{\n\
                             \x20       let keep{n} = pv{n} + mask_slack * params[{rate}];\n\
                             \x20       for (var m{n} = i32(params[{first}]); \
                             m{n} < min(i32(params[{count}]), {COPY_MASK}); m{n} = m{n} + 1) {{\n\
                             {skip_m}\
                             \x20           if (pattern_sub_{n}({point}) <= keep{n}) {{\n\
                             \x20               mask_kept = mask_kept | (1u << u32(m{n}));\n\
                             \x20           }}\n\
                             \x20       }}\n\
                             \x20       mask_seen = true;\n\
                             \x20   }}\n",
                            first = info.first,
                            count = info.point.count(),
                            point = info.point.instance_at(info.parent, &format!("f32(m{n})")),
                            skip_m = match info.skip {
                                Some(slot) => format!(
                                    "        if (pattern_skipped({slot}u, m{n})) {{ continue; }}\n"
                                ),
                                None => String::new(),
                            },
                        ),
                        None => String::new(),
                    };
                    blocks_before.push(format!("    var pv{n} = 1e30;\n{walk}{record}"));
                    format!("pv{n}")
                }
                // **The n-ary smooth minimum, accumulated exactly as the field accumulates
                // it.** `max(k, min aᵢ) − (Σ (k−aᵢ)₊ⁿ)^(1/n)` — the sum runs over every
                // instance in the window at once rather than blending them in pairs, which
                // would be order-dependent where three copies meet. Two running numbers.
                Some(r) => {
                    let e = info.exponent;
                    let inv = 1.0 / e;
                    let walk = info.walk(&n.to_string(), Some(&format!("max(pv{n}, params[{r}])")), |q| {
                        format!(
                            "        let d{n} = pattern_sub_{n}({q});\n\
                             \x20       pv{n} = min(pv{n}, d{n});\n\
                             \x20       pi{n} = pi{n} + pow(max(params[{r}] - d{n}, 0.0), \
                             {e:?});\n"
                        )
                    });
                    // Divided by W^(1/n) exactly as the field divides it; see
                    // `pattern_join_divisor` in the prelude for why the window's size.
                    let (walk, block) = if tapers(graph) {
                        let rho = finite(
                            fieldform_field::taper_ratio(document, graph, id),
                            "a tapering pattern's bound",
                        )?;
                        // The radius this point gets, read once ahead of the loop.
                        let k = format!("params[{r}]");
                        (
                            walk.replace(&k, &format!("ke{n}")),
                            format!(
                                "    let ke{n} = tapered({k});\n    var pv{n} = 1e30;\n    \
                                 var pi{n} = 0.0;\n{{walk}}    \
                                 let pr{n} = (max(ke{n}, pv{n}) - pow(pi{n}, {inv:?})) \
                                 / {};\n",
                                info.taper_divisor(r, rho)
                            ),
                        )
                    } else {
                        let divisor = info.join_divisor(r);
                        (
                            walk,
                            format!(
                                "    var pv{n} = 1e30;\n    var pi{n} = 0.0;\n{{walk}}    \
                                 let pr{n} = (max(params[{r}], pv{n}) - pow(pi{n}, {inv:?})) \
                                 / {divisor};\n"
                            ),
                        )
                    };
                    blocks_before.push(block.replace("{walk}", &walk));
                    format!("pr{n}")
                }
            }
        })
    }

    // The point already changed for the subtree below, and a reflection preserves
    // distance — so unlike a transform there is nothing left to undo.
    // Joined, the answer is the nearer of the body and its reflection — the union
    // that used to be a second block. The input appears at two contexts, which is
    // exactly what `point_assignment` records for it.
    fn mirror(self, input: &Operand, join: &bool) -> Result<String, CodegenError> {
        let NodeEmitter { document, graph, slot, id, ctx, inner, .. } = self;
        Ok({
            let reflected = operand(&slot, (*input, inner))?;
            // A folded mirror has no second reading to take the smaller of: the point was
            // put on the body's side, which is the side the `min` would have chosen.
            let folded =
                fieldform_field::mirror_one_side(document, graph, id).is_some();
            if *join && !folded {
                format!("min({}, {reflected})", operand(&slot, (*input, ctx))?)
            } else {
                reflected
            }
        })
    }

    // Divided by how far the winding can stretch material, which is what keeps the
    // value a lower bound rather than an overstated distance.
    //
    // **In a slot, not in the source.** The factor comes from the body's own extent and
    // from the rate, so it moves whenever either does — and written as a literal it made
    // the rate the one number in a block panel that could not be dragged: every frame of
    // the drag was a source the driver had never seen, and a cold compile is measured in
    // seconds (`wing_compile_cost`). A sweep's bend was the same and was moved to a slot
    // for the same reason; `draggable_numbers` is what says which numbers are still
    // like this, and the twist was the last one.
    fn twist(self, input: &Operand) -> Result<String, CodegenError> {
        let NodeEmitter { document, graph, params, slot, id, inner, .. } = self;
        Ok({
            let stretch = finite(twist_stretch(document, graph, id), "a twist's stretch")?;
            let s = params.len();
            params.push(ParamRef::Literal(stretch));
            format!("({} / params[{s}])", operand(&slot, (*input, inner))?)
        })
    }

    // Divided by a parameter, as the twist above it now is. An attractor's divisor is
    // `1 + strength` and nothing else, where a twist's is recomputed from the body's
    // extent -- but both are worked out here and uploaded, so dragging either writes a
    // buffer instead of rebuilding the shader, which is what makes the preview live
    // (D4).
    fn attract(self, input: &Operand) -> Result<String, CodegenError> {
        let NodeEmitter { slot, attract_slots, inner, .. } = self;
        Ok({
            let (st, _) = attract_slots[&inner];
            format!(
                "({} / attract_stretch(params[{st}]))",
                operand(&slot, (*input, inner))?
            )
        })
    }

    // The point already changed for the subtree below; what is left here is scaling
    // the distance back out, the other half of what keeps this a true distance rather
    // than the unscaled body's.
    fn transform(self, input: &Operand) -> Result<String, CodegenError> {
        let NodeEmitter { slot, placement_slots, inner, .. } = self;
        Ok({
            let (_, s) = &placement_slots[&inner];
            format!("params[{s}] * {}", operand(&slot, (*input, inner))?)
        })
    }

    // **Three bindings and no placement.** The spine and its frame are in world
    // coordinates, so the point is read as it is; the profile is read in the section
    // plane the frame spans, which is why its polyline goes up flattened like an
    // extrude's. The stretch is the leaf's own number asked of the graph, as a pipe's
    // divisor is -- `sweep_stretch` says why it is the same figure at either tolerance.
    fn sweep(
        self,
        outline: &fieldform_scene::Outline,
        spine: &CurveId,
        section: &fieldform_scene::Section,
        law: &fieldform_scene::SweepLaw,
        rims: &fieldform_scene::Rims,
    ) -> Result<String, CodegenError> {
        let NodeEmitter { document, graph, mut curves, mut params, id, p, .. } = self;
        Ok({
            // The section first, as it always was: a profile's one polyline, or a wall's
            // three.
            let bound: Vec<usize> = match outline {
                fieldform_scene::Outline::Closed { .. } => {
                    vec![intern(&mut curves, CurveUse::Profile(id))]
                }
                fieldform_scene::Outline::Wall { .. } => WALL_PARTS
                    .iter()
                    .map(|part| intern(&mut curves, CurveUse::Wall(id, *part)))
                    .collect(),
            };
            let c = bound[0];
            let s = intern(&mut curves, CurveUse::Spine(*spine));
            let f = intern(&mut curves, CurveUse::Frame(id));
            // A slot rather than a generated branch, as the extrude's `symmetric` is.
            let closed = params.len();
            params.push(ParamRef::Literal(
                if document.curves.get(spine).is_some_and(|c| c.is_closed()) { 1.0 } else { 0.0 },
            ));
            let stretch = finite(
                fieldform_field::sweep_stretch(document, outline, *spine, law),
                "a sweep's bend",
            )?;
            // A slot for the reason a driven pipe's divisor is: the bend is the spine's
            // shape, and this is the block whose spine people drag.
            let bend = params.len();
            params.push(ParamRef::Literal(stretch));
            // **And the tube the section reads truly in, beside it** -- the section's reach and
            // the spine's tube, both nought where the whole reading is divided
            // (`fieldform_field::sweep_tube`). Slots, so a spine dragged into or out of a tube
            // changes two numbers and not the shader; a straight spine's endless tube goes up as
            // a number too large to reach, since a shader may assume no infinities.
            let (reach, tube) = fieldform_field::sweep_tube(document, outline, *spine, section, law)
                .map(|(reach, tube)| (reach, tube.min(1e30)))
                .unwrap_or((0.0, 0.0));
            params.push(ParamRef::Literal(reach));
            params.push(ParamRef::Literal(tube));
            // **Where the solid sits: at the profile, not at the curve.** The spine gives
            // the path's shape and the profile gives where it starts, so the point is
            // carried back by the same offset `fieldform_field::sweep_offset` gives the
            // compile — three slots rather than three literals, because a profile and a
            // spine are both things people drag and a literal would recompile the shader
            // at the frame rate.
            let moved = fieldform_field::sweep_offset(document, outline, *spine)
                .unwrap_or_else(nalgebra::Vector3::zeros);
            let origin = params.len();
            for axis in 0..3 {
                params.push(ParamRef::Literal(moved[axis]));
            }
            // How the section is held: the profile's own two axes, and which of the three
            // ways to hold them. The shader decomposes them against its own frame, as the
            // kernel does against its own — see `fieldform_field::sweep_axes`.
            let (ua, vb) = fieldform_field::sweep_axes(document, outline, *spine)
                .unwrap_or((nalgebra::Vector3::x(), nalgebra::Vector3::y()));
            let axes = params.len();
            for axis in 0..3 {
                params.push(ParamRef::Literal(ua[axis]));
            }
            for axis in 0..3 {
                params.push(ParamRef::Literal(vb[axis]));
            }
            params.push(ParamRef::Literal(match section {
                fieldform_scene::Section::Square => 0.0,
                fieldform_scene::Section::Skewed => 1.0,
                fieldform_scene::Section::Parallel => 2.0,
            }));
            // What both kinds are told after their bindings: where, how it closes, how it
            // bends, and how the section is held.
            let carried = format!(
                "{p} - vec3<f32>(params[{origin}], \
                 params[{}], params[{}]), params[{closed}], \
                 vec3<f32>(params[{bend}], params[{}], params[{}]), \
                 vec3<f32>(params[{axes}], params[{}], params[{}]), \
                 vec3<f32>(params[{}], params[{}], params[{}]), params[{}]",
                origin + 1,
                origin + 2,
                bend + 1,
                bend + 2,
                axes + 1,
                axes + 2,
                axes + 3,
                axes + 4,
                axes + 5,
                axes + 6,
            );
            // **A sweep that has not asked compiles to the call it always was, byte for
            // byte**, as a full revolve does: the law is six more slots and a different
            // function, so setting one rebuilds the shader and dragging any of its numbers
            // refills a buffer. See `law_slots`.
            let under = law_slots(&mut params, document, outline, *spine, law);
            let call = match (outline, under) {
                (fieldform_scene::Outline::Closed { .. }, None) => {
                    format!("sweep_distance({s}u, {f}u, {c}u, {carried})")
                }
                (fieldform_scene::Outline::Closed { .. }, Some(law_call)) => {
                    // The miters beside the section, bound only when this law mitres --
                    // `law_mitres`, which the normal pass asks too. A law that does not
                    // reads the section itself and is handed its own binding back, which
                    // the shader then never looks at.
                    let m = match law_mitres(law) {
                        true => intern(&mut curves, CurveUse::Miters(id)),
                        false => c,
                    };
                    format!("sweep_law_distance({s}u, {f}u, {c}u, {m}u, {carried}, {law_call})")
                }
                // An open section: the wall's four numbers last, as on an extrude.
                (fieldform_scene::Outline::Wall { .. }, under) => {
                    let t = wall_slots(&mut params, document, graph, id, outline);
                    // **A mitred draft takes its own pair of entry points**, which bind
                    // the chain and the miters and not the mitres' two lists of bare
                    // edges: a shader that carried both readings inlined two polyline
                    // walks seven times over and the driver would not compile it.
                    match (under, law_mitres(law)) {
                        (Some(law_call), true) => {
                            let m = intern(&mut curves, CurveUse::Miters(id));
                            format!(
                                "wall_sweep_sharp_distance({s}u, {f}u, {c}u, {m}u, \
                                 {carried}, {}, {law_call})",
                                wall_numbers_as_a_vector(t),
                            )
                        }
                        (Some(law_call), false) => format!(
                            "wall_sweep_law_distance({s}u, {f}u, {c}u, {}u, {}u, {c}u, \
                             {carried}, {}, {law_call})",
                            bound[1],
                            bound[2],
                            wall_numbers_as_a_vector(t),
                        ),
                        (None, _) => format!(
                            "wall_sweep_distance({s}u, {f}u, {c}u, {}u, {}u, {carried}, {})",
                            bound[1],
                            bound[2],
                            wall_numbers_as_a_vector(t),
                        ),
                    }
                }
            };
            // The rim last of all the sweep's slots, whichever of the five calls it is.
            rimmed_call(call, rims, &mut params)
        })
    }

    // **Everything from one build** (`fieldform_field::loft_built`): the centerline as
    // the loft flattened it, its frame, every section projected square at its own
    // station, and the stations themselves. A derived centerline is no curve of the
    // document's, and a named one flattened here at another tolerance would put every
    // station somewhere else -- so the spine is keyed by the node, not the curve.
    //
    // **The stations and the bindings they read are a table in `params`**, ruled so by
    // the owner: a station moves whenever a section or the centerline is dragged, and
    // in the source it would rebuild the shader at the frame rate. The count is in the
    // source, since a section added is a structural edit and rebuilds anyway.
    fn loft(
        self,
        sections: &[fieldform_scene::Outline],
        spine: &Option<CurveId>,
        smooth: &fieldform_scene::LoftSmooth,
        rims: &fieldform_scene::Rims,
    ) -> Result<String, CodegenError> {
        let NodeEmitter { document, mut curves, params, id, p, .. } = self;
        Ok({
            let lofted = fieldform_field::loft_built(document, sections, *spine, *smooth)
                .ok_or(CodegenError::LoftCannotBeBuilt(id))?;
            let s = intern(&mut curves, CurveUse::LoftSpine(id));
            let f = intern(&mut curves, CurveUse::Frame(id));
            let table = params.len();
            for k in 0..sections.len() {
                let binding = intern(&mut curves, CurveUse::LoftSection(id, k));
                // The rails' slopes beside the section at G1; at G0 the slot names the
                // section again, so the layout is one layout and the toggle still refills
                // a buffer -- though it rebinds, so it rebuilds, which `replace_node`
                // treats as structural through `curves()`.
                let slopes = match smooth {
                    fieldform_scene::LoftSmooth::G1 => intern(&mut curves, CurveUse::LoftSlope(id, k)),
                    fieldform_scene::LoftSmooth::G0 => binding,
                };
                params.push(ParamRef::Literal(lofted.station(k)));
                params.push(ParamRef::Literal(binding as f64));
                params.push(ParamRef::Literal(slopes as f64));
            }
            // Slots, as a sweep's `closed` is: a toggle is a number somebody changes.
            let eased = params.len();
            params.push(ParamRef::Literal(match smooth {
                fieldform_scene::LoftSmooth::G0 => 0.0,
                fieldform_scene::LoftSmooth::G1 => 1.0,
            }));
            let bend = params.len();
            params.push(ParamRef::Literal(finite(lofted.stretch(), "a loft's bound")?));
            // A hard rim is the call it always was; a rim is the `_rim_` call with its reach a
            // slot after the bound and its exponent a literal, as the extrude's is.
            if rims.is_plainly_hard() {
                format!(
                    "loft_distance({s}u, {f}u, {p}, {table}u, {}u, params[{eased}], \
                     params[{bend}])",
                    sections.len(),
                )
            } else {
                let r = params.len();
                params.push(rims.radius.clone());
                params.extend(rims.fillet().second_leg().cloned());
                format!(
                    "loft_rim_distance({s}u, {f}u, {p}, {table}u, {}u, params[{eased}], \
                     params[{bend}], {})",
                    sections.len(),
                    rim_vector(r, rims, "0.0"),
                )
            }
        })
    }
}

/// A body's call with its rim: the call's `_rim_` twin with the rim's slot appended last, or
/// the call as it was for a hard rim, which pushes nothing -- so a solid without rims compiles
/// to the text it always did.
fn rimmed_call(call: String, rims: &fieldform_scene::Rims, params: &mut Vec<ParamRef>) -> String {
    if rims.is_plainly_hard() {
        return call;
    }
    let r = params.len();
    params.push(rims.radius.clone());
    params.extend(rims.fillet().second_leg().cloned());
    rim_call(&call, &rim_vector(r, rims, "0.0"))
}

/// A revolve's rimmed call: the reach's slot and then the square-corner scale's, a literal the
/// compile worked out from the section's distance from the axis (`morph_divisors`, which carries
/// every per-node number the shader must agree with the field on). Nought -- the bound -- when
/// no compile handed one over, as the generator's own tests do not.
fn rimmed_revolve_call(
    call: String,
    rims: &fieldform_scene::Rims,
    scale: Option<f64>,
    params: &mut Vec<ParamRef>,
) -> String {
    if rims.is_plainly_hard() {
        return call;
    }
    let r = params.len();
    params.push(rims.radius.clone());
    params.extend(rims.fillet().second_leg().cloned());
    params.push(ParamRef::Literal(scale.unwrap_or(0.0)));
    rim_call(&call, &rim_vector(r, rims, &format!("params[{}]", r + rim_slots(rims))))
}

/// A box's or a cylinder's edge as the prelude's `_edge_` functions take it -- the reach a slot
/// pushed now, the shape's exponent a literal -- or `None` for a hard one, which pushes nothing.
fn edge_vector(edge: &fieldform_scene::Fillet, params: &mut Vec<ParamRef>) -> Option<String> {
    if edge.is_plainly_hard() {
        return None;
    }
    let r = params.len();
    params.push(edge.radius);
    Some(format!("vec2<f32>(params[{r}], {:?})", edge.exponent() as f32))
}

/// A revolve's rim vector in the normal pass -- the two slots `rimmed_revolve_call` pushed,
/// counted rather than pushed -- or `None` for a hard rim, which pushed nothing.
fn edge_like_slot(rims: &fieldform_scene::Rims, params: &mut usize) -> Option<String> {
    if rims.is_plainly_hard() {
        return None;
    }
    let r = *params;
    *params += rim_slots(rims) + 1;
    Some(rim_vector(r, rims, &format!("params[{}]", r + rim_slots(rims))))
}

/// The same in the normal pass, which counts the slot the value pass pushed rather than pushing.
fn edge_slot(edge: &fieldform_scene::Fillet, params: &mut usize) -> Option<String> {
    if edge.is_plainly_hard() {
        return None;
    }
    let r = *params;
    *params += 1;
    Some(format!("vec2<f32>(params[{r}], {:?})", edge.exponent() as f32))
}

/// A rim as the prelude's functions take it: the reach from slot `r`, the shape's exponent and
/// which ends as literals -- 0 both, 1 the start, 2 the end, which `rim_join_ends` reads -- and
/// `square`, the text of the square-corner scale: `0.0` for the max-norm bound, `1.0` for an
/// upright extrude's exact corner, a slot for a revolve's (`Rim::cut`).
///
/// **A chamfer's second leg rides in the exponent's place, negated.** A chamfer's exponent is
/// one and nothing else, so a negative exponent is unambiguous: `rim_join` reads `−y` as the leg
/// along the end and the shape as the bevel. The vector stays four wide, which every rim function
/// in the prelude takes; a fifth component would have been forty signatures for one number. The
/// leg is a slot pushed right after the reach's, in both passes (`rim_slots`), so dragging it
/// writes a buffer rather than compiling a shader.
fn rim_vector(r: usize, rims: &fieldform_scene::Rims, square: &str) -> String {
    let ends = match rims.at {
        fieldform_scene::RimEnds::Both => 0.0f32,
        fieldform_scene::RimEnds::Start => 1.0,
        fieldform_scene::RimEnds::End => 2.0,
    };
    let shape = match rims.fillet().second_leg() {
        Some(_) => format!("-params[{}]", r + 1),
        None => format!("{:?}", rims.exponent() as f32),
    };
    format!("vec4<f32>(params[{r}], {shape}, {ends:?}, {square})")
}

/// How many slots a rim's vector reads after its reach: one more for a chamfer's second leg.
fn rim_slots(rims: &fieldform_scene::Rims) -> usize {
    1 + usize::from(rims.fillet().second_leg().is_some())
}

/// Whether an extrude's two readings are square to each other, so its rims read the exact
/// corner: swept along its profile's own normal, with no draft on either side -- the same
/// facts the compile reads to decide it.
fn extrude_rims_are_square(
    document: &fieldform_scene::Document,
    sweep: &fieldform_scene::Sweep,
    draft: &ParamRef,
    sides: &fieldform_scene::Sides,
) -> &'static str {
    if matches!(sweep, fieldform_scene::Sweep::Normal) && extrude_is_upright(document, draft, sides) {
        "1.0"
    } else {
        "0.0"
    }
}

/// `name(args)` as `name`'s `_rim_` twin with `rim` appended: `x_distance` becomes
/// `x_rim_distance` and `x_value_normal` `x_rim_value_normal`, the names the prelude gives them.
fn rim_call(call: &str, rim: &str) -> String {
    let open = call.find('(').expect("a call");
    let name = &call[..open];
    let twin = ["_value_normal", "_distance"]
        .iter()
        .find_map(|tail| name.strip_suffix(tail).map(|head| format!("{head}_rim{tail}")))
        .expect("a call whose name ends as a body's does");
    let args = call[open + 1..].strip_suffix(')').expect("the call closes");
    format!("{twin}({args}, {rim})")
}

/// A leaf that reaches the GPU as a baked texture rather than as an expression.
///
/// Two kinds today and one mechanism: both are answered by closest-point search over triangles,
/// which is not something a shader can do, so both are sampled from a narrow band baked on the
/// CPU. Which one a texture came from stops mattering the moment it is baked, and the shader is
/// written so it cannot ask.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BakedLeaf {
    Cage(CageId),
    Mesh(fieldform_scene::MeshId),
}

/// The number the shader selects a cell with.
///
/// Written here rather than derived from the enums position, so adding a cell to the scenes
/// vocabulary is a compile error here until the shader knows what to do with it.
fn lattice_cell_index(cell: fieldform_scene::LatticeCell) -> f64 {
    match cell {
        fieldform_scene::LatticeCell::Gyroid => 0.0,
        fieldform_scene::LatticeCell::SchwarzP => 1.0,
        fieldform_scene::LatticeCell::SchwarzD => 2.0,
        fieldform_scene::LatticeCell::SchoenIwp => 3.0,
        fieldform_scene::LatticeCell::Neovius => 4.0,
        fieldform_scene::LatticeCell::FischerKochS => 5.0,
        // Six upward is a beam, which `lattice_distance` branches on: everything below it is a
        // banded surface, and everything from here is a distance to a strut set.
        fieldform_scene::LatticeCell::Cubic => 6.0,
        fieldform_scene::LatticeCell::Bcc => 7.0,
        fieldform_scene::LatticeCell::Fcc => 8.0,
        fieldform_scene::LatticeCell::Octet => 9.0,
        fieldform_scene::LatticeCell::Diamond => 10.0,
        // Eleven upward is a prism, which `lattice_distance` answers before anything else: a
        // distance to walls in the plane, in millimetres, with no cell coordinates.
        fieldform_scene::LatticeCell::Honeycomb => 11.0,
        // Twelve upward is a jittered seed grid, and the index is also the order statistic the
        // measure reads: the second nearest for a foam's walls, the third for its edges.
        fieldform_scene::LatticeCell::VoronoiFoam { .. } => 12.0,
        fieldform_scene::LatticeCell::VoronoiBeam { .. } => 13.0,
    }
}

/// Every lattice cell the scene can name.
///
/// A list rather than a derive, and guarded so it cannot fall behind: the test that walks the
/// shader's branches has to walk *all* of them, and a cell missing from here would be a cell
/// nothing checks.
#[cfg(test)]
const EVERY_LATTICE_CELL: [fieldform_scene::LatticeCell; 14] = {
    use fieldform_scene::LatticeCell as C;
    // The match is the guard: adding a variant to `LatticeCell` stops this compiling until it is
    // added to the array below too.
    const fn listed(c: C) -> C {
        match c {
            C::Gyroid | C::SchwarzP | C::SchwarzD => c,
            C::SchoenIwp | C::Neovius | C::FischerKochS => c,
            C::Cubic | C::Bcc | C::Fcc | C::Octet | C::Diamond => c,
            C::Honeycomb | C::VoronoiFoam { .. } | C::VoronoiBeam { .. } => c,
        }
    }
    [
        listed(C::Gyroid),
        listed(C::SchwarzP),
        listed(C::SchwarzD),
        listed(C::SchoenIwp),
        listed(C::Neovius),
        listed(C::FischerKochS),
        listed(C::Cubic),
        listed(C::Bcc),
        listed(C::Fcc),
        listed(C::Octet),
        listed(C::Diamond),
        listed(C::Honeycomb),
        listed(C::VoronoiFoam { jitter: fieldform_scene::ParamRef::Literal(0.75), seed: 0 }),
        listed(C::VoronoiBeam { jitter: fieldform_scene::ParamRef::Literal(0.75), seed: 0 }),
    ]
};

/// The number the shader selects a phase with.
///
/// Written out for the same reason as the cell above: a phase the scene grows and the shader has
/// not been taught is a compile error here rather than a silent fall-through to sheet.
fn lattice_phase_index(phase: &LatticePhase) -> f64 {
    match phase {
        LatticePhase::Sheet { .. } => 0.0,
        LatticePhase::Solid { .. } => 1.0,
        LatticePhase::Strut { .. } => 2.0,
    }
}

/// Generates `field_normal`, which answers with the field's own normal rather than the
/// gradient of its sampled distance.
///
/// This mirrors `GraphField::normal_at` on the CPU: a boolean's gradient at a point is the
/// gradient of whichever operand *defined* the value there, so the same comparison that picked
/// the value picks the normal, and a difference negates the operand it subtracts.
///
/// A blend goes further than the CPU does. `GraphField` returns `None` there and lets the
/// caller fall back to differencing, because near a seam both fields contribute and there is no
/// single active operand. But for the polynomial smooth minimum the mixed gradient is *exact*,
/// not an approximation: differentiating `mix(b, a, h) - k h (1-h)` leaves a term multiplied by
/// `(a - b) - k(1 - 2h)`, and `h` is defined such that `a - b = k(1 - 2h)`, so that term is
/// identically zero. What remains is `mix(grad_b, grad_a, h)`.
fn normal_function(
    document: &fieldform_scene::Document,
    graph: &Graph,
    order: &[NodeId],
    cages: &[BakedLeaf],
    curves: &[CurveUse],
    // Each body's guard slots, as the value pass allocated them, so the two passes ask the
    // same boxes and cannot pick different owners.
    boxes: &[Option<usize>],
    // Where the value pass found the per-tile extents, so this pass reads the same six slots.
    cull: Option<usize>,
    prune_patterns: bool,
    // The same floor the value pass used, so both passes take the same runs -- which is what
    // keeps their parameter counting in step. See `looped_run`.
    run_floor: usize,
) -> Result<String, CodegenError> {
    let mut slot: HashMap<(NodeId, usize), usize> = HashMap::new();
    let mut params = 0usize;

    // The same point chain the value pass builds, counted rather than collected — this walks
    // the identical order, so the parameter indices agree by construction, exactly as
    // `point_slots` and `push_point` already do for a primitive's centre.
    let plan = point_assignment(document, graph, order)?;
    let mut body = String::from("    let p0 = p;\n");
    // **Each node's two lines, kept apart rather than appended to one function.** The normal
    // pass is the heavier of the two -- a value *and* a gradient per node -- so leaving it whole
    // while splitting the distance pass would have kept the stall with nothing to show for it.
    let mut node_lines: Vec<String> = Vec::new();
    // A pattern's subtree as a function; and the statements that precede a node's own line,
    // for the pattern's loop and the boolean's fold -- see the same pair in the value pass.
    let mut subtrees: Vec<String> = Vec::new();
    let mut blocks_before: Vec<String> = Vec::new();
    let mut placement_slots: HashMap<usize, (String, usize)> = HashMap::new();
    // A mirror's plane normal, per declared point, since the normal pass has to reflect a
    // direction back out with the same plane the point was reflected in.
    let mut mirror_normals: HashMap<usize, String> = HashMap::new();
    // And, for a mirror that folds rather than reflects, whether this point went across -- so
    // the arm below turns its normal back out exactly when it did.
    let mut mirror_folded: HashMap<usize, String> = HashMap::new();
    // The same for an attractor, and for the same reason as in the value pass.
    let mut attract_slots: HashMap<usize, (usize, usize)> = HashMap::new();
    // The same for a radial pattern: which frame and count a declared point was folded with, so
    // the normal found inside an instance can be turned back out by the angle it went in by.
    // What each pattern's loop needs, by the context standing for its current step.
    let mut pattern_loops: HashMap<usize, PatternLoop> = HashMap::new();
    // One pattern a program has its copies masked: see `COPY_MASK`.
    let mut mask_taken = false;
    // One frame per pattern occurrence, matching the value pass slot for slot.
    let mut shared_frame: HashMap<(NodeId, usize), (String, String)> = HashMap::new();
    for (index, (parent, node_id, offset)) in plan.contexts.iter().enumerate() {
        match graph.node(*node_id) {
            Some(Node::Transform { translation, rotation, .. }) => {
                let t = point_slots(&mut params, translation);
                let euler = point_slots(&mut params, rotation);
                let s = params;
                params += 1;
                // Three more, matching the value pass slot for slot: it pushes a pivot here,
                // and a count that disagreed would shift every parameter after it, silently.
                let pivot = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                body.push_str(&format!(
                    "    let p{} = to_local_about(p{parent}, {t}, {euler}, params[{s}], \
                     {pivot});\n",
                    index + 1,
                ));
                placement_slots.insert(index + 1, (euler, s));
            }
            Some(Node::Twist { .. }) => {
                let at = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let ax = params;
                params += 1;
                let r = params;
                params += 1;
                body.push_str(&format!(
                    "    let p{} = unwind_point(p{parent}, radians(params[{r}]), \
                     params[{ax}], {at});\n",
                    index + 1,
                ));
            }
            // The same two slots the value pass pushes, counted rather than collected. The
            // curve is already interned by then, so this looks it up instead of adding it.
            Some(Node::Attract { toward, .. }) => {
                let c = curves
                    .iter()
                    .position(|x| *x == attractor_use(*toward))
                    .expect("interned by the value pass");
                let st = params;
                params += 1;
                let fo = params;
                params += 1;
                attract_slots.insert(index + 1, (st, fo));
                body.push_str(&format!(
                    "    let p{} = attract_point({c}u, p{parent}, params[{st}], \
                     params[{fo}]);\n",
                    index + 1,
                ));
            }
            Some(Node::Mirror { frame, .. }) => {
                let (at, normal) = frame_slots(&mut params, document, graph, frame);
                match fieldform_field::mirror_one_side(document, graph, *node_id) {
                    Some(side) => {
                        // Whether this point was folded, declared here beside the fold itself so
                        // the arm below turns a normal back out exactly when the point went
                        // across. Written as its own line because the body functions are
                        // assembled by name: reaching the normal brings the flag with it.
                        body.push_str(&format!(
                            "    let mf{} = dot(p{parent} - {at}, {normal}) * {side:?} < 0.0;\n\
                             \x20   let p{} = mirror_fold(p{parent}, {at}, {normal}, {side:?});\n",
                            index + 1,
                            index + 1,
                        ));
                        mirror_folded.insert(index + 1, format!("mf{}", index + 1));
                    }
                    None => body.push_str(&format!(
                        "    let p{} = mirror_point(p{parent}, {at}, {normal});\n",
                        index + 1,
                    )),
                }
                mirror_normals.insert(index + 1, normal);
            }
            // One step of a pattern's window -- see `window_of`. The same fold the value
            // pass emits, in this pass's own slot vocabulary, and it has to stay the same: the
            // two passes count parameters in step with each other.
            // Nothing emitted, and the same slots pushed in the same order as the value pass.
            // **The two passes count parameters in step with each other**, so a slot added to
            // one and not the other shifts every number after it -- silently, since the shader
            // still compiles and simply reads the wrong ones.
            Some(Node::Pattern { input, kind, fillet, .. }) => {
                let numbers = pattern_numbers(document, graph, *node_id);
                let _ = *offset;
                let rounding = (!fillet.is_plainly_hard()).then(|| {
                    let at = params;
                    params += 1;
                    at
                });
                // The value pass pushes the prune's five slots here when it prunes; count them
                // so every slot after them still agrees. The normal pass itself walks the whole
                // loop: it runs once a hit pixel, not once a step.
                let pruned = prune_patterns && pattern_prune_floor(document, graph, input).is_ok();
                if pruned {
                    params += PRUNE_SLOTS;
                }
                let prune: Result<usize, String> = Err(String::new());
                // Counted as the value pass pushed it; this pass walks every copy regardless.
                if masks_copies(rounding.is_some(), *parent) && !mask_taken {
                    mask_taken = true;
                    params += 1;
                }
                let mask: Option<usize> = None;
                let exponent = fillet.exponent();
                let whole = 2 * window_of(document, graph, *node_id, &numbers.layout)
                    .end()
                    .max(&0)
                    .to_owned() as usize
                    + 1
                    >= numbers.layout.total as usize;
                let reach = params;
                params += 1;
                match kind {
                    fieldform_scene::PatternKind::Linear { .. } if numbers.grid.is_some() => {
                        let g = numbers.grid.as_ref().expect("guarded");
                        let unused = [ParamRef::Literal(0.0); 3];
                        let anchor = point_slots(&mut params, &unused);
                        let disp0 = point_slots(&mut params, &unused);
                        let e1 = point_slots(&mut params, &unused);
                        let e2 = point_slots(&mut params, &unused);
                        let d1 = point_slots(&mut params, &unused);
                        let d2 = point_slots(&mut params, &unused);
                        let cn = params;
                        params += 1;
                        let n1 = params;
                        params += 1;
                        let n2 = params;
                        params += 1;
                        let r1 = params;
                        params += 1;
                        let r2 = params;
                        params += 1;
                        let first = params;
                        params += 1;
                        let extras = count_pattern_extras(&mut params, &numbers);
                        pattern_loops.insert(
                            index + 1,
                            PatternLoop {
                                parent: *parent,
                                reach,
                                whole,
                                rounding,
                                exponent,
                                prune: prune.clone(),
                                mask,
                                first,
                                left_out: extras.left_out,
                                skip: extras.skip,
                                point: PatternPoint::Grid {
                                    anchor,
                                    disp0,
                                    e1,
                                    e2,
                                    d1,
                                    d2,
                                    count: cn,
                                    columns: n1,
                                    rows: n2,
                                    reach: [r1, r2],
                                    whole: [
                                        2 * g.reach[0] + 1 >= g.columns as usize,
                                        2 * g.reach[1] + 1 >= g.rows as usize,
                                    ],
                                },
                            },
                        );
                    }
                    fieldform_scene::PatternKind::Linear { .. } => {
                        let u = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                        let pitch = params;
                        params += 1;
                        let cn = params;
                        params += 1;
                        let origin = params;
                        params += 1;
                        let first = params;
                        params += 1;
                        let extras = count_pattern_extras(&mut params, &numbers);
                        pattern_loops.insert(
                            index + 1,
                            PatternLoop {
                                parent: *parent,
                                reach,
                                whole,
                                rounding,
                                exponent,
                                prune: prune.clone(),
                                mask,
                                first,
                                left_out: extras.left_out,
                                skip: extras.skip,
                                point: PatternPoint::Linear {
                                    unit: u,
                                    pitch,
                                    count: cn,
                                    origin,
                                    bias: extras.bias,
                                },
                            },
                        );
                    }
                    fieldform_scene::PatternKind::Radial { .. } => {
                        // Shared per occurrence, exactly as the value pass shares it.
                        let (at, euler) = shared_frame
                            .entry((*node_id, *parent))
                            .or_insert_with(|| {
                                let at = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                                let euler =
                                    point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                                (at, euler)
                            })
                            .clone();
                        let cn = params;
                        params += 1;
                        let origin = params;
                        params += 1;
                        let first = params;
                        params += 1;
                        let extras = count_pattern_extras(&mut params, &numbers);
                        pattern_loops.insert(
                            index + 1,
                            PatternLoop {
                                parent: *parent,
                                reach,
                                whole,
                                rounding,
                                exponent,
                                prune: prune.clone(),
                                mask,
                                first,
                                left_out: extras.left_out,
                                skip: extras.skip,
                                point: PatternPoint::Radial {
                                    at,
                                    euler,
                                    count: cn,
                                    step: extras.step,
                                    origin,
                                    bias: extras.bias,
                                    arc: extras.arc,
                                    is_arc: numbers.arc,
                                },
                            },
                        );
                    }
                }
            }
            _ => unreachable!("point_assignment records the nodes that move a point"),
        }
    }

    for (id, ctx) in &plan.emit {
        let (id, ctx) = (*id, *ctx);
        let node = graph.node(id).ok_or(CodegenError::DanglingNode(id))?;
        let n = slot.len();
        let p = format!("p{ctx}");
        let inner = plan.inputs_of(id, ctx);

        // Both the value and the normal, because choosing between operands needs the values.
        let (value, normal) = match node {
            Node::Cage(cage) => {
                let leaf = BakedLeaf::Cage(*cage);
                let i = cages.iter().position(|c| *c == leaf).expect("interned above");
                (
                    format!("sample_baked({i}u, {p})"),
                    format!("sample_baked_normal({i}u, {p})"),
                )
            }
            Node::Mesh(mesh) => {
                let leaf = BakedLeaf::Mesh(*mesh);
                let i = cages.iter().position(|c| *c == leaf).expect("interned above");
                (
                    format!("sample_baked({i}u, {p})"),
                    format!("sample_baked_normal({i}u, {p})"),
                )
            }
            Node::Sphere { .. } => {
                // Three either way -- see `place_slots`, which the value pass uses to push what
                // this one is counting.
                let c = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let r = params;
                params += 1;
                (
                    format!("sphere_distance({p}, {c}, params[{r}])"),
                    format!("sphere_normal({p}, {c})"),
                )
            }
            Node::Cuboid { size, edge, .. } => {
                // Three either way -- see `place_slots`, which the value pass uses to push what
                // this one is counting.
                let c = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let s = point_slots(&mut params, size);
                match edge_slot(edge, &mut params) {
                    None => (
                        format!("cuboid_distance({p}, {c}, {s})"),
                        format!("cuboid_normal({p}, {c}, {s})"),
                    ),
                    Some(e) => (
                        format!("cuboid_edge_distance({p}, {c}, {s}, {e})"),
                        format!("cuboid_edge_normal({p}, {c}, {s}, {e})"),
                    ),
                }
            }
            Node::SkewedBox { .. } => {
                // Twelve either way: the centre's three, then three edges of three.
                let unused = [ParamRef::Literal(0.0); 3];
                let c = point_slots(&mut params, &unused);
                let e0 = point_slots(&mut params, &unused);
                let e1 = point_slots(&mut params, &unused);
                let e2 = point_slots(&mut params, &unused);
                let args = format!("{p}, {c}, {e0}, {e1}, {e2}");
                (format!("skewed_box_distance({args})"), format!("skewed_box_normal({args})"))
            }
            Node::Cylinder { edge, .. } => {
                let (a, b) = ends_slots(&mut params);
                let r = params;
                params += 1;
                let args = format!("{p}, {a}, {b}, params[{r}]");
                match edge_slot(edge, &mut params) {
                    None => (format!("cylinder_distance({args})"), format!("cylinder_normal({args})")),
                    Some(e) => (
                        format!("cylinder_edge_distance({args}, {e})"),
                        format!("cylinder_edge_normal({args}, {e})"),
                    ),
                }
            }
            Node::Cone { edge, .. } => {
                let (a, b) = ends_slots(&mut params);
                let r = params;
                params += 2;
                let args = format!("{p}, {a}, {b}, params[{r}], params[{}]", r + 1);
                match edge_slot(edge, &mut params) {
                    None => (format!("cone_distance({args})"), format!("cone_normal({args})")),
                    Some(e) => (format!("cone_edge_distance({args}, {e})"), format!("cone_edge_normal({args}, {e})")),
                }
            }
            Node::Torus { .. } => {
                // Three either way -- see `place_slots`, which the value pass uses to push what
                // this one is counting.
                let c = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let r = params;
                params += 2;
                let args = format!("{p}, {c}, params[{r}], params[{}]", r + 1);
                (format!("torus_distance({args})"), format!("torus_normal({args})"))
            }
            Node::Lattice { map, phase, .. } => {
                let c = params;
                params += 12;
                // The same expression and the same divisor the value pass wrote. The normal is
                // the fixed lattice's of that size, as a driven pipe's is its spine's: the
                // ramp's own gradient turns it a little, and gently for any grade worth tracing.
                let driver = phase.param().driver();
                let sizing = match driver {
                    Some(driver) => format!("({})", value_of(&slot, (driver, ctx))?),
                    None => format!("params[{}]", c + 6),
                };
                let divisor = finite(
                    fieldform_field::driven_lattice_divisor(document, graph, id),
                    "a graded lattice's bound",
                )?;
                let (value, normal) = if matches!(map, fieldform_scene::CellMap::Conformal { .. }) {
                    let args = format!(
                        "{p}, params[{c}], u32(params[{}]), params[{}], params[{}], params[{}],                          {sizing}, params[{}], params[{}]",
                        c + 2,
                        c + 3,
                        c + 4,
                        c + 5,
                        c + 7,
                        c + 8,
                    );
                    (
                        format!("conformal_lattice_distance({args})"),
                        format!("conformal_lattice_normal({args})"),
                    )
                } else {
                    let args = format!(
                        "{p}, params[{c}], params[{}], params[{}], params[{}], params[{}],                          params[{}], {sizing}, params[{}], params[{}], params[{}], params[{}]",
                        c + 1,
                        c + 2,
                        c + 3,
                        c + 4,
                        c + 5,
                        c + 8,
                        c + 9,
                        c + 10,
                        c + 11,
                    );
                    (format!("lattice_distance({args})"), format!("lattice_normal({args})"))
                };
                match driver {
                    Some(_) => (format!("({value}) / {divisor:?}"), normal),
                    None => (value, normal),
                }
            }
            Node::Plane { .. } => {
                // Three either way -- see `place_slots`, which the value pass uses to push what
                // this one is counting.
                let o = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let f = params;
                params += 1;
                let args = format!("{p}, {o}, params[{f}]");
                (format!("plane_distance({args})"), format!("plane_normal({args})"))
            }
            Node::Point { at } => {
                let c = point_slots(&mut params, at);
                (format!("point_distance({p}, {c})"), format!("point_normal({p}, {c})"))
            }
            Node::Capsule { .. } => {
                let (a, b) = ends_slots(&mut params);
                let r = params;
                params += 2;
                let args = format!("{p}, {a}, {b}, params[{r}], params[{}]", r + 1);
                (format!("capsule_distance({args})"), format!("capsule_normal({args})"))
            }
            Node::Extrude { outline, sweep, draft, sides, rims, .. } => {
                let found = |use_of: CurveUse| {
                    curves.iter().position(|x| *x == use_of).expect("interned above")
                };
                let square = extrude_rims_are_square(document, sweep, draft, sides);
                // The rim's slot comes last of all the extrude's, as the value pass pushed it:
                // after the depth, and after a wall's four. `None` for a hard rim, which
                // pushed nothing.
                let rimmed = !rims.is_plainly_hard();
                // Nine slots, matching `extrude_placement`, counted in the same order.
                let at = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let euler = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let dir = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                // Six more slots, counted in the order the value pass pushes them
                // (`depth_slots`): the two passes agree by counting rather than by reasoning,
                // so a slot added to one and not the other shifts every number after it,
                // silently.
                let depth = depth_read(params);
                params += DEPTH_SLOTS;
                let local = format!("to_local({p}, {at}, {euler}, 1.0)");
                // The miters, where the value pass bound them -- found by asking the same
                // question of the same node rather than by counting.
                let mitres = curves.iter().position(|x| *x == CurveUse::Miters(id));
                let rim_slot = |params: &mut usize| -> Option<String> {
                    rimmed.then(|| {
                        let r = *params;
                        *params += rim_slots(rims);
                        rim_vector(r, rims, square)
                    })
                };
                match (outline, mitres) {
                    (fieldform_scene::Outline::Closed { .. }, None) => {
                        let c = found(CurveUse::Profile(id));
                        match rim_slot(&mut params) {
                            None => (
                                format!("extrude_distance({c}u, {local}, {dir}, {depth})"),
                                format!("extrude_normal({c}u, {local}, {dir}, {depth}, {euler})"),
                            ),
                            Some(rim) => (
                                format!("extrude_rim_distance({c}u, {local}, {dir}, {depth}, {rim})"),
                                format!(
                                    "extrude_rim_normal({c}u, {local}, {dir}, {depth}, {euler}, \
                                     {rim})"
                                ),
                            ),
                        }
                    }
                    (fieldform_scene::Outline::Closed { .. }, Some(m)) => {
                        let c = found(CurveUse::Profile(id));
                        match rim_slot(&mut params) {
                            None => (
                                format!("extrude_sharp_distance({c}u, {m}u, {local}, {dir}, {depth})"),
                                format!(
                                    "extrude_sharp_normal({c}u, {m}u, {local}, {dir}, {depth}, \
                                     {euler})"
                                ),
                            ),
                            Some(rim) => (
                                format!(
                                    "extrude_sharp_rim_distance({c}u, {m}u, {local}, {dir}, \
                                     {depth}, {rim})"
                                ),
                                format!(
                                    "extrude_sharp_rim_normal({c}u, {m}u, {local}, {dir}, \
                                     {depth}, {euler}, {rim})"
                                ),
                            ),
                        }
                    }
                    // The wall's four, counted where the value pass pushed them.
                    (fieldform_scene::Outline::Wall { .. }, mitres) => {
                        let [c, f, s] = WALL_PARTS.map(|part| found(CurveUse::Wall(id, part)));
                        let t = params;
                        params += 4;
                        if let Some(rim) = rim_slot(&mut params) {
                            let wall = format!(
                                "vec4<f32>(params[{t}], params[{}], params[{}], params[{}])",
                                t + 1,
                                t + 2,
                                t + 3,
                            );
                            let numbers = format!(
                                "params[{t}], params[{}], params[{}], params[{}]",
                                t + 1,
                                t + 2,
                                t + 3,
                            );
                            match mitres {
                                Some(m) => (
                                    format!(
                                        "wall_extrude_sharp_rim_distance({c}u, {m}u, {local}, \
                                         {dir}, {depth}, {wall}, {rim})"
                                    ),
                                    format!(
                                        "wall_extrude_sharp_rim_normal({c}u, {m}u, {local}, \
                                         {dir}, {depth}, {wall}, {euler}, {rim})"
                                    ),
                                ),
                                None => (
                                    format!(
                                        "wall_extrude_rim_distance({c}u, {f}u, {s}u, {local}, \
                                         {dir}, {depth}, {numbers}, {rim})"
                                    ),
                                    format!(
                                        "wall_extrude_rim_normal({c}u, {f}u, {s}u, {local}, \
                                         {dir}, {depth}, {numbers}, {euler}, {rim})"
                                    ),
                                ),
                            }
                        } else if let Some(m) = mitres {
                            let wall = format!(
                                "vec4<f32>(params[{t}], params[{}], params[{}], params[{}])",
                                t + 1,
                                t + 2,
                                t + 3,
                            );
                            (
                                format!(
                                    "wall_extrude_sharp_distance({c}u, {m}u, {local}, {dir},                                      {depth}, {wall})"
                                ),
                                format!(
                                    "wall_extrude_sharp_normal({c}u, {m}u, {local}, {dir},                                      {depth}, {wall}, {euler})"
                                ),
                            )
                        } else {
                            let args = format!(
                                "{c}u, {f}u, {s}u, {local}, {dir}, {depth}, params[{t}], \
                                 params[{}], params[{}], params[{}]",
                                t + 1,
                                t + 2,
                                t + 3,
                            );
                            (
                                format!("wall_extrude_distance({args})"),
                                format!("wall_extrude_normal({args}, {euler})"),
                            )
                        }
                    }
                }
            }
            Node::Revolve { outline, angle, sides, rims, .. } => {
                let found = |use_of: CurveUse| {
                    curves.iter().position(|x| *x == use_of).expect("interned above")
                };
                let at = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let euler = point_slots(&mut params, &[ParamRef::Literal(0.0); 3]);
                let local = format!("to_local({p}, {at}, {euler}, 1.0)");
                // The wedge's two slots after everything else the kind pushes, counted by the
                // same decision the value pass made: full turn, none.
                let partial = fieldform_field::revolve_wedge(document, angle, sides)
                    .ok()
                    .filter(|(_, total)| *total < 360.0 - 1e-9)
                    .is_some();
                match outline {
                    fieldform_scene::Outline::Closed { .. } => {
                        let c = found(CurveUse::Profile(id));
                        if partial {
                            let w = params;
                            params += 2;
                            let args = format!("{c}u, {local}, params[{w}], params[{}]", w + 1);
                            // The rim's slot after the wedge's two, as the value pass pushed it.
                            match edge_like_slot(rims, &mut params) {
                                None => (
                                    format!("revolve_wedge_distance({args})"),
                                    format!("revolve_wedge_normal({args}, {euler})"),
                                ),
                                Some(rim) => (
                                    format!("revolve_wedge_rim_distance({args}, {rim})"),
                                    format!("revolve_wedge_rim_normal({args}, {rim}, {euler})"),
                                ),
                            }
                        } else {
                            (
                                format!("revolve_distance({c}u, {local})"),
                                format!("revolve_normal({c}u, {local}, {euler})"),
                            )
                        }
                    }
                    // The wall's four and the one that says its band crosses the axis, counted
                    // where the value pass pushed them, then the wedge's two.
                    fieldform_scene::Outline::Wall { .. } => {
                        let [c, f, s] = WALL_PARTS.map(|part| found(CurveUse::Wall(id, part)));
                        let t = params;
                        params += 5;
                        let args = format!(
                            "{c}u, {f}u, {s}u, {local}, {}, params[{}]",
                            wall_numbers_as_a_vector(t),
                            t + 4,
                        );
                        if partial {
                            let w = params;
                            params += 2;
                            let args = format!("{args}, params[{w}], params[{}]", w + 1);
                            match edge_like_slot(rims, &mut params) {
                                None => (
                                    format!("wall_revolve_wedge_distance({args})"),
                                    format!("wall_revolve_wedge_normal({args}, {euler})"),
                                ),
                                Some(rim) => (
                                    format!("wall_revolve_wedge_rim_distance({args}, {rim})"),
                                    format!("wall_revolve_wedge_rim_normal({args}, {rim}, {euler})"),
                                ),
                            }
                        } else {
                            (
                                format!("wall_revolve_distance({args})"),
                                format!("wall_revolve_normal({args}, {euler})"),
                            )
                        }
                    }
                }
            }
            Node::Pipe { curve, radius } => {
                let c = curves
                    .iter()
                    .position(|x| *x == CurveUse::Spine(*curve))
                    .expect("interned above");
                // The same slot, counted the same way -- and the same *absence* of one when the
                // radius is driven, since the value pass pushes none then either. The two
                // passes agree by counting rather than by reasoning.
                let value = match radius.driver() {
                    None => {
                        let r = params;
                        params += 1;
                        format!("pipe_distance({c}u, {p}, params[{r}])")
                    }
                    Some(driver) => {
                        let amount = value_of(&slot, (driver, ctx))?;
                        // Counted as the value pass pushed it.
                        let d = params;
                        params += 1;
                        format!("((pipe_distance({c}u, {p}, 0.0)) - ({amount})) / params[{d}]")
                    }
                };
                // The spine's normal either way, which is what a driven offset does too: the
                // radius moves the surface along that normal without turning it, except where
                // it is changing fastest, and the offset arm has always taken the body's.
                (value, format!("pipe_normal({c}u, {p})"))
            }
            // The same slot, counted the same way: the two passes agree by counting rather
            // than by reasoning, so a parameter pushed in one and not the other shifts every
            // number after it and nothing says so.
            Node::Morph { a, b, t } => {
                let (va, vb) = (value_of(&slot, (*a, inner))?, value_of(&slot, (*b, inner))?);
                let (ga, gb) = (normal_of(&slot, (*a, inner))?, normal_of(&slot, (*b, inner))?);
                // Counted the same way as the value pass: a slot pushed in one and not the
                // other shifts every parameter after it, silently.
                let amount = match t.driver() {
                    None => {
                        let i = params;
                        params += 1;
                        format!("params[{i}]")
                    }
                    Some(driver) => value_of(&slot, (driver, ctx))?,
                };
                // The gradient of `(1 − t)·A + t·B` is `(1 − t)·∇A + t·∇B`, which is exact
                // rather than a plausible mix — and the fallback matters: where the two
                // gradients oppose, the combination is near zero and normalizing it returns
                // rounding noise. Below the threshold this takes whichever operand is nearer
                // its own surface, which is the one deciding the value there.
                let t = format!("clamp({amount}, 0.0, 1.0)");
                blocks_before.push(format!(
                    "    let mt{n} = {t};\n\
                     \x20   let mg{n} = mix({ga}, {gb}, mt{n});\n\
                     \x20   let mn{n} = select(select({gb}, {ga}, abs({va}) <= abs({vb})), \
                     normalize(mg{n}), length(mg{n}) > 1e-6);\n"
                ));
                (format!("mix({va}, {vb}, mt{n})"), format!("mn{n}"))
            }
            Node::Boolean { op, inputs } => {
                // **The same run the value pass took, decided by the same function.** The two
                // passes agree about parameter slots by counting the same things in the same
                // order, so a run taken here and not there — or there and not here — would shift
                // every slot after it with nothing to say so. The `_nearest` loop walks the run
                // once and hands back the winning member's normal in `xyz` and its value in `w`,
                // which is one call where the fold would otherwise have had two names an operand.
                let run = looped_run(*op, graph, inputs, run_floor);
                let mut values: Vec<String> = Vec::new();
                let mut normals: Vec<String> = Vec::new();
                match run.as_ref().zip(run_first_curve(&curves, run.as_ref())) {
                    Some((run, first)) => {
                        for input in inputs.iter().take(run.at) {
                            values.push(value_of(&slot, (*input, inner))?);
                            normals.push(normal_of(&slot, (*input, inner))?);
                        }
                        let base = params;
                        // Counted as the value pass pushed them: `slots_each` a member, then the
                        // count's own slot. Counted rather than resolved — nothing here reads a
                        // number, which is why the two passes cannot disagree about a value.
                        params += run.slots();
                        let count = base + run.len() * run.kind.slots_each();
                        let (_, walk) = run.kind.walkers();
                        blocks_before.push(format!(
                            "    let pr{n} = {walk}({first}u, {base}u, \
                             u32(params[{count}]), {p});\n"
                        ));
                        values.push(format!("pr{n}.w"));
                        normals.push(format!("pr{n}.xyz"));
                        for input in inputs.iter().skip(run.at + run.len()) {
                            values.push(value_of(&slot, (*input, inner))?);
                            normals.push(normal_of(&slot, (*input, inner))?);
                        }
                    }
                    None => {
                        for input in inputs.iter() {
                            values.push(value_of(&slot, (*input, inner))?);
                            normals.push(normal_of(&slot, (*input, inner))?);
                        }
                    }
                }
                let mut block = String::new();
                let folded = fold_boolean_with_normal(&mut block, *op, n, &values, &normals)
                    .ok_or(CodegenError::NothingToCombine(id))?;
                if !block.is_empty() {
                    blocks_before.push(block);
                }
                folded
            }
            Node::Blend { op, a, b, fillet, .. } => {
                let (va, vb) = (value_of(&slot, (*a, inner))?, value_of(&slot, (*b, inner))?);
                let (ga, gb) = (normal_of(&slot, (*a, inner))?, normal_of(&slot, (*b, inner))?);
                // One slot either way, counted as the value pass pushed it: the radius, or --
                // driven -- the divisor, with the radius the ramp's own expression. The two
                // passes agree by counting rather than by reasoning.
                let mut second_slot = None;
                let (k, divided_by) = match fillet.radius.driver() {
                    None => {
                        let r = params;
                        params += 1;
                        // The second leg's slot, counted as the value pass pushed it.
                        if fillet.second_leg().is_some() {
                            second_slot = Some(params);
                            params += 1;
                        }
                        (format!("params[{r}]"), None)
                    }
                    Some(driver) => {
                        let amount = value_of(&slot, (driver, ctx))?;
                        let d = params;
                        params += 1;
                        (format!("({amount})"), Some(d))
                    }
                };
                // A chamfer of two legs: the operands and their gradients scaled as the value
                // pass scales them, and the radius the lesser leg.
                let (va, vb, ga, gb, k) = match second_slot {
                    Some(sb) => {
                        let m = format!("min({k}, params[{sb}])");
                        (
                            format!("(({va}) * {m} / params[{sb}])"),
                            format!("(({vb}) * {m} / {k})"),
                            format!("(({ga}) * ({m} / params[{sb}]))"),
                            format!("(({gb}) * ({m} / {k}))"),
                            m,
                        )
                    }
                    None => (va, vb, ga, gb, k),
                };
                let n = fillet.exponent();
                // The negations that turn a smooth union into a difference or an intersection
                // apply to the gradients too, and in the same places — a sign dropped on one
                // side only produces a surface that shades as though it were lit from within.
                let (na, nb, sign) = match op {
                    BooleanOp::Union => (va.clone(), vb.clone(), 1.0),
                    BooleanOp::Difference => (format!("-({va})"), vb.clone(), -1.0),
                    BooleanOp::Intersection => (format!("-({va})"), format!("-({vb})"), -1.0),
                };
                let (na_grad, nb_grad) = match op {
                    BooleanOp::Union => (ga.clone(), gb.clone()),
                    BooleanOp::Difference => (format!("-({ga})"), gb.clone()),
                    BooleanOp::Intersection => (format!("-({ga})"), format!("-({gb})")),
                };
                // Only the value divides. The divisor is a positive constant, so the gradient's
                // direction is the undivided one's and the normal below needs nothing done.
                let (value, normal) = if tapers(graph) {
                    let rho = fieldform_field::taper_ratio(document, graph, id);
                    (
                        format!(
                            "(smooth_min_tapering({na}, {nb}, {k}, {n:?}) / \
                             blend_taper_divisor({k}, {n:?}, {rho:?}))"
                        ),
                        // Over the same divisor as the value, so that a blend handed to another
                        // blend arrives with the gradient it really has. See `blend_gradient`.
                        format!(
                            "(smooth_min_normal_tapering({na}, {nb}, {na_grad}, {nb_grad}, \
                             {k}, {n:?}) / blend_taper_divisor({k}, {n:?}, \
                             {rho:?}))"
                        ),
                    )
                } else {
                    (
                        format!("blend_bound({na}, {nb}, {k}, {n:?})"),
                        format!(
                            "blend_bound_normal({na}, {nb}, {na_grad}, {nb_grad}, {k}, \
                             {n:?})"
                        ),
                    )
                };
                // A driven radius divides the value by its bound, as the value pass does; the
                // normal keeps the local radius's direction, the ramp's own gradient turning it
                // a little and gently for any grade worth tracing -- the pipe's reasoning.
                let value = match divided_by {
                    Some(d) => format!("(({value}) / params[{d}])"),
                    None => value,
                };
                if sign < 0.0 {
                    (format!("-({value})"), format!("-({normal})"))
                } else {
                    (value, normal)
                }
            }
            // The value divides and the normal is left to the caller's differencing: the
            // warp's Jacobian is not orthonormal, so a rotated normal would be plausible and
            // not true. Returning the inner one unchanged is the same fallback a blend takes.
            Node::Twist { input, .. } => {
                // One slot, counted as the value pass pushed it: the two passes agree by
                // counting rather than by reasoning, so a slot added to one and not the other
                // shifts every number after it and nothing says so.
                let s = params;
                params += 1;
                (
                    format!("({} / params[{s}])", value_of(&slot, (*input, inner))?),
                    normal_of(&slot, (*input, inner))?,
                )
            }
            // The value divides; the normal is **turned**, not passed through.
            //
            // This arm read `normal_of(input)` at first, under a comment claiming the caller
            // differenced the composed field the way it does for a blend. Nothing differences:
            // `field_normal` is what the marcher reads, so the *unwarped* body's normal was
            // lighting an attracted surface. `attract_normal` applies `Dg^T`, which is what a
            // warped gradient actually is.
            Node::Attract { input, toward, .. } => {
                let c = curves
                    .iter()
                    .position(|x| *x == attractor_use(*toward))
                    .expect("interned by the value pass");
                let (st, fo) = attract_slots[&inner];
                (
                    format!(
                        "({} / attract_stretch(params[{st}]))",
                        value_of(&slot, (*input, inner))?
                    ),
                    format!(
                        "attract_normal({c}u, p{inner}, params[{st}], params[{fo}], {})",
                        normal_of(&slot, (*input, inner))?
                    ),
                )
            }
            // The value comes straight through and the normal is reflected back out: the
            // mirrored component negates, the others do not, and the result is still unit
            // length because a reflection is an isometry.
            // The nearest instance owns the answer, chosen by value the way a union chooses
            // between its operands. A linear pattern's normal needs nothing done to it -- a
            // translation leaves directions alone -- while a turn's has to come back out by the
            // angle its point went in by.
            // **Walked at run time, exactly as the value pass walks it**, so the two agree
            // about which instance owns the answer. The subtree is a function returning both
            // numbers at once -- gradient in `xyz`, value in `w` -- because the choice needs
            // the value and a second call to get it would double the work.
            Node::Pattern { input, .. } => {
                let info = pattern_loops.get(&inner).expect("a pattern declares its own context");
                let v = value_of(&slot, (*input, inner))?;
                let g = normal_of(&slot, (*input, inner))?;
                let root = format!("vec4<f32>({g}, {v})");
                let inner_body =
                    closure_of(&body, &node_lines, &root, &format!("    let p{inner} = p;\n"));
                subtrees.push(format!(
                    "fn pattern_subn_{n}(p: vec3<f32>) -> vec4<f32> {{\n{inner_body}    \
                     return {root};\n}}\n\n"
                ));
                // A linear pattern's normal needs nothing done to it -- a translation leaves
                // directions alone -- while a turn's has to come back out by the angle its
                // point went in by. **By the angle its point went in by**, which means asking
                // the loop: what `s{n}` counts is an absolute instance when the whole pattern
                // is walked and an offset from the point's own cell when a window is, and the
                // two are different rotations. See `instance_normal_at`.
                let brought_out = if info.whole {
                    info.point.instance_normal_at(&format!("w{n}.xyz"), &format!("f32(s{n})"))
                } else {
                    match &info.point {
                        PatternPoint::Radial { at, euler, count, step, origin, bias, arc, .. } => format!(
                            "pattern_normal_radial(w{n}.xyz, p{ctx}, {at}, {euler}, \
                             params[{count}], params[{step}], params[{origin}], params[{bias}], \
                             params[{arc}], f32(s{n}))"
                        ),
                        PatternPoint::Linear { .. } | PatternPoint::Grid { .. } => format!("w{n}.xyz"),
                    }
                };
                match info.rounding {
                    None => {
                        let walk = info.walk(&n.to_string(), None, |q| {
                            format!(
                                "        let w{n} = pattern_subn_{n}({q});\n\
                                 \x20       if (w{n}.w < pv{n}) {{ pv{n} = w{n}.w; \
                                 pg{n} = {brought_out}; }}\n"
                            )
                        });
                        blocks_before.push(format!(
                            "    var pv{n} = 1e30;\n    var pg{n} = vec3<f32>(0.0, 0.0, 1.0);\n\
                             {walk}"
                        ));
                        (format!("pv{n}"), format!("pg{n}"))
                    }
                    // The n-ary form of `smooth_min_normal`: each instance's gradient weighted
                    // by how far *inside* the fillet it is, to the power n − 1, and the sum
                    // normalized. Where nothing is inside, every weight is nought and the
                    // nearest instance's own normal is the answer -- which is what a hard fold
                    // would have given, and is the same fallback the binary form has.
                    Some(r) => {
                        let e = info.exponent;
                        let inv = 1.0 / e;
                        // The radius this point gets: the slot itself, or what is left of it
                        // near a critical surface -- read once, ahead of the loop.
                        let tapering = tapers(graph);
                        let k = if tapering { format!("ke{n}") } else { format!("params[{r}]") };
                        let walk = info.walk(&n.to_string(), None, |q| {
                            format!(
                                "        let w{n} = pattern_subn_{n}({q});\n\
                                 \x20       let g{n} = {brought_out};\n\
                                 \x20       if (w{n}.w < pv{n}) {{ pv{n} = w{n}.w; pg{n} = g{n}; }}\n\
                                 \x20       let x{n} = max({k} - w{n}.w, 0.0);\n\
                                 \x20       pi{n} = pi{n} + pow(x{n}, {e:?});\n\
                                 \x20       pw{n} = pw{n} + pow(x{n}, {:?}) * g{n};\n",
                                e - 1.0
                            )
                        });
                        // The value divides as the value pass divides it; the direction does
                        // not change, so `pgr` is left alone.
                        let (divisor, before, after) = if tapering {
                            let rho = fieldform_field::taper_ratio(document, graph, id);
                            (
                                info.taper_divisor(r, rho),
                                format!("    let ke{n} = tapered(params[{r}]);\n"),
                                // Tapering, and with something inside the fillet for the
                                // radius to matter to: `field_normal` differences this pixel.
                                format!(
                                    "    if (ke{n} < params[{r}] && pi{n} > 0.0) \
                                     {{ taper_hit = true; }}\n"
                                ),
                            )
                        } else {
                            (info.join_divisor(r), String::new(), String::new())
                        };
                        blocks_before.push(format!(
                            "{before}    var pv{n} = 1e30;\n    var pg{n} = vec3<f32>(0.0, 0.0, 1.0);\n\
                             \x20   var pi{n} = 0.0;\n    var pw{n} = vec3<f32>(0.0);\n{walk}{after}\
                             \x20   let pr{n} = (max({k}, pv{n}) - pow(pi{n}, {inv:?})) \
                             / {divisor};\n\
                             \x20   var pgr{n} = pg{n} / {divisor};\n\
                             \x20   if (pi{n} > 0.0) {{ pgr{n} = pw{n} \
                             / (pow(pi{n}, {:?}) * {divisor}); }}\n",
                            (e - 1.0) / e
                        ));
                        (format!("pr{n}"), format!("pgr{n}"))
                    }
                }
            }
            Node::Mirror { input, join, .. } => {
                let normal = mirror_normals
                    .get(&inner)
                    .cloned()
                    .unwrap_or_else(|| "vec3<f32>(0.0, 0.0, 1.0)".to_string());
                let reflected_v = value_of(&slot, (*input, inner))?;
                let reflected_g =
                    format!("mirror_normal({}, {normal})", normal_of(&slot, (*input, inner))?);
                match mirror_folded.get(&inner) {
                    // Folded: one reading, and its normal is turned back out only where the
                    // point itself went across. The same choice the value made, made the same
                    // way, which is what keeps a surface lit from the reflection it was drawn
                    // from.
                    Some(flag) => {
                        let here_g = normal_of(&slot, (*input, inner))?;
                        (reflected_v, format!("select({here_g}, {reflected_g}, {flag})"))
                    }
                    // Whichever half is nearer owns the normal, chosen the same way a union
                    // chooses between its operands — by which value is smaller.
                    None if *join => {
                        let here_v = value_of(&slot, (*input, ctx))?;
                        let here_g = normal_of(&slot, (*input, ctx))?;
                        (
                            format!("min({here_v}, {reflected_v})"),
                            format!("select({reflected_g}, {here_g}, {here_v} <= {reflected_v})"),
                        )
                    }
                    None => (reflected_v, reflected_g),
                }
            }
            // The value shifts and the normal does not move at all: a constant has no
            // gradient. This is the same reason the CPU path delegates its normal straight
            // through, and the reason an offset body stays blendable.
            //
            // A driven distance *does* turn the surface — the true normal is `∇F − ∇d` — and
            // the analytic one is not available here any more than it is on the CPU, which
            // returns `None` and lets a central difference answer. The shader has no such
            // fallback per node, so it passes the body's normal through: the shading is a
            // little wrong on a steeply varying offset, and the silhouette, which is what the
            // value decides, is exactly right. Worth stating rather than discovering.
            Node::Offset { input, distance } => {
                let body = value_of(&slot, (*input, inner))?;
                let value = match distance.driver() {
                    None => {
                        let d = params;
                        params += 1;
                        format!("{body} - params[{d}]")
                    }
                    Some(driver) => {
                        let amount = value_of(&slot, (driver, ctx))?;
                        // Counted as the value pass pushed it.
                        let d = params;
                        params += 1;
                        format!("(({body}) - ({amount})) / params[{d}]")
                    }
                };
                (value, normal_of(&slot, (*input, inner))?)
            }
            // The scale multiplies the value and cancels out of the gradient entirely, so
            // the normal is the inner one merely rotated — which is why a placed body shades
            // identically to an unplaced one instead of flattening as it grows.
            Node::Transform { input, .. } => {
                let (euler, s) = &placement_slots[&inner];
                (
                    format!("params[{s}] * {}", value_of(&slot, (*input, inner))?),
                    format!("euler_rotation({euler}) * {}", normal_of(&slot, (*input, inner))?),
                )
            }
            // No surface, so no normal. The vector is never read: a ramp is reached only as an
            // offset's amount, and that arm takes the *body's* normal rather than this one.
            Node::Ramp { input, .. } => {
                let (lo, span, out_lo, out_span, shape) = ramp_constants(document, node);
                (
                    format!(
                        "ramp({}, {lo:?}, {span:?}, {out_lo:?}, {out_span:?}, {shape:?})",
                        value_of(&slot, (*input, inner))?,
                    ),
                    "vec3<f32>(0.0, 0.0, 1.0)".to_string(),
                )
            }
            Node::Shell { .. } => {
                unreachable!("lowered into a difference before any point was assigned")
            }
            Node::Sweep { outline, spine, law, rims, .. } => {
                let first = match outline {
                    fieldform_scene::Outline::Closed { .. } => CurveUse::Profile(id),
                    fieldform_scene::Outline::Wall { .. } => CurveUse::Wall(id, WALL_PARTS[0]),
                };
                let c = curves.iter().position(|x| *x == first).expect("interned above");
                let s = curves
                    .iter()
                    .position(|x| *x == CurveUse::Spine(*spine))
                    .expect("interned above");
                let f = curves
                    .iter()
                    .position(|x| *x == CurveUse::Frame(id))
                    .expect("interned above");
                // One slot, counted as the value pass pushed it.
                let closed = params;
                params += 1;
                // And the bend's three, likewise: the stretch, the reach and the tube.
                let bend = params;
                params += 3;
                // And the three the move to the profile takes.
                let origin = params;
                params += 3;
                // And the seven the held section takes: two axes and the flag.
                let axes = params;
                params += 7;
                // And the law's six, where the value pass pushed them -- before the wall's
                // four, which is the order `law_slots` and `wall_slots` are called in. A
                // plain law pushed none, and this arm must count exactly what that one did:
                // a slot miscounted here reads somebody else's number as a twist.
                let under = law_reader(&mut params, law);
                // The same move to the profile the value pass makes, so the normal is read
                // where the surface actually is.
                let at = format!(
                    "{p} - vec3<f32>(params[{origin}], params[{}], params[{}])",
                    origin + 1,
                    origin + 2
                );
                let args = format!(
                    "{s}u, {f}u, {c}u, {at}, params[{closed}], \
                     vec3<f32>(params[{bend}], params[{}], params[{}]), \
                     vec3<f32>(params[{axes}], params[{}], params[{}]), \
                     vec3<f32>(params[{}], params[{}], params[{}]), params[{}]",
                    bend + 1,
                    bend + 2,
                    axes + 1,
                    axes + 2,
                    axes + 3,
                    axes + 4,
                    axes + 5,
                    axes + 6,
                );
                // **One call for both, because walking the spine is what costs.** Asking
                // `sweep_distance` and then a normal walked the spine's index twice at the same
                // point, and the driver unrolls that whole walk into every call site it finds.
                // `sweep_value_normal` carries the table.
                match outline {
                    fieldform_scene::Outline::Closed { .. } => {
                        let (name, tail) = match &under {
                            Some(law) => ("sweep_law_value_normal", format!(", {law}")),
                            None => ("sweep_value_normal", String::new()),
                        };
                        // The miters go in after the section, where the value pass put them --
                        // or the section itself where this law does not mitre, which is what
                        // the value pass wrote and what the shader never reads.
                        let args = match &under {
                            Some(_) => {
                                let m = curves
                                    .iter()
                                    .position(|x| *x == CurveUse::Miters(id))
                                    .unwrap_or(c);
                                let head = format!("{s}u, {f}u, {c}u, ");
                                let rest = args.strip_prefix(&head).expect("begins with these");
                                format!("{head}{m}u, {rest}")
                            }
                            None => args.clone(),
                        };
                        blocks_before
                            .push(format!("    let sw{n} = {name}({args}{tail});\n"));
                    }
                    // The wall's other two polylines after the first, where the value pass
                    // interned them, and its four slots after the sweep's fourteen and the
                    // law's, where a law took any.
                    fieldform_scene::Outline::Wall { .. } => {
                        let rest = |part: usize| {
                            curves
                                .iter()
                                .position(|x| *x == CurveUse::Wall(id, WALL_PARTS[part]))
                                .expect("interned above")
                        };
                        let t = params;
                        params += 4;
                        // `args` begins "spine, frame, chain, "; the faces and the sides follow
                        // the chain, so they are put in after its comma.
                        let head = format!("{s}u, {f}u, {c}u, ");
                        let rest_of = args.strip_prefix(&head).expect("the args begin with these");
                        // The same choice the value pass made, from the same question.
                        let mitred = curves.iter().position(|x| *x == CurveUse::Miters(id));
                        match (&under, mitred) {
                            (Some(law_call), Some(m)) => blocks_before.push(format!(
                                "    let sw{n} = wall_sweep_sharp_value_normal({head}{m}u, \
                                 {rest_of}, {}, {law_call});\n",
                                wall_numbers_as_a_vector(t),
                            )),
                            (Some(law_call), None) => blocks_before.push(format!(
                                "    let sw{n} = wall_sweep_law_value_normal({head}{}u, {}u, \
                                 {c}u, {rest_of}, {}, {law_call});\n",
                                rest(1),
                                rest(2),
                                wall_numbers_as_a_vector(t),
                            )),
                            (None, _) => blocks_before.push(format!(
                                "    let sw{n} = wall_sweep_value_normal({head}{}u, {}u, \
                                 {rest_of}, {});\n",
                                rest(1),
                                rest(2),
                                wall_numbers_as_a_vector(t),
                            )),
                        }
                    }
                }
                // **The rim's slot after everything the sweep counted**, as the value pass
                // pushed it, and the walk's call turned into its `_rim_` twin -- whichever of
                // the five it is.
                if !rims.is_plainly_hard() {
                    let r = params;
                    params += rim_slots(rims);
                    let line = blocks_before.pop().expect("the sweep wrote its walk");
                    let call = line
                        .strip_prefix(&format!("    let sw{n} = "))
                        .and_then(|rest| rest.strip_suffix(";\n"))
                        .expect("the walk is one let");
                    let rim = rim_vector(r, rims, "0.0");
                    blocks_before.push(format!("    let sw{n} = {};\n", rim_call(call, &rim)));
                }
                (format!("sw{n}.w"), format!("sw{n}.xyz"))
            }
            // One walk for value and normal, as a sweep's, and the slots counted exactly as the
            // value pass pushed them: three per section, then the smoothness and the bound.
            Node::Loft { sections, rims, .. } => {
                let s = curves
                    .iter()
                    .position(|x| *x == CurveUse::LoftSpine(id))
                    .expect("interned above");
                let f = curves
                    .iter()
                    .position(|x| *x == CurveUse::Frame(id))
                    .expect("interned above");
                let table = params;
                params += 3 * sections.len();
                let eased = params;
                params += 1;
                let bend = params;
                params += 1;
                if rims.is_plainly_hard() {
                    blocks_before.push(format!(
                        "    let sw{n} = loft_value_normal({s}u, {f}u, {p}, {table}u, {}u, \
                         params[{eased}], params[{bend}]);\n",
                        sections.len(),
                    ));
                } else {
                    // The rim's slot after the bound, as the value pass pushed it.
                    let r = params;
                    params += rim_slots(rims);
                    blocks_before.push(format!(
                        "    let sw{n} = loft_rim_value_normal({s}u, {f}u, {p}, {table}u, {}u, \
                         params[{eased}], params[{bend}], {});\n",
                        sections.len(),
                        rim_vector(r, rims, "0.0"),
                    ));
                }
                (format!("sw{n}.w"), format!("sw{n}.xyz"))
            }
            Node::Unknown(_) => unreachable!("refused in generate, before any lowering"),
        };

        for block in blocks_before.drain(..) {
            node_lines.push(block);
        }
        node_lines.push(format!("    let v{n} = {value};\n    let g{n} = {normal};\n"));
        slot.insert((id, ctx), n);
    }

    // **One function per body here too**, returning the gradient and the value together: the
    // fold below needs both, and a `vec4` carries them without a second call or an out
    // parameter. See `split_into_bodies` for why the split exists at all.
    // The same order the value pass folds in, from the same function — the two are one surface
    // and a body at a different index in each would be one body shaded as another.
    let roots = document.folded_bodies(graph);
    if roots.is_empty() {
        return Err(CodegenError::Empty);
    }
    let returned: Vec<String> = roots
        .iter()
        .map(|id| {
            Ok(format!(
                "vec4<f32>({}, {})",
                normal_of(&slot, (*id, 0))?,
                value_of(&slot, (*id, 0))?
            ))
        })
        .collect::<Result<_, CodegenError>>()?;
    let (bodies_source, calls) =
        split_into_bodies(&body, &node_lines, &returned, "body_normal", "vec4<f32>");
    // Subtree functions first: WGSL wants a function declared before it is called.
    //
    // **And every profile solid read through its `_bake` copy** -- the value a body hands up
    // beside its normal as much as the normal itself -- so the normal pass holds no outline walk
    // and no section walk for the driver to inline. See the profile-solids marker in the
    // prelude; `to_bake_names` leaves a name that is not in the region, and the value pass, as
    // they are.
    let functions = to_bake_names(&format!("{}{bodies_source}", subtrees.concat()), bake_copy_names());

    // The union's normal is the active body's, chosen the same way the boolean above chooses
    // between its operands — by which value is smaller, the earlier body keeping a tie. A
    // running nearest and a statement per body rather than one nested expression: nesting is
    // not only deep here but *quadratic*, since the running minimum appears inside each step's
    // condition as well as inside the next one's. The same box guard as the value fold, in the
    // same order, so a body the value pass skipped is skipped here too and the two passes
    // cannot disagree about which body owns the surface.
    let mut body = String::new();
    if tapers(graph) {
        body.push_str(
            "    crit_gap = 3.4e38;\n    crit_gap = critical_gap(p);\n    taper_hit = false;\n",
        );
    }
    for (i, slot) in boxes.iter().enumerate() {
        if let Some(slot) = slot {
            body.push_str(&format!("    let g{i} = box_guard(p, {slot}u);\n"));
        }
    }
    body.push_str("    var nearest = vec4<f32>(0.0, 0.0, 1.0, 3.4e38);\n");
    // Each body's normal starts with no seam, and the hit body's seam is what the pass keeps:
    // a blend in a body the fold passes over must not leave its scale on the pixel. See
    // `seam_scale` in the prelude.
    body.push_str("    var hit_scale = 1.0;\n");
    for (i, call) in calls.iter().enumerate() {
        let take = format!(
            "seam_scale = 1.0; let r{i} = {call}; if (r{i}.w < nearest.w) {{ nearest = r{i}; \
             hit_scale = seam_scale; }}"
        );
        // **Word for word what the value fold does, including the bound.** A normal is taken from
        // the body the value pass found nearest, so a pass that ruled out a different set -- or
        // that let its running minimum fall to a different number -- would hand back the normal of
        // a surface the depth did not come from. The extent's contribution lowers `nearest.w` here
        // exactly as it lowers `best` there; it never becomes the answer, because `nearest.w` is
        // internal to this fold and only `nearest.xyz` is returned.
        let guarded = boxes
            .get(i)
            .copied()
            .flatten()
            .map(|_| format!("if (g{i} < nearest.w) {{ {take} }}"))
            .unwrap_or_else(|| take.clone());
        match cull {
            Some(_) => body.push_str(&format!(
                "    if (!tile_ruled_out({i}u)) {{ {guarded} }}\n"
            )),
            None => body.push_str(&format!("    {guarded}\n")),
        }
    }
    // **A pixel on a thinning fillet is differenced.** A radius that varies with position adds
    // `(1 − s)·∇gap` to the blend's gradient, and ∇gap is a world vector while the blend may
    // sit under any nesting of placements, mirrors, warps and pattern steps. The blend's own
    // normal says when it was tapering; four more field reads then give a normal that is right
    // under all of them, which is what the CPU does for every blend. Four rather than six: the
    // tetrahedral stencil. The step is sized against f32 at this distance from the origin.
    if tapers(graph) {
        // **`tile_live` is saved and cleared around the stencil** (review point 6). These four
        // reads are the one differencing path left in the renderer, and they are taken at points
        // beside the ray rather than on it -- so a body the tile ruled out can be what is nearest
        // at `p + e`, and a normal differenced from a field missing it is a normal of a different
        // surface. The offsets are small against the cull's world pad, so this is belt and braces;
        // it is kept because "small enough" is a claim about two numbers that may both move, and
        // the saving it gives up is four samples on the pixels of a thinning fillet alone.
        body.push_str(
            "    if (taper_hit) {\n\
             \x20       let was_live = tile_live;\n\
             \x20       tile_live = false;\n\
             \x20       let h = max(0.02, 2e-4 * length(p));\n\
             \x20       let e = vec2<f32>(h, -h);\n\
             \x20       nearest = vec4<f32>(\n\
             \x20           e.xyy * field(p + e.xyy) + e.yyx * field(p + e.yyx)\n\
             \x20           + e.yxy * field(p + e.yxy) + e.xxx * field(p + e.xxx),\n\
             \x20           nearest.w);\n\
             \x20       tile_live = was_live;\n\
             \x20   }\n",
        );
    }
    Ok(format!(
        "{functions}\
         // The field's own normal, not the gradient of its sampled distance. See\n\
         // `normal_function` in codegen.rs for why those differ and why it matters.\n\
         fn field_normal(p: vec3<f32>) -> vec3<f32> {{\n\
         {body}    seam_scale = hit_scale;\n\
         \x20   return normalize(nearest.xyz);\n\
         }}\n"
    ))
}

/// Appends a place's three components and names them as a `vec3`.
///
/// **Three slots whichever kind it is**, because the value pass and the normal pass agree about
/// the parameter buffer by counting — a place that pushed two slots in one shape and three in
/// another would shift every number after it, silently, with the shader still compiling.
///
/// A *typed* place pushes its own references, so a coordinate driven by a named parameter stays
/// live in the buffer and moving it costs a write rather than a recompile. A *taken* one is
/// resolved to numbers here, because the buffer holds numbers and the answer belongs to another
/// block — and moving that block is a graph change, which recompiles anyway.
fn place_slots(
    params: &mut Vec<ParamRef>,
    graph: &Graph,
    document: &fieldform_scene::Document,
    consumer: NodeId,
    place: &fieldform_scene::Place,
) -> String {
    if let Some(xyz) = place.typed() {
        return push_point(params, &xyz);
    }
    let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
    // Through the same reading the field uses, consumer and all -- a place resolved in a
    // different frame here would put the shader's surface somewhere the field says it is not,
    // which is exactly what `gpu_matches_cpu` exists to catch.
    let at = fieldform_field::resolve_place_for(document, graph, consumer, place, &numbers)
        .unwrap_or_else(|_| nalgebra::Point3::origin());
    push_point(
        params,
        &[
            ParamRef::Literal(at.x),
            ParamRef::Literal(at.y),
            ParamRef::Literal(at.z),
        ],
    )
}

/// The same, for a placement's pivot, which is read in the frame *outside* that placement.
///
/// Separate from [`place_slots`] by one call, and it has to be: a placement's own numbers are
/// written in the frame its parent establishes, not in the frame it produces, so resolving a
/// pivot the way a sphere's centre is resolved would come back with this placement already
/// applied to it -- the placement the pivot is there to define.
fn pivot_slots(
    params: &mut Vec<ParamRef>,
    graph: &Graph,
    document: &fieldform_scene::Document,
    placement: NodeId,
    about: &fieldform_scene::Place,
) -> String {
    if let Some(xyz) = about.typed() {
        return push_point(params, &xyz);
    }
    let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
    let at = fieldform_field::resolve_pivot(document, graph, placement, about, &numbers)
        .unwrap_or_else(|_| nalgebra::Point3::origin());
    push_point(
        params,
        &[
            ParamRef::Literal(at.x),
            ParamRef::Literal(at.y),
            ParamRef::Literal(at.z),
        ],
    )
}

/// Appends a primitive's three centre components and names them as a `vec3`.
///
/// The centre goes through the parameter buffer like every other number, so moving a sphere
/// costs a buffer write rather than a shader recompile.
fn push_point(params: &mut Vec<ParamRef>, centre: &[ParamRef; 3]) -> String {
    let first = params.len();
    params.extend_from_slice(centre);
    format!("vec3<f32>(params[{first}], params[{}], params[{}])", first + 1, first + 2)
}

/// Where an extrude's profile plane is and which way it is swept, in nine parameter slots.
///
/// **Nine either way**, so the value pass and the normal pass agree by counting rather than by
/// reasoning: three for the plane's origin, three for its rotation, three for the direction. The
/// straight sweep uploads `(0, 0, 1)` and the WGSL reduces to what it always did.
///
/// The plane is worked out from the profile's own points, not read from the node — there is
/// nothing on the node to read any more, which is the point of the change. It is resolved to
/// numbers here because the shader has no curves in it.
fn extrude_placement(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    outline: &fieldform_scene::Outline,
    sweep: &fieldform_scene::Sweep,
) -> (String, String, String) {
    // Asked of the function the field compiles from, so that the plane this places the solid
    // in is the plane the field laid it in — for a profile's loop and a wall's chain alike.
    let (origin, normal) = fieldform_field::extrude_section(
        document,
        outline,
        sweep,
        fieldform_field::PROFILE_TURN_DEGREES,
    )
    .map(|section| (section.origin, section.normal))
    .unwrap_or((nalgebra::Point3::origin(), nalgebra::Vector3::z()));
    let rotation = fieldform_field::rotation_from_z(normal);
    let (rx, ry, rz) = rotation.euler_angles();

    // In the profile's own frame, where the straight sweep is `+Z` by construction.
    let local = match sweep {
        fieldform_scene::Sweep::Normal => nalgebra::Vector3::z(),
        other => sweep_world_direction(document, other)
            .map(|d| rotation.inverse_transform_vector(&d))
            .and_then(|d| d.try_normalize(1e-12))
            .unwrap_or_else(nalgebra::Vector3::z),
    };

    let lit = |v: f64| ParamRef::Literal(v);
    let at = push_point(params, &[lit(origin.x), lit(origin.y), lit(origin.z)]);
    let euler = push_point(
        params,
        &[lit(rx.to_degrees()), lit(ry.to_degrees()), lit(rz.to_degrees())],
    );
    let dir = push_point(params, &[lit(local.x), lit(local.y), lit(local.z)]);
    (at, euler, dir)
}

/// How many slots an extrude's depth takes after its placement: `ExtrudeDepth`'s six.
const DEPTH_SLOTS: usize = 6;

/// Pushes the six slots that say how an extrude's depth lies, in the order the shader's
/// `ExtrudeDepth` takes them, and hands back the expression that builds it.
///
/// Side one's depth, the tangent of its draft, which way the depth lies as a number -- nought
/// for one side, one for half each way, two for two sides -- where the material begins, and
/// side two's depth and draft tangent, nought where there is no side two. **Every one a number
/// rather than a generated branch**, so ticking a side or typing a start refills a buffer
/// instead of rebuilding the shader, which is the reason every other number here is a
/// `params[..]` read. The tangents are worked out here because the shader wants a slope while
/// the document holds an angle, and a buffer carrying degrees would need the conversion at
/// every sample.
///
/// **One function for the single extrude and for a member of a run**, as `wall_slots` is: a
/// run's slots are the same numbers pushed again, and two places pushing them is two orders
/// to keep the same.
fn depth_slots(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    start: &ParamRef,
    height: &ParamRef,
    draft: &ParamRef,
    sides: &fieldform_scene::Sides,
) -> String {
    let first = params.len();
    let tangent = |draft: &ParamRef| ParamRef::Literal(resolved(document, draft).to_radians().tan());
    params.push(*height);
    params.push(tangent(draft));
    let (which, other_height, other_draft) = match sides {
        fieldform_scene::Sides::One => (0.0, ParamRef::Literal(0.0), ParamRef::Literal(0.0)),
        fieldform_scene::Sides::Symmetric => (1.0, ParamRef::Literal(0.0), ParamRef::Literal(0.0)),
        fieldform_scene::Sides::Two { height, draft } => (2.0, *height, tangent(draft)),
    };
    params.push(ParamRef::Literal(which));
    params.push(*start);
    params.push(other_height);
    params.push(other_draft);
    debug_assert_eq!(params.len() - first, DEPTH_SLOTS);
    depth_read(first)
}

/// The `ExtrudeDepth` built from six consecutive slots, for the pass that counts rather than
/// pushes.
fn depth_read(first: usize) -> String {
    format!(
        "ExtrudeDepth(params[{first}], params[{}], params[{}], params[{}], params[{}], params[{}])",
        first + 1,
        first + 2,
        first + 3,
        first + 4,
        first + 5,
    )
}

/// The three polylines a wall is read from, in the order they are bound and passed.
const WALL_PARTS: [fieldform_field::WallPart; 3] = [
    fieldform_field::WallPart::Chain,
    fieldform_field::WallPart::Faces,
    fieldform_field::WallPart::Sides,
];

/// What a wall's shader call is told about it besides its thickness: which way the band lies
/// from the curve, whether its ends are round, and whether the chain closes.
///
/// **Read from the wall the field builds**, because two of the three are not on the node.
/// Whether a chain closes is a fact about where its curves' ends are, and which way a loop's
/// band lies depends on how the loop winds; both are settled where the wall is made, and asking
/// there is what stops this from settling them differently. `+1` is to the left of an open
/// chain and outward from a loop, which is the sign the shader's band is written in.
///
/// A wall that cannot be built is refused by the field before anything is drawn, so what is
/// sent for one is only what the node itself says.
fn wall_numbers(
    document: &fieldform_scene::Document,
    graph: &Graph,
    id: NodeId,
    outline: &fieldform_scene::Outline,
) -> [f64; 3] {
    use fieldform_scene::{WallEnds, WallSide};
    let flag = |on: bool| if on { 1.0 } else { 0.0 };
    match fieldform_field::outline_wall(
        document,
        graph,
        id,
        fieldform_field::PROFILE_TURN_DEGREES,
    ) {
        Some(wall) => {
            let (lo, hi) = wall.band();
            let facing = if lo + hi > 0.0 {
                1.0
            } else if lo + hi < 0.0 {
                -1.0
            } else {
                0.0
            };
            [facing, flag(wall.ends() == Some(WallEnds::Round)), flag(wall.is_closed())]
        }
        None => match outline {
            fieldform_scene::Outline::Wall { side, ends, .. } => [
                match side {
                    WallSide::Centre => 0.0,
                    WallSide::Left => 1.0,
                    WallSide::Right => -1.0,
                },
                flag(*ends == WallEnds::Round),
                0.0,
            ],
            fieldform_scene::Outline::Closed { .. } => [0.0; 3],
        },
    }
}

/// A wall's four slots written as the one `vec4` the revolve's and the sweep's calls take them
/// as, from the place `wall_slots` said the first was pushed.
fn wall_numbers_as_a_vector(first: usize) -> String {
    format!(
        "vec4<f32>(params[{first}], params[{}], params[{}], params[{}])",
        first + 1,
        first + 2,
        first + 3
    )
}

/// One if a thin revolve's band reaches across its axis, so the shader reads it from both
/// sides, and nought if it keeps to one. Asked of the wall the field builds, as `wall_numbers`
/// asks, so the shader and `Footprint::about_axis` take the second reading for the same walls.
fn wall_crosses_the_axis(
    document: &fieldform_scene::Document,
    graph: &Graph,
    id: NodeId,
) -> f64 {
    let crosses =
        fieldform_field::outline_wall(document, graph, id, fieldform_field::PROFILE_TURN_DEGREES)
            .is_some_and(|wall| fieldform_field::Footprint::Wall(wall).crosses_the_axis());
    if crosses { 1.0 } else { 0.0 }
}

/// Pushes a partial revolve's two slots -- where its wedge begins and how far it turns, in
/// radians -- and says where the first is, or pushes nothing and says so for a full turn.
///
/// From the same reading the compile makes (`revolve_wedge`, `Profile::azimuth`): the wedge
/// begins at the section's own half-plane, back by whatever the sides say, and that half-plane
/// is where the section's points are about the frame's axis. **Numbers in slots, so dragging
/// an angle refills a buffer**; the one text change is at a full turn, where the call is the
/// one every revolve always had. A revolve that cannot be read -- no section, an angle that is
/// not a number -- is refused by the compile, so what is pushed for one is only a full turn.
fn wedge_slots(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    graph: &Graph,
    id: NodeId,
    angle: &ParamRef,
    sides: &fieldform_scene::AngleSides,
) -> Option<usize> {
    let (from, total) = fieldform_field::revolve_wedge(document, angle, sides).ok()?;
    if total >= 360.0 - 1e-9 {
        return None;
    }
    let Some(Node::Revolve { outline, frame, .. }) = graph.node(id) else { return None };
    let numbers = |r: &ParamRef| document.resolve(r);
    let (translation, rotation) =
        fieldform_field::resolve_frame(document, graph, frame, &numbers).ok()?;
    let into = move |p: nalgebra::Point3<f64>| {
        nalgebra::Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
    };
    let points = fieldform_field::outline_chain(document, outline.curve(), frame.curve_source())
        .ok()?
        .points;
    let starts_at = fieldform_field::Profile::azimuth(&points, into);
    let first = params.len();
    params.push(ParamRef::Literal(starts_at + from.to_radians()));
    params.push(ParamRef::Literal(total.to_radians()));
    Some(first)
}

/// Pushes the four slots a wall adds to an extrude's twelve, and says where the first is.
///
/// The thickness as the document holds it, then `wall_numbers`' three: the order
/// `wall_extrude_distance` takes them in, and the order `wall_extrude_run_distance` reads them
/// at `slot + 15` to `slot + 18`. **One function for the single extrude and for a member of a
/// run**, because a run's slots are the same numbers pushed a second time, and two places
/// pushing them is two orders to keep the same.
fn wall_slots(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    graph: &Graph,
    id: NodeId,
    outline: &fieldform_scene::Outline,
) -> usize {
    let first = params.len();
    // A thickness driven through space has no one value to send. The field refuses it by name
    // and nothing is drawn, so this slot is only ever read for a thickness that is a number.
    params.push(match outline {
        fieldform_scene::Outline::Wall { thickness, .. } if thickness.driver().is_none() => {
            *thickness
        }
        _ => ParamRef::Literal(0.0),
    });
    for number in wall_numbers(document, graph, id, outline) {
        params.push(ParamRef::Literal(number));
    }
    first
}

/// Whether an extrude's walls are upright on both sides, so that there is no offset for the
/// corners to be cut either way.
///
/// **Asked so that a block with Sharp set and no draft compiles to the call it always had.**
/// The two answers are the same solid with no draft (`with_no_draft_the_two_are_one_solid`),
/// and binding a polyline to say so would cost a buffer and a slot for nothing.
fn extrude_is_upright(
    document: &fieldform_scene::Document,
    draft: &ParamRef,
    sides: &fieldform_scene::Sides,
) -> bool {
    let flat = |r: &ParamRef| resolved(document, r) == 0.0;
    flat(draft)
        && match sides {
            fieldform_scene::Sides::Two { draft, .. } => flat(draft),
            _ => true,
        }
}

/// The miters binding a block needs, or `None` when its draft rounds its corners and there is
/// nothing to displace.
///
/// **Interned in the value pass and found by the normal pass**, exactly as a profile and a
/// wall are: the two passes must agree about which bindings exist and in what order, and the
/// one way to be sure is for both to ask the same question of the same node.
fn miters_binding(
    curves: &mut Vec<CurveUse>,
    id: NodeId,
    corners: fieldform_scene::Corners,
    draft_free: bool,
) -> Option<usize> {
    if corners.is_default() || draft_free {
        return None;
    }
    Some(intern(curves, CurveUse::Miters(id)))
}

/// Whether a sweep's law actually mitres anything: a draft, and corners set sharp.
///
/// **Asked the same way by both passes**, from the node alone, so the two agree about whether
/// the binding exists without either counting. A scale moves no wall along its own normal, so
/// there is nothing for a miter vector to displace and nothing is bound.
fn law_mitres(law: &fieldform_scene::SweepLaw) -> bool {
    !law.corners.is_default() && matches!(law.taper, fieldform_scene::Taper::Draft(_))
}

/// The six slots a section law needs, and the `SweepLaw(...)` that reads them -- or `None` for
/// the plain law, which is what keeps a sweep that has not asked byte-identical.
///
/// **Slots rather than literals, every one**: a taper, a twist and an extent are all numbers
/// somebody drags, and a literal would rebuild the shader at the frame rate. The window is in
/// millimetres of arc rather than percent, because that is what the landing's own arc is
/// measured in and converting it per sample would be arithmetic in the hot path.
///
/// Degrees become radians and a draft its tangent here, exactly as `fieldform_field::sweep_law`
/// does for the kernel -- the two sides read one document and must agree about what a number
/// means.
fn law_slots(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    law: &fieldform_scene::SweepLaw,
) -> Option<String> {
    if law.is_plain() {
        return None;
    }
    let Ok(read) = fieldform_field::sweep_law(document, law) else {
        // Unresolvable numbers; the compile refuses this node by name and nothing is drawn.
        return None;
    };
    let _ = outline;
    // The window in millimetres, from the path the field flattens -- the same length
    // `Swept::under_law` divides by, so the two windows are one window.
    let length = document
        .curves
        .get(&spine)
        .map(|c| {
            fieldform_field::Spine::of_curve(c, fieldform_field::DEFAULT_TOLERANCE_MM).length()
        })
        .unwrap_or(0.0);
    let (kind, amount) = match read.taper {
        fieldform_field::Taper::None => (0.0, 0.0),
        fieldform_field::Taper::Scale { end } => (1.0, end),
        fieldform_field::Taper::Draft { tangent } => (2.0, tangent),
    };
    let shape = match law.shape {
        fieldform_scene::RampShape::Linear => 0.0,
        fieldform_scene::RampShape::Smooth => 1.0,
        fieldform_scene::RampShape::Smoother => 2.0,
    };
    let first = params.len();
    let numbers = [
        kind,
        amount,
        read.twist,
        shape,
        read.window.0 * length,
        read.window.1 * length,
        match law.corners {
            fieldform_scene::Corners::Round => 0.0,
            fieldform_scene::Corners::Sharp => 1.0,
        },
    ];
    debug_assert_eq!(numbers.len(), LAW_SLOTS, "the normal pass counts this many");
    for number in numbers {
        params.push(ParamRef::Literal(number));
    }
    Some(format!(
        "SweepLaw(params[{first}], params[{}], params[{}], params[{}], params[{}], params[{}], \
         params[{}])",
        first + 1,
        first + 2,
        first + 3,
        first + 4,
        first + 5,
        first + 6,
    ))
}

/// The normal pass's half of [`law_slots`]: the same six slots counted rather than pushed,
/// and the same `SweepLaw(...)` to read them.
///
/// **Two functions because the two passes keep their slots differently** -- the value pass
/// owns the `params` vector and pushes, the normal pass is handed a count and advances it --
/// and the one thing they must agree about is how many, which is why the number is written
/// once here as the array's own length.
fn law_reader(params: &mut usize, law: &fieldform_scene::SweepLaw) -> Option<String> {
    if law.is_plain() {
        return None;
    }
    let first = *params;
    *params += LAW_SLOTS;
    Some(format!(
        "SweepLaw(params[{first}], params[{}], params[{}], params[{}], params[{}], params[{}], \
         params[{}])",
        first + 1,
        first + 2,
        first + 3,
        first + 4,
        first + 5,
        first + 6,
    ))
}

/// How many slots a section law takes: a taper's kind and amount, a twist, a shape, the
/// window's two ends, and whether a draft mitres its corners.
const LAW_SLOTS: usize = 7;

/// Which way a stated sweep runs, in world coordinates.
///
/// `Normal` is not answered here: it is derived from the profile, and only the caller holding the
/// plane can say what it is.
fn sweep_world_direction(
    document: &fieldform_scene::Document,
    sweep: &fieldform_scene::Sweep,
) -> Option<nalgebra::Vector3<f64>> {
    match sweep {
        fieldform_scene::Sweep::Normal => None,
        fieldform_scene::Sweep::Along { curve } => {
            let c = document.curves.get(curve)?;
            let (from, to) = c.ends();
            (to - from).try_normalize(1e-12)
        }
        fieldform_scene::Sweep::Toward { xyz } => {
            let mut out = nalgebra::Vector3::zeros();
            for (slot, r) in out.iter_mut().zip(xyz) {
                *slot = resolved(document, r);
            }
            out.try_normalize(1e-12)
        }
    }
}

/// The two points of an [`Ends`], as parameter slots.
///
/// **Six slots either way, and that is the point.** Ends taken from a curve are resolved here
/// and uploaded as plain numbers, because the shader has no curves in it — but they take the same
/// six slots typed ones would, so the normal pass can go on counting `point_slots` twice without
/// knowing which kind it is looking at. The two passes agree by walking the same nodes in the
/// same order; a variant that consumed a different number of slots would break that silently, and
/// the symptom would be every parameter after it reading the one next door.
fn push_ends(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    graph: &Graph,
    consumer: NodeId,
    ends: &fieldform_scene::Ends,
) -> (String, String) {
    match ends {
        // Each end is a place, and each is three slots whichever kind -- see `place_slots`.
        fieldform_scene::Ends::At { from, to } => (
            place_slots(params, graph, document, consumer, from),
            place_slots(params, graph, document, consumer, to),
        ),
        fieldform_scene::Ends::From { curve } => {
            let (a, b) = document
                .curves
                .get(curve)
                .map(|c| c.ends())
                .unwrap_or((nalgebra::Point3::origin(), nalgebra::Point3::origin()));
            let literal = |p: nalgebra::Point3<f64>| {
                [
                    ParamRef::Literal(p.x),
                    ParamRef::Literal(p.y),
                    ParamRef::Literal(p.z),
                ]
            };
            (
                push_point(params, &literal(a)),
                push_point(params, &literal(b)),
            )
        }
        // Standing on a block that has a facing. Resolved here and uploaded as numbers, the same
        // as a curve's ends and for the same reason: the shader has no planes in it either, and
        // both still take the six slots typed ends would.
        fieldform_scene::Ends::On { .. } => {
            let numbers = |r: &ParamRef| -> f64 { document.resolve(r).unwrap_or(0.0) };
            let (a, b) = fieldform_field::ends_at(ends, document, graph, &numbers);
            let literal = |p: nalgebra::Point3<f64>| {
                [
                    ParamRef::Literal(p.x),
                    ParamRef::Literal(p.y),
                    ParamRef::Literal(p.z),
                ]
            };
            (
                push_point(params, &literal(a)),
                push_point(params, &literal(b)),
            )
        }
    }
}

/// The same as [`push_point`], for the normal pass, which counts slots rather than collecting
/// them — it walks the identical node order, so the indices agree by construction.
fn point_slots(params: &mut usize, _centre: &[ParamRef; 3]) -> String {
    let first = *params;
    *params += 3;
    format!("vec3<f32>(params[{first}], params[{}], params[{}])", first + 1, first + 2)
}

/// The slots an [`Ends`] occupies in the normal pass.
///
/// Always the two points' worth, whichever kind it is — the counterpart of `push_ends`, which
/// uploads six either way. Written as its own function rather than two `point_slots` calls at
/// each site so that the pairing with `push_ends` is one thing to keep true rather than three.
fn ends_slots(params: &mut usize) -> (String, String) {
    // `point_slots` reads nothing from what it is handed; it counts.
    let unused = [ParamRef::Literal(0.0); 3];
    (point_slots(params, &unused), point_slots(params, &unused))
}

/// Where every node has to be evaluated, allowing for a node reached through more than one
/// point.
///
/// # Why this is a tree of *contexts* and not one point per node
///
/// A `Transform`, `Mirror` or `Twist` changes the point its subtree is sampled at. If a node is
/// reached both directly and through one of those, it genuinely needs evaluating twice, at two
/// different arguments — which is what the CPU field does by recursing. An earlier version of
/// this gave each node a single point and refused the case; that refusal is what made
/// `body ∪ mirror(body)` — a mirror's *main* use — impossible to express.
///
/// So a **point context** is a path rather than an index: context 0 is the point the field was
/// called with, and every occurrence of a point-changing node under some context creates a new
/// one. The same transform reached through two different parents produces two contexts, which
/// is the case one index per node could not represent.
///
/// Returns the contexts in declaration order — each names its parent, so `p{i+1}` can be
/// emitted from `p{parent}` — and the `(node, context)` pairs to emit, ordered so that every
/// operand is defined before whatever refers to it.
fn point_assignment(
    // Unused since a pattern stopped declaring one context per step of its window: the window's
    // reach came from the document, and it is a parameter now. Kept in the signature because
    // every other stage of the walk takes the document and a lone exception is a thing to
    // wonder about later.
    document: &fieldform_scene::Document,
    graph: &Graph,
    order: &[NodeId],
) -> Result<PointPlan, CodegenError> {
    // Each declared point: what it is derived from, which node derives it, and which step of
    // that node's window it is. The offset is zero for everything but a pattern, which is the
    // only node that samples its subtree at more than one point without being more than one
    // node -- see `pattern_window`.
    let mut contexts: Vec<(usize, NodeId, isize)> = Vec::new();
    // The contexts a point-changer hands *down*, per occurrence of it. A list rather than one,
    // because a pattern hands down its whole window and its value is the nearest of them.
    let mut below: HashMap<(NodeId, usize), Vec<usize>> = HashMap::new();
    let mut pairs: HashSet<(NodeId, usize)> = HashSet::new();

    // Seeded in the document's body order for the reason the folds are: both passes walk this
    // plan and count their slots in step, so the numbering has to be settled by something that
    // does not move when an unrelated body is created.
    let mut stack: Vec<(NodeId, usize)> =
        document.folded_bodies(graph).into_iter().map(|id| (id, 0)).collect();
    // **A critical surface is evaluated at the world point as well as wherever it sits.** The
    // gap a fillet tapers by is pointwise in world space (spec §6.4), and a marked node under a
    // placement is otherwise only ever sampled at a local one. A context is a path, so this is
    // one more occurrence of the node at context 0 -- the mechanism a joined mirror already
    // uses -- and both passes walk this plan, so they still count their slots in step.
    if tapers(graph) {
        stack.extend(graph.critical().iter().map(|mark| (mark.node, 0)));
    }
    while let Some((id, ctx)) = stack.pop() {
        if !pairs.insert((id, ctx)) {
            continue;
        }
        let node = graph.node(id).ok_or(CodegenError::DanglingNode(id))?;
        let inner: Vec<usize> = if changes_the_point(node) {
            below
                .entry((id, ctx))
                .or_insert_with(|| {
                    // **One context, even for a pattern.** A pattern used to declare its
                    // whole window here -- one point per step -- and the subtree below it was
                    // then emitted once per point. That is what made a single patterned body
                    // take a minute to compile: the code multiplied by the window while the
                    // shape did not. It now declares one context, standing for "the step being
                    // considered", and the window becomes a loop at run time. See the pattern
                    // arms in both passes, and backlog item 34.
                    std::iter::once(0)
                        .map(|offset| {
                            contexts.push((ctx, id, offset));
                            // Context 0 is the caller's own point, so the first is 1.
                            contexts.len()
                        })
                        .collect()
                })
                .clone()
        } else {
            vec![ctx]
        };
        for input in node.inputs() {
            for c in &inner {
                stack.push((input, *c));
            }
        }
        // A joined mirror needs its input **twice**: reflected, and where the body already is,
        // because the union has both halves. This is the case one point per node could not
        // express, and the reason a mirror was two blocks until contexts existed.
        //
        // Unless the body never crosses its own plane, and then it needs it once: the point is
        // folded onto the body's side instead of reflected, and the half the `min` would have
        // chosen is the only one written. That halves the subtree — two spine walks for a
        // mirrored sweep instead of four — and it is the same number, not a near one. See
        // `fieldform_field::mirror_one_side`.
        if let Node::Mirror { input, join: true, .. } = node
            && fieldform_field::mirror_one_side(document, graph, id).is_none()
            // An empty slot is no edge, here as everywhere: `inputs` above leaves one out, and
            // this second push of the same slot leaves it out for the same reason -- the plan is
            // what the refusal is read off, so it is built over a holed graph too.
            && let Some(source) = input.source()
        {
            stack.push((source, ctx));
        }
    }

    // Evaluation order puts operands before their consumers, and a pair's operands are always
    // pairs of nodes earlier in it — so sorting the pairs by that position is a valid order for
    // all of them at once, whatever contexts they are in.
    let rank: HashMap<NodeId, usize> =
        order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut emit: Vec<(NodeId, usize)> = pairs.into_iter().collect();
    emit.sort_by_key(|(id, ctx)| (rank.get(id).copied().unwrap_or(usize::MAX), *ctx));

    Ok(PointPlan { contexts, below, emit })
}

/// Whether this part's fillets taper: something is marked critical, and something is rounded.
///
/// Decided from the graph's *shape* -- a fillet that is not plainly hard -- and not from a
/// radius's value, so dragging a radius never changes the answer and never rebuilds the shader.
/// Adding or removing a mark does, which is right: it changes what is evaluated.
fn tapers(graph: &Graph) -> bool {
    !graph.critical().is_empty()
        && graph.nodes().any(|(_, n)| n.fillet().is_some_and(|f| !f.is_plainly_hard()))
}

/// Whether a node samples its subtree somewhere other than its own point.
fn changes_the_point(node: &Node) -> bool {
    matches!(
        node,
        Node::Transform { .. }
            | Node::Mirror { .. }
            | Node::Twist { .. }
            | Node::Attract { .. }
            | Node::Pattern { .. }
    )
}

/// Everything about a pattern's copies the shader is handed, resolved once from its layout.
///
/// **One answer, asked of the field crate.** Where the copies are, how far apart, which exist and
/// where the fold counts them from are all [`PatternLayout`]'s, which the field compiles from too;
/// both passes of the shader ask this, which is a pure function of the document, so they cannot
/// part. The same discipline `window_of` and the origin always had, with more to agree about now
/// that a pattern has options.
struct PatternNumbers {
    layout: fieldform_field::PatternLayout,
    /// Where the body sits along the fold, with the steps to copy 0 included.
    origin: f64,
    /// A row's gap in millimetres and its direction; a ring has none.
    pitch: f64,
    unit: [f64; 3],
    /// A ring's angle between neighbours in radians; a row has none.
    step: f64,
    /// Steps from the body to copy 0: nought unless the pattern is centred.
    bias: f64,
    /// Whether a ring stops short of a full turn, which changes how a point finds its copy.
    arc: bool,
    /// The lowest copy the block holds: one when the only copy left out is the first -- which is
    /// what leaving out the body always meant -- and nought otherwise.
    first: f64,
    /// How many copies the block leaves out, which the rounded fold's divisor reads.
    left_out: f64,
    /// The skip list as `f32`-exact words, 24 copies to a word, when more than the first copy is
    /// left out. `None` for every pattern that was expressible before options, which therefore
    /// generates exactly the loop it always did.
    words: Option<Vec<f64>>,
    /// A grid's own numbers; `None` for a row and a ring.
    grid: Option<GridNumbers>,
}

fn pattern_numbers(
    document: &fieldform_scene::Document,
    graph: &Graph,
    node: NodeId,
) -> PatternNumbers {
    let layout = fieldform_field::PatternLayout::of_or_inert(document, graph, node);
    let Some(Node::Pattern { input, .. }) = graph.node(node) else {
        panic!("a pattern's numbers were asked of something that is not a pattern");
    };
    let input = source_of(*input);
    let origin = fieldform_field::pattern_origin_in(document, graph, input, &layout);
    let (pitch, unit, step, arc) = match &layout.shape {
        fieldform_field::PatternShape::Row { step, .. } => {
            let pitch = step.norm();
            let unit = step.unscale(pitch.max(1e-12));
            (pitch, [unit.x, unit.y, unit.z], 0.0, false)
        }
        fieldform_field::PatternShape::Ring { step, arc, .. } => (0.0, [0.0; 3], *step, *arc),
        fieldform_field::PatternShape::Grid { .. } => (0.0, [0.0; 3], 0.0, false),
    };
    let grid = match &layout.shape {
        fieldform_field::PatternShape::Grid { step, count, bias, step2, count2, bias2 } => {
            let dual = fieldform_field::grid_dual(step, step2)
                .expect("a layout refuses two steps along one line");
            let anchor = fieldform_field::pattern_grid_anchor_in(document, graph, input, &layout);
            let radius = match graph.node(node) {
                Some(n) => fillet_radius(document, n),
                None => 0.0,
            };
            let disp0 = step * *bias + step2 * *bias2;
            let v = |x: &nalgebra::Vector3<f64>| [x.x, x.y, x.z];
            Some(GridNumbers {
                anchor: [anchor.x, anchor.y, anchor.z],
                disp0: v(&disp0),
                e1: v(step),
                e2: v(step2),
                d1: v(&dual[0]),
                d2: v(&dual[1]),
                columns: f64::from(*count),
                rows: f64::from(*count2),
                reach: fieldform_field::pattern_reach_axes_in(document, graph, input, &layout, radius),
            })
        }
        _ => None,
    };
    let only_the_first = layout.excluded.iter().skip(1).all(|e| !*e);
    let (first, words) = if only_the_first {
        (if layout.excluded[0] { 1.0 } else { 0.0 }, None)
    } else {
        let mut words = vec![0.0f64; (layout.total as usize).div_ceil(24)];
        for (i, left_out) in layout.excluded.iter().enumerate() {
            if *left_out {
                words[i / 24] += f64::from(1u32 << (i % 24));
            }
        }
        (0.0, Some(words))
    };
    PatternNumbers {
        origin,
        pitch,
        unit,
        step,
        bias: layout.bias(),
        arc,
        first,
        left_out: layout.left_out() as f64,
        words,
        grid,
        layout,
    }
}

/// What a grid is handed beyond what any pattern is: where copy 0 stands, the two steps and their
/// duals, the counts and the window along each direction. Vectors in millimetres, the rest numbers.
struct GridNumbers {
    anchor: [f64; 3],
    /// From the body to copy 0: the placement a copy's number is added to.
    disp0: [f64; 3],
    e1: [f64; 3],
    e2: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    columns: f64,
    rows: f64,
    reach: [usize; 2],
}

/// The slots a pattern's options add after the five it always had, in the order both passes agree on.
struct PatternExtras {
    left_out: usize,
    bias: usize,
    step: usize,
    arc: usize,
    skip: Option<usize>,
}

/// Pushes them for the value pass: how many copies are left out, the steps to copy 0, a ring's
/// angle between neighbours, whether it is an arc, and then the skip words if there are any.
///
/// All four numbers go in for a row and a ring alike, so the layout of the slots does not depend on
/// which kind it is; a row simply never reads the last two.
fn push_pattern_extras(params: &mut Vec<ParamRef>, n: &PatternNumbers) -> PatternExtras {
    let left_out = params.len();
    params.push(ParamRef::Literal(n.left_out));
    let bias = params.len();
    params.push(ParamRef::Literal(n.bias));
    let step = params.len();
    params.push(ParamRef::Literal(n.step));
    let arc = params.len();
    params.push(ParamRef::Literal(if n.arc { 1.0 } else { 0.0 }));
    let skip = n.words.as_ref().map(|words| {
        let at = params.len();
        params.extend(words.iter().map(|w| ParamRef::Literal(*w)));
        at
    });
    PatternExtras { left_out, bias, step, arc, skip }
}

/// The same slots, counted by the normal pass, which pushes nothing: **the two passes count
/// parameters in step with each other**, so a slot added to one and not the other shifts every
/// number after it, silently.
fn count_pattern_extras(params: &mut usize, n: &PatternNumbers) -> PatternExtras {
    let left_out = *params;
    *params += 1;
    let bias = *params;
    *params += 1;
    let step = *params;
    *params += 1;
    let arc = *params;
    *params += 1;
    let skip = n.words.as_ref().map(|words| {
        let at = *params;
        *params += words.len();
        at
    });
    PatternExtras { left_out, bias, step, arc, skip }
}

/// A pattern's fillet radius in millimetres, or nought where there is none.
///
/// Through the document, so a radius that names a parameter sizes the window the way a typed one
/// does — the same rule every other number here is read by.
fn fillet_radius(document: &fieldform_scene::Document, node: &Node) -> f64 {
    match node {
        Node::Pattern { fillet, .. } => document.resolve(&fillet.radius).unwrap_or(0.0).max(0.0),
        _ => 0.0,
    }
}

/// Which steps of its window a node samples its subtree at.
///
/// One, for everything that moves the point once. A pattern is the exception and the reason this
/// exists: its instances are a regular lattice of one body, so the nearest one is found by
/// folding the point into a cell rather than by trying them all — and the window around that
/// cell is a handful of steps whatever the count is.
///
/// **This is what stops a pattern's shader growing with its count.** Every instance used to be
/// written out, so a hundred copies emitted the body a hundred times and a thousand emitted it a
/// thousand — this codebase has already met the end of that road once, where two thousand bodies
/// simulated to a twenty-two-million-character expression. The window is decided by the body's
/// own size, not by how many there are.
///
/// The reach, how many cells either side of a point's own the loop walks, is the field's own
/// answer, `pattern_reach_in`, asked rather than repeated, because the field and the shader
/// disagreeing about it would show as material missing from a picture the field says is solid.
fn window_of(
    document: &fieldform_scene::Document,
    graph: &Graph,
    node_id: NodeId,
    layout: &fieldform_field::PatternLayout,
) -> std::ops::RangeInclusive<isize> {
    let reach = match graph.node(node_id) {
        Some(node @ Node::Pattern { input, .. }) => fieldform_field::pattern_reach_in(
            document,
            graph,
            source_of(*input),
            layout,
            fillet_radius(document, node),
        ) as isize,
        _ => 0,
    };
    -reach..=reach
}

/// How a pattern's step point is built, so the loop that walks its window can build one per step.
///
/// The step itself is not here: it is the loop's own variable, which is the whole point. Every
/// other ingredient is a parameter slot, so the shader does not change when a count does.
enum PatternPoint {
    /// `bias` is the slot holding the steps from the body to copy 0, which the origin already
    /// includes for finding a point's copy and the slide has to include for placing it.
    Linear { unit: String, pitch: usize, count: usize, origin: usize, bias: usize },
    /// `step` is the angle between neighbours and `arc` whether the ring stops short of a turn,
    /// both slots; `is_arc` says the same thing to the generator, which has to choose between a loop
    /// that wraps and one that does not, and cannot read a slot.
    Radial {
        at: String,
        euler: String,
        count: usize,
        step: usize,
        origin: usize,
        bias: usize,
        arc: usize,
        is_arc: bool,
    },
    /// Rows and columns. Seven vectors' worth of slots and the counts; `count` is the *total*, as it
    /// is for the others, and `columns`/`rows` the two directions'. `reach` is the window along each
    /// and `whole` says, for each direction, that the window covers it — decided here because the
    /// loop that walks a covered direction is a different loop.
    Grid {
        anchor: String,
        disp0: String,
        e1: String,
        e2: String,
        d1: String,
        d2: String,
        count: usize,
        columns: usize,
        rows: usize,
        reach: [usize; 2],
        whole: [bool; 2],
    },
}

impl PatternPoint {
    /// The slot holding how many instances there are, which the window needs to stay inside.
    fn count(&self) -> usize {
        match self {
            PatternPoint::Linear { count, .. }
            | PatternPoint::Radial { count, .. }
            | PatternPoint::Grid { count, .. } => *count,
        }
    }

    /// Instance `index` applied to the parent's point, with no window and no fold.
    fn instance_at(&self, parent: usize, index: &str) -> String {
        match self {
            PatternPoint::Linear { unit, pitch, bias, .. } => format!(
                "pattern_slide_linear(p{parent}, {unit}, params[{pitch}], {index} + params[{bias}])"
            ),
            PatternPoint::Radial { at, euler, step, bias, .. } => format!(
                "pattern_spin_radial(p{parent}, {at}, {euler}, params[{step}], params[{bias}], \
                 {index})"
            ),
            PatternPoint::Grid { disp0, e1, e2, columns, .. } => format!(
                "pattern_slide_grid(p{parent}, {disp0}, {e1}, {e2}, params[{columns}], {index})"
            ),
        }
    }

    /// The direction half of [`Self::instance_at`], by the *same* index.
    ///
    /// **Written beside it because they have to be the same rotation.** They were not: the
    /// point went in through `pattern_spin_radial`, which takes an absolute instance index,
    /// while the normal came back out through `pattern_normal_radial`, which works the index
    /// out from the point's own angle and treats what it is handed as an offset from that. The
    /// two therefore differed by the point's own cell, so the normal was turned by a whole
    /// number of steps too far — and by a *different* number either side of a cell boundary.
    ///
    /// That boundary is the middle of the fillet between two copies, so the shading creased
    /// exactly along the valleys while the geometry stayed smooth. Reported as "hard edges in
    /// the middle of the G1 fillet", confirmed by an STL of the same body being smooth.
    fn instance_normal_at(&self, normal: &str, index: &str) -> String {
        match self {
            // A row's, and a grid's, instances differ by a translation, which leaves directions
            // alone.
            PatternPoint::Linear { .. } | PatternPoint::Grid { .. } => normal.to_string(),
            PatternPoint::Radial { euler, step, bias, .. } => format!(
                "pattern_spin_normal_radial({normal}, {euler}, params[{step}], params[{bias}], \
                 {index})"
            ),
        }
    }
}

/// How many slots a pattern's prune takes: the sphere's centre and radius, and the divisor.
const PRUNE_SLOTS: usize = 5;

/// **Which copies of a pattern can matter in a cell of the skip grid** (plan §5 V6).
///
/// V6 is "a shortened tape per brick", and a generated shader cannot have one: every variant
/// is a pipeline, and a pipeline is a second of the driver's time. What it can have is the
/// same decision made at run time from something recorded per cell -- so the pass that fills
/// the skip grid also records, for each cell, which copies of a pattern could be the nearest
/// anywhere inside it, and the march skips the rest there.
///
/// The test, at a cell's centre `c` with half diagonal `h`, for copies whose value changes by
/// at most `L` a millimetre: copy `i` is out when `vᵢ(c) > min(c) + 2·L·h`. Anywhere in the
/// cell `vᵢ ≥ vᵢ(c) − L·h` and the minimum is at most `min(c) + L·h`, so such a copy is never
/// the minimum there, and a `min` that leaves it out returns **the same bits**. That is the
/// whole of what is claimed, and `framed_capture` holds the frames equal.
///
/// Narrow on purpose, to what that argument covers as it stands:
///
/// - a **hard** fold. A rounded one sums its copies and wants its own rule.
/// - a pattern that sees **the world's own point** (context nought). Under a placement, a
///   twist or another pattern, a cell of the world is some other region of the pattern's
///   space, and `h` is not its size.
/// - **one** pattern a program, its copies' bits from nought, and only the first thirty-two.
///   Nothing about the count is written into the source, so changing it is still a buffer
///   write: a copy past the thirty-second is simply never skipped.
fn masks_copies(rounded: bool, parent: usize) -> bool {
    !rounded && parent == 0
}

/// The bits a mask has.
const COPY_MASK: u32 = 32;

/// **Which operands of a join can matter in a cell of the skip grid** -- V6's general half.
///
/// The same idea as the copies above, one level up. A union, an intersection, a difference
/// and a blend each take operands, and over most of space one operand settles the answer: far
/// from the pipe a blend with the pipe *is* the other body, outside the cube a lattice
/// intersected with the cube *is* the cube. An operand that cannot matter anywhere in a cell
/// need not be evaluated there, and if it is a lattice or a pipe that is most of the reading.
///
/// **What stands in for it, and why the bits are the same.** A `min` ignores `+1e30` and a
/// `max` ignores `-1e30`, exactly. A blend reports `blend_bound(x, y)`, the max-norm distance
/// to its own zero set, and wherever `y >= |x| + k` the nearest of that set is the straight
/// piece `x = 0`: the value is `x`, to the bit, and the arc is not even looked at -- which is
/// what `1e30` gives too. So the operand is replaced by the sentinel its join ignores, and the
/// join computes what it would have computed.
///
/// **When.** At a cell's centre, with `h` its half diagonal and `L` each operand's rate:
/// an operand is out when its least possible value over the cell, `v - L*h`, is at or beyond
/// every other operand's greatest, `w + L*h` -- the mirror image for a `max` -- and for a
/// blend, where the other operand's sign does not matter but its size does, at or beyond
/// `|w| + L*h + k`. Recorded by the same fill, read by the same `field_here`.
///
/// Only at the world's own point, as for copies and for the same reason; only operands dear
/// enough to be worth a function call and a bit (`MASK_WEIGHT`); never in a part with critical
/// marks, whose blends taper through a different form that this argument has not been made
/// for; and thirty-two a program.
///
/// # What it took to make this pay, and how it was found
///
/// **As first built it lost on every document the owner had**, and the reason was not the one
/// two of us guessed at. The frame timing said masked was slower and could not say why, and the
/// two candidate causes wanted opposite fixes -- the masks never fire, so the condition here is
/// wrong; or they fire and the apparatus outweighs them, so the mechanism is. `mask_fire_rate`
/// reads the grid back and settles it: they fire. On WING-2 the offset was ruled out in 71.5%
/// of the cells the march must step inside, and the frame was 39% slower anyway.
///
/// So the defect was in what the condition *counts*. Two rules came out of it, and both are
/// tests of structure alone:
///
/// - **What the operand alone reads, not what it reads.** `closure_of` copies its subtree into
///   `masked_operand_N` rather than deferring it, so work the join's *other* operands also read
///   is emitted twice and evaluated twice; skipping one reader of it saves none of it. WING-2's
///   offset is an offset of the fuselage the blend reads anyway.
/// - **Only an operand dropped for reading *far*.** An intersection, and the kept side of a
///   difference, drop an operand for reading very negative -- deep inside its own solid, where
///   the march hardly goes. WING-2's `Blend 3416` is that operand and was ruled out in **0.0%**
///   of cells, on both grids, every run.
///
/// With both, measured over six rounds of twenty frames, least of the rounds, one device, arms
/// interleaved (ms a frame, 720 px and 1440 px):
///
/// | | masks | no masks |
/// |---|---|---|
/// | eight pipes on spokes, unioned | **17.0 / 56.7** | 20.8 / 64.5 |
/// | the same eight, chained by 3 mm blends | **24.3 / 57.2** | 34.2 / 65.1 |
/// | WING-2 | 13.4 / 54.0 | 13.8 / 55.7 |
/// | attractor-infill | 15.9 / 56.7 | 16.5 / 58.2 |
/// | complex-vase (nothing qualifies) | 36.6 / 142.2 | 36.2 / 142.0 |
/// | MIRROR-CUBE-BOOL | 6.8 / 32.3 | 6.7 / 30.7 |
///
/// A quarter to a third off the shape it is for, and inside the noise everywhere else -- where
/// the noise floor is about 5%, taken from the documents whose three arms compile to
/// byte-identical source. Cold, the extra compile is 6-22% (`compile_scaling`'s last series),
/// against the 3x that masking shared subtrees used to cost.
///
/// **Why the gate is not a question about where the operands are**, which is the premise the
/// feature rests on and would be the obvious thing to ask: `graph_extent` and `graph_lipschitz`
/// both resolve through `Document::parameter_value`, and `generate`'s contract one page up is
/// that two graphs differing only in their numbers produce byte-identical source. A gate on
/// extents would put a cold driver compile inside a number drag. Three more reasons it could
/// not be made to work from `Extent` in any case, each its own fact: `Extent::union` returns the
/// enclosing box, so "the operands are apart" is exactly what the type throws away; an extent
/// bounds a zero set and only grows outward, so it cannot see the interior case above; and the
/// cube the grid is filled over is the camera's scene, which codegen cannot see.
struct OperandMasks {
    /// Whether to mask anything at all -- see the verdict above.
    on: bool,
    used: u32,
    /// The node behind each bit, in bit order.
    of: Vec<NodeId>,
}

/// What an operand has to weigh for masking it to pay: `body_weight`'s scale, on which a
/// primitive is one and a curve walk eight. A lattice is counted as a walk here, which
/// `body_weight` does not need to do -- a lattice is never a true distance, so the box guard
/// never asks.
const MASK_WEIGHT: usize = 4;

/// **What masking this operand can actually save: the weight of the nodes only it reads.**
///
/// Weighing the whole subtree answers how dear the operand is, which is the wrong question on
/// its own -- `mask_fire_rate` found masks firing in 71.5% of WING-2's near cells and buying
/// nothing. The reason is mechanical: `closure_of` *copies* an operand's subtree into
/// `masked_operand_N` rather than deferring it, so a node the join's other operands also read
/// is emitted twice and evaluated twice, and skipping one reader of it saves none of its work.
/// WING-2's offset is an offset of the fuselage that the blend reads anyway: it weighs 9 and
/// saves 1.
///
/// So: `root` is ours only if this join is its one consumer, and a node below it only if every
/// node that reads it is ours too -- the exclusion has to carry downward, because a node read
/// by one that is emitted twice is emitted twice itself.
///
/// **Structure only, deliberately.** The obvious alternative is to ask where the operands
/// *are* -- which is the premise the feature rests on -- and that cannot be done here:
/// `graph_extent` and `graph_lipschitz` both resolve through `Document::parameter_value`, and
/// `generate`'s contract one page up is that two graphs differing only in their numbers produce
/// byte-identical source. A gate on extents would put a cold driver compile inside a number
/// drag. This one moves only when the shape does.
fn exclusive_weight(graph: &Graph, join: NodeId, root: NodeId) -> usize {
    let subtree = graph.subtree_of(root);
    let mut readers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (id, node) in graph.nodes() {
        for input in node.inputs() {
            readers.entry(input).or_default().push(id);
        }
    }
    let mut ours = subtree;
    loop {
        let shared: Vec<NodeId> = ours
            .iter()
            .copied()
            .filter(|id| {
                let outside = |other: &NodeId| {
                    if *id == root {
                        *other != join
                    } else {
                        !ours.contains(other)
                    }
                };
                readers.get(id).is_some_and(|rs| rs.iter().any(outside))
            })
            .collect();
        if shared.is_empty() {
            break;
        }
        for id in shared {
            ours.remove(&id);
        }
    }
    ours.iter().filter_map(|id| graph.node(*id)).map(|n| node_weight(n, true)).sum()
}

/// One operand of a join, as the join should read it: by its own name, or through a mask.
struct JoinOperand {
    /// The name the join reads.
    name: String,
    /// How fast its value changes, floored at one; and its mask's bit, if it has one.
    rate: f64,
    bit: Option<u32>,
}

impl OperandMasks {
    /// `name` as the join should read it. When the operand qualifies, its subtree becomes a
    /// function, and a block is pushed that calls it unless the cell has ruled it out.
    #[allow(clippy::too_many_arguments)]
    fn operand(
        &mut self,
        document: &fieldform_scene::Document,
        graph: &Graph,
        join: NodeId,
        id: NodeId,
        name: String,
        context: usize,
        sentinel: &str,
        points: &str,
        lines: &[String],
        subtrees: &mut Vec<String>,
        blocks: &mut Vec<String>,
    ) -> JoinOperand {
        let rate = fieldform_field::graph_lipschitz(document, graph, id).max(1.0);
        // **Only an operand the join drops for reading LARGE**, which the sentinel already
        // says: `1e30` is what a `min` ignores. The other direction -- every operand of an
        // intersection, and the kept side of a difference -- is dropped for reading very
        // *negative*, which is deep inside its own solid, and the march is hardly ever there.
        // Measured: WING-2's `Blend 3416` is the kept side of its top difference and is ruled
        // out in 0.0% of cells, on both grids, every run; the vase's `Boolean 9395` likewise.
        // Derived from the sentinel rather than matched again at the call sites, so the two
        // cannot fall out of step.
        let dropped_when_far = sentinel == "1e30";
        let eligible = self.on
            && dropped_when_far
            && context == 0
            && self.used < COPY_MASK
            && rate.is_finite()
            && !tapers(graph)
            && exclusive_weight(graph, join, id) >= MASK_WEIGHT;
        if !eligible {
            return JoinOperand { name, rate, bit: None };
        }
        let bit = self.used;
        self.used += 1;
        self.of.push(id);
        let inner = closure_of(points, lines, &name, "    let p0 = p;\n");
        subtrees.push(format!(
            "fn masked_operand_{bit}(p: vec3<f32>) -> f32 {{\n{inner}    return {name};\n}}\n\n"
        ));
        blocks.push(format!(
            "    var mo{bit} = {sentinel};\n\
             \x20   if (!operand_masked({bit}u)) {{ mo{bit} = masked_operand_{bit}(p0); }}\n"
        ));
        JoinOperand { name: format!("mo{bit}"), rate, bit: Some(bit) }
    }
}

/// The record a join leaves while the skip grid is being filled: for each masked operand,
/// whether this cell keeps it. `terms` are the operands as the join's `min` or `max` sees
/// them -- negated where the join negates -- with their rates and bits; `floor` is a blend's
/// radius, which an operand must also clear, or nothing.
///
/// Appended to the last block pushed, because a block is only emitted where something reads a
/// name it declares, and this declares none of its own.
fn record_operands(
    tag: usize,
    blocks: &mut [String],
    terms: &[(String, f64, Option<u32>)],
    least: bool,
    floor: Option<&str>,
) {
    if terms.iter().all(|(_, _, bit)| bit.is_none()) || terms.len() < 2 {
        return;
    }
    let Some(last) = blocks.last_mut() else { return };
    let mut record = format!(
        "    if (mask_mode == 1u) {{\n        let mh{tag} = 0.5 * mask_slack;\n"
    );
    for (i, (term, rate, bit)) in terms.iter().enumerate() {
        let Some(bit) = bit else { continue };
        // The others, at their worst for this operand: their greatest where the join takes the
        // least, their least where it takes the greatest.
        let others: Vec<String> = terms
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, (other, r, _))| {
                if least {
                    format!("(({other}) + {r:?} * mh{tag})")
                } else {
                    format!("(({other}) - {r:?} * mh{tag})")
                }
            })
            .collect();
        let mut against = others[0].clone();
        for other in &others[1..] {
            against = if least { format!("min({against}, {other})") } else { format!("max({against}, {other})") };
        }
        if let Some(k) = floor {
            // A blend, which is only ever a smooth *minimum* here: its forms negate into one.
            // `blend_bound` ignores an operand once it is further than the other by the radius
            // -- `y >= |x| + k` -- and not before: nearer than that the arc between them is the
            // nearest part of the zero set and both operands set the value. So the other
            // operand's *size* at its greatest over the cell, and the radius on top.
            let (other, r, _) = &terms[1 - i];
            against = format!("(abs({other}) + {r:?} * mh{tag} + {k})");
        }
        let out = if least {
            format!("(({term}) - {rate:?} * mh{tag}) >= {against}")
        } else {
            format!("(({term}) + {rate:?} * mh{tag}) <= {against}")
        };
        record.push_str(&format!(
            "        if (!({out})) {{ ops_kept = ops_kept | (1u << {bit}u); }}\n\
             \x20       ops_seen = ops_seen | (1u << {bit}u);\n"
        ));
    }
    record.push_str("    }\n");
    last.push_str(&record);
}

/// The sphere a pattern's sub-field lies in and the largest divisor it applies — or why the
/// pattern's loop is left as it was.
///
/// **Narrower than "any `Distance` or `LowerBound` sub-field", on purpose** (plan §5 V3). The
/// prune skips an instance whose floor `(|q − c| − ρ) / D` is no less than the running minimum,
/// which is only right if the sub-field never reads *below* that floor. `LowerBound` promises
/// the opposite side — a value no greater than the distance — so the floor has to be argued
/// kind by kind, and is admitted only where the argument holds:
///
/// - **A solid primitive** reads its true distance, which is at least the distance to any
///   sphere that holds it.
/// - **A pipe**, of a fixed or a driven radius: `(d_spine − r) / D` with `r` at most the ramp's
///   largest output, and the extent is already the spine grown by that output, so
///   `d_spine − r ≥ |q − c| − ρ`. `D` is `driven_pipe_divisor`, the figure the field divides by.
/// - **A union, a mirror, a hard pattern**: a minimum of floored values over a sphere holding
///   every operand. **A difference**: `max(a, −b) ≥ a`, and its extent is `a`'s.
/// - **A placement**: a similarity scales the value and the sphere alike.
/// - **A fixed offset over an undivided field**: `F − d` with the extent grown by `d`.
///
/// And refused, each for a reason that is a wrong picture rather than a slow one:
///
/// - **An intersection**, hard or filleted. Its extent is the *overlap* of its operands' boxes,
///   so the sphere can be far smaller than the region where `max(a, b)` is still small.
/// - **A baked leaf** — a cage or a mesh. Outside its bake box it answers with the distance to
///   that box, which is wider than its extent; the trap the per-body box guard met.
/// - **An offset over a divided field**: `G/D − d` floors at a sphere grown by `d·D`, not `d`.
/// - **A profile** (extrude, revolve, sweep): the display path reads a baked profile, short by
///   its margin. **A warp** (twist, attractor), **a morph**, **a lattice**, **a rounded fold or
///   blend**: not argued, so not admitted.
///
/// A sub-field that is refused keeps the loop it had, with the reason in the generated source.
fn pattern_prune_floor(
    document: &fieldform_scene::Document,
    graph: &Graph,
    input: &Operand,
) -> Result<([f64; 3], f64, f64), String> {
    /// The block in a slot, or the reason there is no floor to argue for one. A refusal here costs
    /// the pattern its prune and nothing else -- the loop keeps the form it had, with the reason
    /// in the generated source -- which is why a hole is answered rather than refused: both passes
    /// ask this while the plan is being priced, and a holed block is refused once, by `generate`.
    fn filled(slot: &Operand) -> Result<NodeId, String> {
        slot.source().ok_or_else(|| "a block in the sub-graph has an empty slot".to_string())
    }
    fn admit(
        document: &fieldform_scene::Document,
        graph: &Graph,
        id: NodeId,
        divisor: &mut f64,
        depth: usize,
    ) -> Result<(), String> {
        if depth > 256 {
            return Err("the sub-graph is too deep to walk".to_string());
        }
        let Some(node) = graph.node(id) else {
            return Err("the sub-graph names a node that is not there".to_string());
        };
        match node {
            Node::Sphere { .. }
            | Node::Cuboid { .. }
            | Node::SkewedBox { .. }
            | Node::Cylinder { .. }
            | Node::Cone { .. }
            | Node::Capsule { .. }
            | Node::Torus { .. } => Ok(()),
            Node::Pipe { .. } => {
                *divisor = divisor.max(fieldform_field::driven_pipe_divisor(document, graph, id));
                Ok(())
            }
            Node::Boolean { op: BooleanOp::Union, inputs } => {
                for i in inputs {
                    admit(document, graph, filled(i)?, divisor, depth + 1)?;
                }
                Ok(())
            }
            Node::Boolean { op: BooleanOp::Difference, inputs } => match inputs.first() {
                Some(first) => admit(document, graph, filled(first)?, divisor, depth + 1),
                None => Err("a difference of nothing".to_string()),
            },
            Node::Boolean { op: BooleanOp::Intersection, .. } => Err(
                "an intersection's extent is the overlap of its operands, which does not bound \
                 where its value is small"
                    .to_string(),
            ),
            Node::Transform { input, .. } | Node::Mirror { input, .. } => {
                admit(document, graph, filled(input)?, divisor, depth + 1)
            }
            Node::Pattern { input, fillet, .. } if fillet.is_plainly_hard() => {
                admit(document, graph, filled(input)?, divisor, depth + 1)
            }
            Node::Offset { input, distance } if distance.driver().is_none() => {
                let mut inner = 1.0;
                admit(document, graph, filled(input)?, &mut inner, depth + 1)?;
                if inner > 1.0 {
                    return Err("an offset over a divided field floors at a wider sphere".to_string());
                }
                Ok(())
            }
            Node::Cage(_) | Node::Mesh(_) => Err(
                "a baked leaf answers with its bake box outside it, which is wider than its extent"
                    .to_string(),
            ),
            other => {
                let kind = format!("{other:?}");
                let kind = kind.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("node");
                Err(format!("no floor is argued for a {kind} in the sub-graph"))
            }
        }
    }
    let mut divisor = 1.0f64;
    let input = filled(input)?;
    admit(document, graph, input, &mut divisor, 0)?;
    let (lo, hi) = fieldform_field::graph_extent(document, graph, input)
        .as_box()
        .ok_or_else(|| "the sub-graph has no finite bounds".to_string())?;
    let centre = (lo.coords + hi.coords) * 0.5;
    // Padded, as the body guard's box is and for the same reason: the floor and the value are
    // different arithmetic, and a thousandth keeps a rounding error from skipping an instance
    // the fold would have taken.
    let radius = (hi - lo).norm() * 0.5 * (1.0 + 1e-3) + 1e-6;
    if !(radius.is_finite() && divisor.is_finite() && centre.iter().all(|v| v.is_finite())) {
        return Err("the sub-graph's bounds are not finite".to_string());
    }
    Ok(([centre.x, centre.y, centre.z], radius, divisor.max(1.0)))
}

/// A pattern's window, as the ingredients of a loop rather than a list of unrolled steps.
struct PatternLoop {
    /// The point the pattern is sampled at, which every step is measured from.
    parent: usize,
    /// Whether the window is as wide as the pattern, so every instance is walked once instead.
    ///
    /// Decided here rather than in the shader because both numbers it depends on -- the reach
    /// and the count -- are fixed for a given shader: editing either is a graph edit, which
    /// rebuilds. The radius stays a slot; this does not need to be one.
    whole: bool,
    /// The parameter slot holding the fillet's radius, when the copies are rounded into each
    /// other rather than merely unioned.
    ///
    /// **`None` generates exactly the code a hard fold always generated.** Rounding costs a
    /// `pow` per instance per step, and all but a few patterns are hard — so whether there is a
    /// fillet at all is decided here, while the radius itself stays a slot so that dragging it
    /// writes a buffer. The same trade `Node::Twist` makes, and for the same reason: going from
    /// hard to rounded is a structural change and rare enough to afford a rebuild.
    rounding: Option<usize>,
    /// The exponent the rounding raises its terms to, written as a literal the way a blend's is.
    exponent: f64,
    /// The parameter slot holding the lowest instance index the block holds: zero normally,
    /// one when the block leaves the body where it already is out of what it holds.
    ///
    /// **A slot rather than a branch in the generator**, so toggling it writes a buffer instead
    /// of rebuilding the shader — the same reason `reach` and the frame are slots.
    first: usize,
    /// The slot holding how many copies the block leaves out, which the rounded fold's divisor
    /// subtracts from the count. For a pattern that leaves out only its first copy this is the
    /// same number `first` is, which is why the divisor could once read `first` for it.
    left_out: usize,
    /// The first of the words the skip list is stored in, when more than the first copy is left
    /// out. `None` generates the loop a pattern without a skip list always had.
    skip: Option<usize>,
    /// The parameter slot holding how many cells either side the loop walks.
    ///
    /// **A parameter and not a literal**, for two reasons. A literal bound is one a driver can
    /// unroll, which would undo the loop and put the code back where it was. And a reach that
    /// lives in the buffer means changing a pattern's count changes no shader text at all --
    /// it becomes a buffer write, like every other number, so patterning a hundred copies and
    /// then a thousand costs nothing to recompile.
    reach: usize,
    point: PatternPoint,
    /// The first of the prune's five slots, or why this pattern's loop is left unpruned.
    prune: Result<usize, String>,
    /// The slot holding how fast one copy's value can change, when this pattern's copies are
    /// masked cell by cell. See [`COPY_MASK`].
    mask: Option<usize>,
}

/// The output of [`point_assignment`].
struct PointPlan {
    /// One entry per declared point, in declaration order: the context it is derived from, the
    /// node that derives it, and which step of that node's window it is. Entry `i` declares
    /// `p{i + 1}`.
    contexts: Vec<(usize, NodeId, isize)>,
    /// For each occurrence of a point-changing node, the contexts it hands to its inputs.
    ///
    /// A list rather than one, because a pattern hands down a whole window and its value is the
    /// nearest of them — see [`window_of`].
    below: HashMap<(NodeId, usize), Vec<usize>>,
    /// Every `(node, context)` to emit, operands first.
    emit: Vec<(NodeId, usize)>,
}

impl PointPlan {
    /// The context this node's *inputs* are evaluated in, given the context it is itself in.
    ///
    /// A pattern hands down one per copy; every other node hands down exactly one, which is
    /// the one this returns.
    fn inputs_of(&self, id: NodeId, ctx: usize) -> usize {
        self.below
            .get(&(id, ctx))
            .and_then(|all| all.first().copied())
            .unwrap_or(ctx)
    }
}

/// How far a twist can stretch material, from the body's own reach from the axis.
///
/// The same factor the CPU computes, and it has to be: two paths dividing by different numbers
/// would put the surface in the same place while scaling the field around it, so a ray stops
/// short on one and not the other. Nothing but a residual against the CPU would catch that.
fn twist_stretch(
    document: &fieldform_scene::Document,
    graph: &Graph,
    id: NodeId,
) -> f64 {
    let Some(Node::Twist { input, degrees_per_mm, frame }) = graph.node(id) else {
        return 1.0;
    };
    let rate = document.resolve(degrees_per_mm).unwrap_or(0.0).to_radians();
    let Some((lo, hi)) = fieldform_field::operand_extent(document, graph, input).as_box() else {
        // Compilation refuses an unbounded twist before a shader is generated, so this is
        // unreachable rather than a guess — and 1.0 would be the *unsafe* direction, since
        // dividing by too little overstates the distance.
        return 1.0;
    };
    // The reach across the axis, which is the direction the winding carries material sideways
    // in — and it has to be the *same* measurement the CPU makes, or the two divide by
    // different numbers and put the surface in one place while scaling the field around it
    // differently. Both now take it from the frame's origin, over all three axes, which is
    // what an axis that can point anywhere costs.
    let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
    let Ok((origin, rotation)) = fieldform_field::resolve_frame(document, graph, frame, &numbers) else {
        return 1.0;
    };
    // The kernel's own measurement, called rather than repeated: both divide by what it
    // returns, so a difference would put the surface in one place and scale the field around it
    // differently — drift only a residual against the other would catch.
    let r = fieldform_field::reach_across_axis((lo, hi), origin, rotation);
    (1.0 + (r * rate).powi(2)).sqrt().max(1.0)
}



/// A frame, as the placement pair `to_local` takes: an origin and Euler degrees.
///
/// Literals rather than live slots, because the Euler angles are *derived* from the frame's
/// axis — there is no parameter that is the pitch. A named parameter driving an axis therefore
/// regenerates the shader rather than only rewriting the buffer, which is the honest cost of
/// letting an axis point anywhere.
fn frame_placement(
    params: &mut Vec<ParamRef>,
    graph: &Graph,
    document: &fieldform_scene::Document,
    frame: &fieldform_scene::Frame,
) -> (String, String) {
    let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
    let (origin, rotation) = fieldform_field::resolve_frame(document, graph, frame, &numbers)
        .unwrap_or_else(|_| (nalgebra::Vector3::zeros(), nalgebra::Rotation3::identity()));
    let (rx, ry, rz) = rotation.euler_angles();
    let at = push_point(
        params,
        &[
            ParamRef::Literal(origin.x),
            ParamRef::Literal(origin.y),
            ParamRef::Literal(origin.z),
        ],
    );
    let euler = push_point(
        params,
        &[
            ParamRef::Literal(rx.to_degrees()),
            ParamRef::Literal(ry.to_degrees()),
            ParamRef::Literal(rz.to_degrees()),
        ],
    );
    (at, euler)
}

/// A frame, as the two shader expressions a mirror needs: a point and a unit normal.
///
/// **Asked of the kernel, not worked out again here.** Everything but a typed frame goes through
/// `resolve_frame` — the same call `frame_placement` above makes for a revolve and a radial
/// pattern — because a frame is a shared type precisely so that one resolver answers for every
/// block that takes one (D9). This used to answer for itself, and drifted three ways while the
/// kernel moved on without it:
///
/// - a plane's **own turn** was dropped. The kernel makes it part of the facing; this returned a
///   bare `±Z`. And because a turned block *is* that block inside a `Node::Transform` by the time
///   either compiler sees it (`with_turns_flattened`, called on line 650), a plane turned at all
///   did not even reach the plane arm —
/// - it fell through to a `_` arm asserting that "the kernel refuses a frame that is not a
///   plane", which stopped being true when `resolve_frame` learned to read a placement and a
///   point. So a mirror aimed at a turned plane, or at a point, mirrored about the **world XY
///   plane through the origin** — and the panel, the picking and the field all said otherwise —
/// - and the curve-borne arms resolved named parameters with `ParamValue::as_literal`, which
///   skips every expression and answers `0.0` for a driven one.
///
/// Six slots in every arm, which is also what makes the two passes agree by counting rather than
/// by reasoning. A typed frame keeps its own `ParamRef`s, so a named parameter driving an origin
/// or an axis still drives it; every other kind is *derived* — a plane's facing is not a number
/// anybody typed here — and goes in as a resolved literal. Both are buffer entries either way, so
/// re-aiming a mirror rewrites six numbers and never the shader.
fn frame_params(
    params: &mut Vec<ParamRef>,
    document: &fieldform_scene::Document,
    graph: &Graph,
    frame: &fieldform_scene::Frame,
) -> (String, String) {
    if let fieldform_scene::Frame::At { origin, axis } = frame {
        let at = push_point(params, origin);
        let aim = push_point(params, axis);
        // Normalised in the shader rather than here: the axis is live, so its length is not
        // known on this side. `aimed` is the kernel's `rotation_from_z` rule — a zero axis is
        // `+Z`, not a division by zero — which a bare `normalize` is not.
        return (at, format!("aimed({aim})"));
    }
    let (origin, normal) = resolved_frame(document, graph, frame);
    let at = push_point(params, &triple(origin));
    // Already unit, since it came out of a rotation.
    (at, push_point(params, &triple(normal)))
}

/// The same frame in the normal pass's vocabulary: slots counted, nothing pushed.
///
/// Six either way, matching [`frame_params`] arm for arm — the two passes walk the same nodes in
/// the same order and a slot pushed in one and not the other shifts every parameter after it,
/// silently, since the shader still compiles and simply reads the wrong ones.
fn frame_slots(
    params: &mut usize,
    document: &fieldform_scene::Document,
    graph: &Graph,
    frame: &fieldform_scene::Frame,
) -> (String, String) {
    if let fieldform_scene::Frame::At { origin, axis } = frame {
        let at = point_slots(params, origin);
        let aim = point_slots(params, axis);
        return (at, format!("aimed({aim})"));
    }
    let unused = [ParamRef::Literal(0.0); 3];
    let at = point_slots(params, &unused);
    let aim = point_slots(params, &unused);
    // Read rather than ignored, so this arm cannot quietly stop matching the value pass's:
    // if resolving ever starts refusing a frame, both passes refuse it in the same breath.
    let _ = resolved_frame(document, graph, frame);
    (at, aim)
}

/// Where a frame sits and which way it points, through the kernel's own resolver.
///
/// A frame the kernel cannot read falls back to the world origin facing `+Z`, which is what
/// `frame_placement` does with the same refusal and for the same reason: compilation refuses such
/// a graph before a shader is generated, so this is the shape of an unreachable case rather than
/// a guess. It must stay one function, because the two passes call it and a difference between
/// them would move the surface in one and not the other.
fn resolved_frame(
    document: &fieldform_scene::Document,
    graph: &Graph,
    frame: &fieldform_scene::Frame,
) -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) {
    // Through the document, never off the parameter: an expression-driven angle resolves here
    // and reads as nothing at all if this asks `ParamValue::as_literal` instead.
    let numbers = |r: &ParamRef| -> Option<f64> { document.resolve(r) };
    let (origin, rotation) = fieldform_field::resolve_frame(document, graph, frame, &numbers)
        .unwrap_or_else(|_| (nalgebra::Vector3::zeros(), nalgebra::Rotation3::identity()));
    (origin, rotation * nalgebra::Vector3::z())
}

/// Three literals, as the parameter pushers take them.
fn triple(v: nalgebra::Vector3<f64>) -> [ParamRef; 3] {
    [
        ParamRef::Literal(v.x),
        ParamRef::Literal(v.y),
        ParamRef::Literal(v.z),
    ]
}


/// Turns the flat list of emitted lines into one function per body, and the calls that fold them.
///
/// **Because a driver optimiser is superlinear in what is live at once.** Every node used to be a
/// `let` inside a single function, which is correct and compiles quickly at one body and does not
/// at three. A function boundary is a register allocation boundary, and several small problems
/// are not one large one.
///
/// Used by both passes, because both had the same shape and the normal pass is the heavier of the
/// two — it emits a value *and* a gradient per node, so leaving it whole would have kept the
/// stall while the distance pass looked fixed.
///
/// `roots` are the expressions each body returns, not names: a distance body returns `n42` and a
/// normal body returns `vec4<f32>(g42, v42)`, and seeding the walk from the identifiers *in* the
/// expression covers both without a special case.
///
/// **What it costs is duplication, and that was measured and kept** (loop spec P7, 2026-10-13). A
/// node feeding two bodies is emitted into both, because there is nowhere else to put it: WGSL
/// has no closures, and threading a dozen values through a signature would reintroduce the live
/// range the split exists to break. Splitting on the shared subgraph instead -- a function per
/// subgraph two bodies share, which each of them calls -- was built in `a59a201` and measured
/// against itself written out. On blocks cut from one patterned ring it saved 0.4 % of the
/// characters at two bodies and 6.5 % at sixteen, and nothing in `generate` or at the driver:
/// 1 102 -> 1 110 ms cold at two, 50 820 -> 50 700 ms at sixteen, medians of five. The driver
/// compiles what it inlines, and it inlines a call at every site (`union-compile-8fa06b7.md`
/// counted the lines), so the copies the text saved were copies it compiled anyway. What does
/// grow is the number of bodies: those blocks compile in 1.1, 1.9, 10 and 51 s at 2, 3, 8 and 16,
/// shared or not.
fn split_into_bodies(
    points: &str,
    lines: &[String],
    roots: &[String],
    called: &str,
    returns: &str,
) -> (String, Vec<String>) {
    let mut functions = String::new();
    let mut calls = Vec::new();
    for (i, root) in roots.iter().enumerate() {
        let inner = closure_of(points, lines, root, "");
        functions.push_str(&format!(
            "fn {called}_{i}(p: vec3<f32>) -> {returns} {{\n{inner}    return {root};\n}}\n\n"
        ));
        calls.push(format!("{called}_{i}(p)"));
    }
    (functions, calls)
}

/// The lines `root` needs, in declaration order, with `prologue` first if it declares something.
///
/// Dependencies are read back out of the emitted text rather than re-derived from the graph. That
/// sounds fragile and is the opposite: the graph would have to be walked *through point contexts*
/// to get the same answer, which is a second implementation of the thing above that could
/// disagree with it. The text cannot disagree with itself, and the names it uses — `n12`, `g7`,
/// `p3` — are ours and unambiguous, matched as whole identifiers so `params` is never a `p`.
///
/// Keyed on every name a line *declares*, not one per line, because a line here can be a whole
/// block: a pattern emits a loop that declares its accumulator and its result together, and
/// reaching either has to bring the whole thing.
fn closure_of(points: &str, lines: &[String], root: &str, prologue: &str) -> String {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    let mut supplied: HashSet<&str> = HashSet::new();
    for name in declares(prologue) {
        supplied.insert(name);
    }
    let point_lines: Vec<&str> = points.lines().collect();
    for (i, line) in point_lines.iter().enumerate() {
        for name in declares(line) {
            by_name.insert(name, i);
        }
    }
    // Offset so one map can address both lists.
    for (i, line) in lines.iter().enumerate() {
        for name in declares(line) {
            by_name.insert(name, point_lines.len() + i);
        }
    }

    let mut wanted: HashSet<usize> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut queue: Vec<String> = identifiers(root);
    queue.extend(identifiers(prologue));
    while let Some(name) = queue.pop() {
        if !seen.insert(name.clone()) || supplied.contains(name.as_str()) {
            continue;
        }
        let Some(&index) = by_name.get(name.as_str()) else { continue };
        if !wanted.insert(index) {
            continue;
        }
        let text = if index < point_lines.len() {
            point_lines[index]
        } else {
            lines[index - point_lines.len()].as_str()
        };
        queue.extend(identifiers(text));
    }

    let mut inner = String::from(prologue);
    for (i, line) in point_lines.iter().enumerate() {
        if wanted.contains(&i) {
            inner.push_str(line);
            inner.push('\n');
        }
    }
    for (i, line) in lines.iter().enumerate() {
        if wanted.contains(&(point_lines.len() + i)) {
            inner.push_str(line);
        }
    }
    inner
}

/// Every name a chunk of generated code declares, over all its lines.
fn declares(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let rest = trimmed
                .strip_prefix("let ")
                .or_else(|| trimmed.strip_prefix("var "))?;
            rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next()
                .filter(|name| !name.is_empty())
        })
        .collect()
}

impl PatternLoop {
    /// The same as `join_divisor`, for a fold whose radius tapers toward a critical surface.
    fn taper_divisor(&self, radius: usize, rho: f64) -> String {
        format!(
            "pattern_taper_divisor(params[{radius}], params[{}], params[{}], params[{}], {:?}, \
             {rho:?})",
            self.reach,
            self.point.count(),
            self.left_out,
            self.exponent,
        )
    }

    /// The call that divides a rounded fold's value, given the slot its radius is in.
    ///
    /// One spelling for the value pass and the normal pass, so the two cannot come to different
    /// conclusions about how far the march may step.
    fn join_divisor(&self, radius: usize) -> String {
        format!(
            "pattern_join_divisor(params[{radius}], params[{}], params[{}], params[{}], {:?})",
            self.reach,
            self.point.count(),
            self.left_out,
            self.exponent,
        )
    }

    /// What the loop is, for the comment the generator leaves beside it.
    fn shape(&self) -> &'static str {
        match (&self.point, self.whole) {
            (PatternPoint::Grid { .. }, true) => "grid, whole",
            (PatternPoint::Grid { .. }, false) => "grid, windowed",
            (PatternPoint::Radial { .. }, true) => "radial, whole",
            (PatternPoint::Radial { .. }, false) => "radial, windowed",
            (PatternPoint::Linear { .. }, true) => "linear, whole",
            (PatternPoint::Linear { .. }, false) => "linear, windowed",
        }
    }

    /// The call that produces the point for step `step`, which is a loop variable and not a slot.
    fn point_at(&self, step: &str) -> String {
        let parent = self.parent;
        match &self.point {
            PatternPoint::Linear { unit, pitch, count, origin, bias } => format!(
                "pattern_point_linear(p{parent}, {unit}, params[{pitch}], params[{count}], \
                 params[{origin}], params[{bias}], {step})"
            ),
            PatternPoint::Radial { at, euler, count, step: angle, origin, bias, arc, .. } => format!(
                "pattern_point_radial(p{parent}, {at}, {euler}, params[{count}], \
                 params[{angle}], params[{origin}], params[{bias}], params[{arc}], {step})"
            ),
            PatternPoint::Grid { .. } => {
                unreachable!("a grid's window is two loops, written by `walk` and not one step")
            }
        }
    }

    /// The loop that walks the window, accumulating with `accumulate` each time round.
    ///
    /// `accumulate` is given the name of the step's point and is expected to be statements.
    fn walk(
        &self,
        tag: &str,
        // The value above which an instance cannot change the fold, in the value pass: the
        // running minimum for a hard fold, and the larger of that and the radius for a rounded
        // one, where a copy contributes whenever it is within the radius. `None` in the normal
        // pass, which walks the whole loop.
        threshold: Option<&str>,
        accumulate: impl Fn(&str) -> String,
    ) -> String {
        let reach = self.reach;
        let first = self.first;
        let count = self.point.count();
        let step = format!("s{tag}");
        let q = format!("q{tag}");
        // What one instance costs the fold: the accumulator, behind the floor test when this
        // pattern is pruned. Under `march-stats` each call of the sub-field is counted, pruned
        // or not, so the saving is a count rather than a timing.
        let counted = if cfg!(feature = "march-stats") && threshold.is_some() {
            "        stat_sub_calls = stat_sub_calls + 1u;\n"
        } else {
            ""
        };
        // **A whole row is left unpruned, on measurement.** Its point is `p - unit * (pitch * k)`,
        // and once the floor reads that point as well as the sub-field, the shader compiler rounds
        // it differently: on four oblique bars 18 579 of 262 144 lattice points moved, as many down
        // as up and by at most 32 in the last place, with the walk's order making no difference and
        // the floor alone making all of it. A skipped instance could only ever raise a minimum, so
        // that is rounding and not a wrong prune -- but the contract is equality to the bit, and
        // here it cannot be held. Little is given up: a row is walked whole only when its window
        // already covers it, which is a short row. A ring's point goes through a rotation and a
        // windowed row's through `pattern_point_linear`, and both hold to the bit.
        let refused: Result<usize, String>;
        let prune = match (&self.prune, &self.point, self.whole) {
            (Ok(_), PatternPoint::Linear { .. }, true) => {
                refused = Err("the compiler rounds a whole row's point differently once the \
                               floor reads it too, so equality to the bit cannot be held"
                    .to_string());
                &refused
            }
            (kept, _, _) => kept,
        };
        let each = match (threshold, prune) {
            (Some(above), Ok(slot)) => format!(
                "        if (pattern_floor({q}, {slot}u) < {above}) {{\n{counted}{}        }}\n",
                accumulate(&q)
            ),
            _ => format!("{counted}{}", accumulate(&q)),
        };
        // A copy the cell's mask rules out is passed over before its point is even made. Only
        // in the value pass, and only for the pattern that has a mask. `index` is the copy's
        // own number, as an `i32` expression, in whichever loop is being written.
        let masked = |index: &str| -> String {
            match (threshold, self.mask) {
                (Some(_), Some(_)) => format!(
                    "        if (copy_masked({index})) {{ continue; }}\n"
                ),
                _ => String::new(),
            }
        };
        let said = match (threshold, prune) {
            (None, _) => String::new(),
            (Some(_), Ok(_)) => format!(
                "    // pattern {tag}: {} loop, pruned by the sub-field's sphere\n",
                self.shape()
            ),
            (Some(_), Err(why)) => format!(
                "    // pattern {tag}: {} loop, unpruned: {why}\n",
                self.shape()
            ),
        };
        // A copy the pattern leaves out is passed over before its point is even made, in both
        // passes: the value and the normal have to agree about which copies exist.
        let skipped = |index: &str| -> String {
            match self.skip {
                Some(slot) => {
                    format!("        if (pattern_skipped({slot}u, {index})) {{ continue; }}\n")
                }
                None => String::new(),
            }
        };
        // **A grid has two windows, and each direction is walked whole or by window on its own.**
        // Rows outside, columns inside, so the copies come in the order their numbers run. A
        // direction its window covers is walked from nought to its count; one it does not is walked
        // either side of the copy nearest the point, with whatever falls off the grid dropped.
        if let PatternPoint::Grid {
            anchor, disp0, e1, e2, d1, d2, columns, rows, reach: [r1, r2], whole: [whole_columns, whole_rows], ..
        } = &self.point
        {
            let parent = self.parent;
            let first = self.first;
            let centre = format!("gc{tag}");
            let need_centre = !(*whole_columns && *whole_rows);
            let header = |var: &str, r: usize, n: usize, whole: bool, axis: &str| -> (String, String, String) {
                if whole {
                    (
                        format!("for (var {var} = 0; {var} < i32(params[{n}]); {var} = {var} + 1) {{"),
                        format!("f32({var})"),
                        String::new(),
                    )
                } else {
                    (
                        format!(
                            "for (var {var} = -i32(params[{r}]); {var} <= i32(params[{r}]); \
                             {var} = {var} + 1) {{"
                        ),
                        format!("(round({centre}.{axis}) + f32({var}))"),
                        // Off the grid: no such copy.
                        format!("if (#v# < 0.0 || #v# >= params[{n}]) {{ continue; }}\n"),
                    )
                }
            };
            let (row_for, row_at, row_check) = header(&format!("ib{tag}"), *r2, *rows, *whole_rows, "y");
            let (col_for, col_at, col_check) = header(&format!("ia{tag}"), *r1, *columns, *whole_columns, "x");
            let (fa, fb, gi) = (format!("fa{tag}"), format!("fb{tag}"), format!("gi{tag}"));
            let mut text = said;
            if need_centre {
                text.push_str(&format!(
                    "    let {centre} = pattern_grid_centre(p{parent}, {anchor}, {e1}, {e2}, {d1}, \
                     {d2}, params[{columns}], params[{rows}]);\n"
                ));
            }
            text.push_str(&format!("    {row_for}\n        let {fb} = {row_at};\n"));
            if !row_check.is_empty() {
                text.push_str(&format!("        {}", row_check.replace("#v#", &fb)));
            }
            text.push_str(&format!("        {col_for}\n            let {fa} = {col_at};\n"));
            if !col_check.is_empty() {
                text.push_str(&format!("            {}", col_check.replace("#v#", &fa)));
            }
            text.push_str(&format!(
                "            let {gi} = i32({fa}) + i32(params[{columns}]) * i32({fb});\n\
                 \x20           if ({gi} < i32(params[{first}])) {{ continue; }}\n\
                 {skip}{skips}\
                 \x20           let {q} = p{parent} - ({disp0} + {e1} * {fa} + {e2} * {fb});\n{each}\
                 \x20       }}\n    }}\n",
                skip = masked(&gi),
                skips = skipped(&gi),
            ));
            return text;
        }
        // **A window wide enough to cover everything walks the instances instead.** Folding is
        // an optimisation, and once the window is as wide as the ring it stops being one: the
        // offsets wrap onto the same instance more than once, which a `min` shrugs off and a
        // sum does not. Walking the indices is then both exact and cheaper.
        if self.whole {
            // **Nearest first, when the fold is a plain minimum and the pattern is pruned.**
            // The running minimum drops on the first call and the floors of the rest are
            // measured against it; in index order the prune would only bite by luck. A minimum
            // does not care what order it is taken in, so the value is the same to the bit. A
            // rounded fold is a *sum*, which does care, so it keeps index order and is pruned
            // where it stands.
            if threshold.is_some() && prune.is_ok() && self.rounding.is_none() {
                let (nearest, wrap, turns) = match &self.point {
                    // A grid returned above, with loops of its own.
                    PatternPoint::Grid { .. } => unreachable!("a grid's loops are written first"),
                    // An arc has two ends and is walked like a row; a full turn wraps.
                    PatternPoint::Radial { at, euler, count, step: angle, origin, arc, is_arc: true, .. } => (
                        format!(
                            "pattern_nearest_radial(p{}, {at}, {euler}, params[{count}], \
                             params[{angle}], params[{origin}], params[{arc}])",
                            self.parent
                        ),
                        format!("if (pk{tag} < 0 || pk{tag} >= pc{tag}) {{ continue; }}"),
                        format!("2 * pc{tag}"),
                    ),
                    PatternPoint::Radial { at, euler, count, step: angle, origin, arc, is_arc: false, .. } => (
                        format!(
                            "pattern_nearest_radial(p{}, {at}, {euler}, params[{count}], \
                             params[{angle}], params[{origin}], params[{arc}])",
                            self.parent
                        ),
                        // Every residue once: 0, +1, −1, … out to half the ring. **The count is added
                        // before the remainder is taken, never after**: the offset is at most half the ring,
                        // so the sum cannot be negative, and the remainder of a negative `i32` is not the
                        // truncated one on the GPU. Written as `((k % n) + n) % n` this loop visited four of
                        // six copies, and the bitwise check against the unpruned field is what said so.
                        format!("pk{tag} = (pk{tag} + pc{tag}) % pc{tag};"),
                        format!("pc{tag}"),
                    ),
                    PatternPoint::Linear { unit, pitch, count, origin, .. } => (
                        format!(
                            "pattern_nearest_linear(p{}, {unit}, params[{pitch}], params[{count}], params[{origin}])",
                            self.parent
                        ),
                        // Off either end of the row is no instance; the far end may be a whole
                        // row away from the nearest, hence twice the count of turns.
                        format!("if (pk{tag} < 0 || pk{tag} >= pc{tag}) {{ continue; }}"),
                        format!("2 * pc{tag}"),
                    ),
                };
                let point = self.point.instance_at(self.parent, &format!("f32(pk{tag})"));
                return format!(
                    "{said}    let pc{tag} = i32(params[{count}]);\n\
                     \x20   let pz{tag} = i32({nearest});\n\
                     \x20   for (var {step} = 0; {step} < {turns}; {step} = {step} + 1) {{\n\
                     \x20       let ph{tag} = ({step} + 1) / 2;\n\
                     \x20       var pk{tag} = pz{tag} + ph{tag};\n\
                     \x20       if (({step} & 1) == 0) {{ pk{tag} = pz{tag} - ph{tag}; }}\n\
                     \x20       {wrap}\n\
                     \x20       if (pk{tag} < i32(params[{first}])) {{ continue; }}\n\
                     {skip}{skips}\
                     \x20       let {q} = {point};\n{each}\
                     \x20   }}\n",
                    skip = masked(&format!("pk{tag}")),
                    skips = skipped(&format!("pk{tag}")),
                );
            }
            let point = self.point.instance_at(self.parent, &format!("f32({step})"));
            return format!(
                "{said}    for (var {step} = i32(params[{first}]); \
                 {step} < i32(params[{count}]); {step} = {step} + 1) {{\n\
                 {skip}{skips}\
                 \x20       let {q} = {point};\n{each}\
                 \x20   }}\n",
                skip = masked(&step),
                skips = skipped(&step),
            );
        }
        let point = self.point_at(&format!("f32({step})"));
        // The step point comes back with the instance it belongs to beside it. An instance
        // below `first` is not one this block holds -- see `Node::Pattern::join` -- and one at
        // or past the count is off the end of a row. Dropped here rather than in each caller's
        // accumulator, so the value pass and the normal pass cannot come to different
        // conclusions about which copies exist, and so that no instance is visited twice.
        format!(
            "{said}    let r{tag} = i32(params[{reach}]);\n\
             \x20   for (var {step} = -r{tag}; {step} <= r{tag}; {step} = {step} + 1) {{\n\
             \x20       let c{tag} = {point};\n\
             \x20       if (c{tag}.w < params[{first}] || c{tag}.w >= params[{count}]) \
             {{ continue; }}\n\
             {skip}{skips}\
             \x20       let {q} = c{tag}.xyz;\n{each}\
             \x20   }}\n",
            skip = masked(&format!("i32(c{tag}.w)")),
            skips = skipped(&format!("i32(c{tag}.w)")),
        )
    }
}

/// The identifier-shaped tokens in a line, so `params[3]` yields `params` and never `p`.
fn identifiers(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in line.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            current.push(c);
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Unions the bodies into one value, as a statement per body rather than one nested
/// expression, or `None` for a part with no bodies.
///
/// `min(min(min(...)))` is the same arithmetic and does not survive being written down: a STEP
/// assembly is one body per solid, and a shader whose expression nests once per body is a
/// parse tree as deep as the part is wide. Sequential `let`s are flat, and the shader compiler
/// sees a straight line no matter how many bodies there are.
///
/// `calls` are the body functions' call expressions; `boxes[i]` is the first of body `i`'s six
/// guard slots, or `None` for a body evaluated at every reading. `cull` is where the per-tile
/// extents start, or `None` where this program does not cull.
fn fold_union(
    body: &mut String,
    calls: &[String],
    boxes: &[Option<usize>],
    cull: Option<usize>,
) -> Option<String> {
    let first = calls.first()?.clone();
    if calls.len() == 1 {
        body.push_str("    owner = 0u;\n");
        return Some(first);
    }
    // A running minimum with the body that produced it, rather than nested `min` calls. The
    // fold stays flat and linear — which is what a 10 000-body STEP assembly needed — and it
    // now also answers *which* body is nearest, which is what lets one be highlighted without
    // a second trace or a second shader.
    //
    // Started from "nothing yet" rather than from the first body, so the first body is guarded
    // like the rest: a guard against an empty minimum always passes, so the first box that is
    // asked is always evaluated and the fold can never come back with nothing.
    // Every guard's reads first, then the chain. Each guard is six buffer reads, and issued
    // inside its own branch they would wait one behind another for the whole chain; issued
    // together up front their latencies overlap, as the bodies' own reads always did.
    for (i, slot) in boxes.iter().enumerate() {
        if let Some(slot) = slot {
            body.push_str(&format!("    let g{i} = box_guard(p, {slot}u);\n"));
        }
    }
    body.push_str("    var best = 3.4e38;\n    owner = 0u;\n");
    for (i, call) in calls.iter().enumerate() {
        let take = format!("let b{i} = {call}; if (b{i} < best) {{ best = b{i}; owner = {i}u; }}");
        // A body outside whose box the running minimum already lies cannot be nearer
        // than it -- see `body_box_slots` for why that holds, and for whom.
        let guarded = boxes
            .get(i)
            .copied()
            .flatten()
            .map(|_| format!("if (g{i} < best) {{ {take} }}"))
            .unwrap_or_else(|| take.clone());
        // **A ruled-out body is dropped, and the march is what makes that safe.** The fold then
        // answers the distance to the *kept* bodies, which over-estimates the distance to the
        // scene -- a body outside this tile cannot be hit by its rays, but it can still be the
        // nearest surface to a point on one. A plain step of that value can never cross a kept
        // surface, so `trace_primary` stops over-relaxing while a mask is live; see the `relax`
        // line in `march.wgsl`, and §1.7a for the artefact that was.
        //
        // **Replacing the body with its extent was tried first and was worse than useless**: it
        // restores the bound, but the box guard already rejects such a body with one box test, so
        // the fold ends up doing the same test twice and gains nothing. Measured over seven
        // scenes, every one of them slower with the mask than without -- 15.7 to 16.6 ms on fifty
        // spread bodies, 13.4 to 38.7 orthographic. Correct and pointless is still pointless.
        match cull {
            Some(_) => body.push_str(&format!(
                "    if (!tile_ruled_out({i}u)) {{ {guarded} }}\n"
            )),
            None => body.push_str(&format!("    {guarded}\n")),
        }
    }
    Some("best".to_string())
}

/// Combines a boolean's operands into one value, as a statement per operand rather than one
/// nested expression, and hands back the name that holds the result — or the lone operand
/// itself, which needs no statement.
///
/// The same reason `fold_union` above is flat: `min(min(min(...)))` is a parse tree as deep as
/// the list is long. A user's boolean is rarely thousands wide, but the shape that failed once
/// is not worth writing a second time.
fn fold_boolean(body: &mut String, op: BooleanOp, acc: &str, operands: &[String]) -> Option<String> {
    let first = operands.first()?;
    if operands.len() == 1 {
        return Some(first.clone());
    }
    body.push_str(&format!("    var {acc} = {first};\n"));
    for operand in &operands[1..] {
        let step = match op {
            BooleanOp::Union => format!("min({acc}, {operand})"),
            // Subtracting is intersecting with the complement, and the complement of a
            // distance field is its negation.
            BooleanOp::Difference => format!("max({acc}, -({operand}))"),
            BooleanOp::Intersection => format!("max({acc}, {operand})"),
        };
        body.push_str(&format!("    {acc} = {step};\n"));
    }
    Some(acc.to_string())
}

/// The same fold, carrying the normal of whichever operand defined the value — the nearest for
/// a union, the farthest for an intersection, the kept body or the farthest complement for a
/// difference, negated where a complement won. Ties go to the earlier operand, as
/// `GraphField::normal_at` gives them.
fn fold_boolean_with_normal(
    body: &mut String,
    op: BooleanOp,
    n: usize,
    values: &[String],
    normals: &[String],
) -> Option<(String, String)> {
    let (first_value, first_normal) = (values.first()?, normals.first()?);
    if values.len() == 1 {
        return Some((first_value.clone(), first_normal.clone()));
    }
    let (value, normal) = (format!("bv{n}"), format!("bg{n}"));
    body.push_str(&format!(
        "    var {value} = {first_value};\n    var {normal} = {first_normal};\n"
    ));
    for (v, g) in values[1..].iter().zip(&normals[1..]) {
        body.push_str(&match op {
            BooleanOp::Union => format!(
                "    if ({v} < {value}) {{ {value} = {v}; {normal} = {g}; }}\n"
            ),
            BooleanOp::Intersection => format!(
                "    if ({v} > {value}) {{ {value} = {v}; {normal} = {g}; }}\n"
            ),
            BooleanOp::Difference => format!(
                "    if (-({v}) > {value}) {{ {value} = -({v}); {normal} = -({g}); }}\n"
            ),
        });
    }
    Some((value, normal))
}

fn value_of(
    slot: &HashMap<(NodeId, usize), usize>,
    at: (impl Into<Operand>, usize),
) -> Result<String, CodegenError> {
    let at = (source_of(at.0), at.1);
    slot.get(&at)
        .map(|n| format!("v{n}"))
        .ok_or(CodegenError::DanglingNode(at.0))
}

fn normal_of(
    slot: &HashMap<(NodeId, usize), usize>,
    at: (impl Into<Operand>, usize),
) -> Result<String, CodegenError> {
    let at = (source_of(at.0), at.1);
    slot.get(&at)
        .map(|n| format!("g{n}"))
        .ok_or(CodegenError::DanglingNode(at.0))
}

/// What a uploaded polyline is for, which decides what the caller must put in it.
///
/// A **spine** is a curve in world coordinates, walked by `pipe_distance`.
///
/// A **profile** is the same curve read in a block's own frame and flattened: `x` and `y` across
/// the plane for an extrude, `(r, z)` for a revolve. Pre-transformed rather than rotated in the
/// shader, because the frame is fixed while the polyline is walked once per *sample* — turning
/// every point on every ray step would pay for a rotation the caller can do once. Keyed by the
/// node rather than the curve, since one curve can be the profile of two blocks aimed
/// differently, and those are two different polylines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveUse {
    Spine(CurveId),
    Profile(NodeId),
    /// A [`fieldform_scene::Node::Point`] an attractor is drawn toward, as a spine of one
    /// vertex.
    ///
    /// **A polyline with a single point, and the shader needs no arm for it.** `closest_on_curve`
    /// answers a span of one with that point — it only walks segments when there are two or more
    /// — so the whole of what a point target costs here is a buffer entry and this variant to
    /// name it. The alternative was teaching the warp about a second kind of target, which would
    /// be two descriptions of "the nearest place on the thing being pulled toward".
    Point(NodeId),
    /// The frame of a [`fieldform_scene::Node::Sweep`]: the rotation-minimising normal at each
    /// sample of its spine, as a polyline of unit vectors the same length as the spine's.
    ///
    /// Keyed by the node, since the frame is seeded from the profile and two sweeps along one
    /// curve with different profiles start differently. Read in step with the spine's own
    /// binding -- `sweep_distance` takes sample `i` of both -- so the caller hands back the frame
    /// computed from the very polyline it uploaded as the spine; `fieldform_field::sweep_frame`
    /// takes those points for exactly that reason.
    Frame(NodeId),
    /// The miter vectors of the section a block reads: where each of its corners goes per
    /// millimetre of draft, as a polyline the same length as the section's own, in step with
    /// it. Only bound when that block's draft **mitres** its corners; a rounded draft needs
    /// nothing of the kind, since it is the section's own reading less a number.
    ///
    /// **A binding of its own rather than two more floats on each point**, which is the shape
    /// a sweep's frame already has. The point buffer packs `(x, y, z, arc)` for every curve
    /// alike, so a profile's spare lanes are not spare in the packer, and widening them for
    /// one block's sake would touch every curve in the document.
    ///
    /// Keyed by the node, as a profile and a wall are: the miters are a property of the
    /// section *as that block lays it*, and `fieldform_field::Profile::miters` is what
    /// computes them so that the shader displaces the same boundary the field does.
    Miters(NodeId),
    /// One of the three polylines a wall is read from, whichever of the three blocks that
    /// take an outline it is on -- an extrude, a revolve or a sweep: the curve it runs along,
    /// and the two lists of bare edges worked out from it. `fieldform_field::Wall::packed` is
    /// what puts each one in the form the shader reads, and `fieldform_field::outline_wall` is
    /// what builds the wall to ask — so a caller has one thing to call and nothing to decide.
    ///
    /// Keyed by the node, as a profile is and for the same reason. **The chain may be uploaded
    /// with a bake, and it is the wall's own** — `Wall::bake`, a grid of the band's distance —
    /// never the bake of the loop the wall runs along, which the shader would have to fold and
    /// cannot safely. `fieldform_field::outline_wall_baked` is what hands back a wall baked by
    /// the rule the field compiles under. The other two polylines never carry one.
    ///
    /// **The one binding whose contents follow a number.** The bare edges are where the band's
    /// own outline is left uncovered, so they move with the thickness: a thickness changing is
    /// a slot refilled *and* two of these lists built again. The application does both, because
    /// every change to a parameter asks for a rebuild (`after_parameter_change`) and a rebuild
    /// uploads every binding. A caller that only refilled `params` on a new value would draw
    /// the old ends on the new band.
    Wall(NodeId, fieldform_field::WallPart),
    /// The centerline of a [`fieldform_scene::Node::Loft`], as `fieldform_field::loft_built`
    /// flattened it. Its frame is a [`CurveUse::Frame`] of the same node, in step with it.
    ///
    /// **Keyed by the node, not by a curve**, unlike a sweep's spine: a loft with no centerline
    /// of its own derives one that is no curve of the document's, and one that is named is
    /// flattened by the loft's own build so that the stations in `params` are measured along
    /// the polyline the shader walks. A spine shared with a sweep would be flattened twice.
    LoftSpine(NodeId),
    /// Section `k` of a loft, projected square to its centerline at its own station, written
    /// relative to the centerline point there, and **resampled into correspondence with the
    /// other sections** -- vertex `j` of each runs to vertex `j` of the next -- by
    /// `fieldform_field::Lofted`, whose `section_points` these are. The shader makes the
    /// section between two stations from these vertex by vertex, so every section carries the
    /// same count; walked, never baked, since the polygon it walks is made on the way.
    LoftSection(NodeId, usize),
    /// The rail slopes of section `k` of a smooth loft, one per vertex and in step with the
    /// section's own binding -- `Lofted::section_slopes`, as `(du/da, dv/da, 0)`. Bound only
    /// at G1: a vertex on a rail is then two points and two slopes read, where computing the
    /// slopes in the shader read four points and two stations, measured at six reads a vertex
    /// against four by `document_cost.rs` on the owner's loft. A linear loft names its section
    /// binding in the slot instead, and the shader never reads it.
    LoftSlope(NodeId, usize),
}

/// Which curve binding an attractor's target occupies.
///
/// One function so the three passes cannot disagree about it: the value pass interns, and the
/// point and normal passes look up what it interned. They used to spell `CurveUse::Spine(curve)`
/// three times, which was fine while there was one kind of target.
fn attractor_use(toward: fieldform_scene::Attractor) -> CurveUse {
    match toward {
        fieldform_scene::Attractor::Curve(id) => CurveUse::Spine(id),
        fieldform_scene::Attractor::Point(id) => CurveUse::Point(id),
    }
}

/// Index of `id` among the things to bind, appending it if this is its first use.
///
/// Reusing the index on repeat is what makes a graph referencing one cage twice bind a single
/// texture rather than two copies of the same bake.
fn intern<T: PartialEq + Copy>(bindings: &mut Vec<T>, id: T) -> usize {
    match bindings.iter().position(|x| *x == id) {
        Some(i) => i,
        None => {
            bindings.push(id);
            bindings.len() - 1
        }
    }
}

/// How many members a run needs before it is emitted as a loop rather than as a call each.
///
/// **Measured, not chosen** — `fieldform-gpu/examples/union_loop_cost.rs`, RTX A500 on Vulkan,
/// the same spokes both ways in one process on one device, cold pipeline and the least of six
/// rounds of twenty frames:
///
/// | pipes | compile, as calls → as a loop | 720 px | 1440 px |
/// |---|---|---|---|
/// | 2 | 623 → **295** | 7.1 → 8.2 | 24.6 → 25.1 |
/// | 3 | 1 467 → **276** | 7.5 → 8.4 | 26.9 → 26.7 |
/// | **4** | 4 464 → **273** | 8.8 → 8.9 | 28.2 → 27.0 |
/// | 5 | 8 801 → **278** | 9.6 → 9.3 | 33.4 → 27.5 |
/// | 8 | 18 073 → **279** | 15.6 → 11.3 | 49.8 → 35.9 |
/// | 16 | 95 060 → **339** | 36.4 → 18.7 | 115.7 → 60.7 |
///
/// **Four, because that is where the frame stops regressing** — not where the compile starts
/// paying, which is two. A loop over two walks replaces two walks, so it saves one copy of the
/// code and adds a loop the driver would otherwise have unrolled for free, and at two and three
/// pipes that reads 12–15% slow at 720 px. The noise floor is not asserted either: the two sphere
/// arms compile to byte-identical source, and their spread is up to 9% at 720 px and 1% at 1440,
/// which is what makes four the first count whose +1% is nothing and three's +12% is something.
///
/// Past four it is one-sided and then overwhelming. **At sixteen: 280× the compile and half the
/// frame** — the same reversal item 34 measured on the pattern's window, where unrolling was
/// spilling registers and the compact loop was not. The loop's own compile is flat (273–339 ms
/// over 2 to 16), which is the whole point: one walk, however many pipes.
///
/// **And the same floor for a fan of blades — checked, not inherited.** An extrude is the other
/// kind of run and a heavier one, 2 328 inlined lines an operand against a pipe's 1 835, so there
/// was no reason to suppose four was still the number. Same example, same machine; the pipes'
/// compile column is left out of this run because the driver served it from its own disk cache,
/// which is what a second reading of a source always is:
///
/// | blades | compile, as calls → as a loop | 720 px | 1440 px |
/// |---|---|---|---|
/// | 2 | 656 → **373** | 6.1 → 6.2 | 25.9 → 26.6 |
/// | 3 | 883 → **375** | 7.5 → 7.8 | 26.6 → **27.8** |
/// | **4** | 1 856 → **377** | 8.1 → 7.9 | 28.6 → 28.3 |
/// | 5 | 2 618 → **382** | 8.2 → 8.3 | 26.4 → 25.7 |
/// | 8 | 6 615 → **380** | 8.8 → 7.5 | 32.3 → 29.6 |
/// | 16 | 49 148 → **394** | 28.3 → 12.4 | 87.8 → 43.5 |
///
/// Four again, and read off the 1440 column rather than the 720 one: the sphere controls spread up
/// to 4% at 720 that day and 1.3% at 1440, so three blades' **+4.5% at 1440** is the one
/// regression in the table larger than the noise beside it, and four's −1.0% is not a regression
/// at all. At sixteen it is **125× the compile and the frame halved** — 2.3× at 720 px and 2.0× at
/// 1440 — and the loop's own compile is flat from 373 to 394 ms across the whole range.
///
/// So one number for both kinds, which is worth more than a number each: a floor per kind would
/// be two things to re-measure whenever the inlining changes, and the measurement says they do
/// not need to differ.
const RUN_FLOOR: usize = 4;

/// How many thin extrudes a union needs before they are walked rather than written out.
///
/// **A number of its own, which `RUN_FLOOR`'s last paragraph argues against, and the
/// measurement is what overrules it.** That floor was set on the frame because below it the
/// compile was not worth arguing over. A wall's is: `fieldform-gpu/examples/wall_cost.rs`, each
/// blade the line down a rectangle's middle given the rectangle's width, as calls and then as
/// a loop.
///
/// | thin blades | compile ms, as calls → as a loop | 720 px | 1440 px |
/// |---|---|---|---|
/// | **2** | 6 331 → **2 417** | 6.3 → 6.5 | 24.4 → 24.6 |
/// | 3 | 13 849 → **2 259** | 6.5 → 6.1 | 26.2 → 25.5 |
/// | 4 | 24 543 → **2 086** | 6.5 → 6.4 | 25.0 → 25.3 |
/// | 8 | 177 694 → **2 092** | 11.6 → 9.5 | 41.1 → 33.1 |
///
/// The loop's compile is flat, 2.1 to 2.4 seconds whatever the count, and written out it is
/// six seconds at two — so two is where the loop starts paying, by two and a half times, and
/// the frame is no reason to wait for more. From two blades to four the arms are within six
/// per cent of each other at 720 px and three at 1440, and that is inside the noise: one blade
/// is the same source in both arms, and read 5.7 and 6.0 at 720 px and 23.3 and 22.0 at 1440.
/// At eight the loop is a fifth faster.
///
/// Every figure is a first reading, from one run, taken after walls were given a bake — so
/// each blade here answers from a grid, as the application's would. Before that the frames
/// were dearer and the same table said the same thing: 7 342 → 2 098 at two, 113 298 → 2 227
/// at eight, with the frame at eight going from 60.2 ms to 42.8.
const WALL_RUN_FLOOR: usize = 2;

/// Where a run's bindings start in the curve list, if they are one consecutive stretch.
///
/// **The condition both run loops rest on**, checked rather than assumed: they read binding
/// `first + k`, so a run whose bindings are interleaved with another node's curve would read the
/// wrong ones and draw a plausible wrong picture. `looped_run` refuses a member anything else
/// consumes, which is what makes them contiguous in practice — but the emission order is what
/// actually decides it, and the emission order is not something the graph can promise.
///
/// **Both passes reach the same answer**, which is what keeps their parameter counting in step.
/// The value pass asks after the run's members have interned — they are emitted before the
/// boolean that reads them — and interning only ever appends, so a stretch that is contiguous
/// then is contiguous in the final list the normal pass asks about.
fn run_first_curve(curves: &[CurveUse], run: Option<&OperandRun>) -> Option<usize> {
    let run = run?;
    let first = curves.iter().position(|x| *x == run.curves[0])?;
    for (k, want) in run.curves.iter().enumerate().skip(1) {
        if curves.get(first + k) != Some(want) {
            return None;
        }
    }
    Some(first)
}

/// What a run is made of, and so which loop walks it and how many slots each member takes.
///
/// **One enum rather than two run types.** A boolean takes at most one run, and both passes have
/// to take *the same* one; two independent detectors would each have to agree with the other
/// about which of them won, in two places, and a disagreement shifts every parameter slot after
/// the boolean with nothing at all to say so.
enum RunKind {
    /// Pipes: a spine binding and one radius slot each.
    ///
    /// The radii themselves, because they are `ParamRef`s the graph already holds and carrying
    /// them reads no value.
    Pipes(Vec<ParamRef>),
    /// Extrudes: a profile binding and twelve slots each.
    ///
    /// The nodes rather than the numbers, because most of an extrude's slots are its *placement*
    /// — the plane worked out from the profile's own points — and that is `extrude_placement`'s
    /// to compute at emission, from the document, exactly as the single-extrude arm computes it.
    /// Resolving them here would mean reading values while deciding whether to loop, which is
    /// the one thing `generate`'s contract forbids.
    Extrudes(Vec<NodeId>),
    /// Thin extrudes: a wall's three bindings and sixteen slots each — a thick extrude's twelve,
    /// then the four `wall_slots` pushes.
    ///
    /// **A kind of its own rather than a flag on `Extrudes`**, because a loop has one body and
    /// these are two calls: a member that was sometimes a profile and sometimes a wall would
    /// make every thick extrude in a run carry four slots and a branch it never takes, and
    /// would change the text generated for a document with no wall in it.
    Walls(Vec<NodeId>),
}

impl RunKind {
    /// How many parameter slots one member takes.
    ///
    /// **The number the shader steps by**, and the number the normal pass counts without reading
    /// anything: the `12u` in `extrude_run_distance` and the 12 here are the same number by
    /// agreement, and `a_looped_run_pushes_the_same_slots_in_both_passes` is what holds them to
    /// it. Sixteen and `wall_extrude_run_distance` likewise, held by
    /// `a_fan_of_thin_extrudes_is_one_walk`.
    fn slots_each(&self) -> usize {
        match self {
            RunKind::Pipes(_) => 1,
            RunKind::Extrudes(_) => 9 + DEPTH_SLOTS,
            RunKind::Walls(_) => 9 + DEPTH_SLOTS + 4,
        }
    }

    /// How many curve bindings one member reads, which the shader also steps by: a wall is
    /// read from three polylines where a spine or a profile is one.
    fn bindings_each(&self) -> usize {
        match self {
            RunKind::Pipes(_) | RunKind::Extrudes(_) => 1,
            RunKind::Walls(_) => WALL_PARTS.len(),
        }
    }

    /// The loops that walk this kind: the value's, then the one that carries a normal too.
    fn walkers(&self) -> (&'static str, &'static str) {
        match self {
            RunKind::Pipes(_) => ("pipe_run_distance", "pipe_run_nearest"),
            RunKind::Extrudes(_) => ("extrude_run_distance", "extrude_run_nearest"),
            RunKind::Walls(_) => ("wall_extrude_run_distance", "wall_extrude_run_nearest"),
        }
    }

    /// How many members make a run of this kind worth walking, given the floor asked for.
    ///
    /// **Two for walls where it is four for the others, and for the reason four was chosen.**
    /// `RUN_FLOOR` is where a pipe's or a blade's *frame* stops regressing, and it could be
    /// chosen on the frame because below it the compile was not worth arguing over — 656 ms
    /// for two blades as calls. A wall is a heavier call, and two of them written out are
    /// 6.3 seconds of the driver's time against the loop's 2.4. `WALL_RUN_FLOOR` has the table.
    ///
    /// A floor above the usual one is a measurement asking for calls, and is given them.
    fn least(&self, floor: usize) -> usize {
        match self {
            RunKind::Walls(_) if floor <= RUN_FLOOR => floor.min(WALL_RUN_FLOOR),
            _ => floor,
        }
    }
}

/// A stretch of a union's operand list that is all one loopable block, which becomes one loop.
///
/// Held as the curve bindings rather than as node ids because that is what the emission needs,
/// and in list order because the loop walks it in that order — which is what makes a tie between
/// two members go the same way here as in `GraphField::normal_at`.
struct OperandRun {
    /// Where in the operand list the run starts, so it is emitted in place and the fold's order
    /// is the order the graph holds.
    at: usize,
    /// The bindings the loop reads as `first + k` — three a member for a run of walls, so
    /// `first + 3k` and the two after it — in list order. Interned consecutively, which is the
    /// condition every loop rests on.
    curves: Vec<CurveUse>,
    kind: RunKind,
}

impl OperandRun {
    /// How many members, which is not how many bindings: a wall is read from three.
    fn len(&self) -> usize {
        self.curves.len() / self.kind.bindings_each()
    }

    /// Every slot the run pushes, the count's own included — what the normal pass advances by.
    fn slots(&self) -> usize {
        self.len() * self.kind.slots_each() + 1
    }
}

/// The longest run of a union's operands that may be walked rather than written out, or `None`.
///
/// **One function, called by the value pass and the normal pass**, because the two agree about
/// parameter slots by counting the same things in the same order (`Node::Pipe`'s arm says so) and
/// a run taken in one pass and not the other would shift every slot after it with nothing to say
/// so. It reads the graph alone — no document, no number — so it cannot make a shader depend on a
/// value, which is `generate`'s standing contract.
///
/// What a member has to be, and why each condition is here rather than hopeful:
///
/// - **A filled slot.** A hole is not a pipe and not an extrude, so it ends a run where it stands
///   -- and it is read here rather than refused, because this walks the whole graph before
///   `generate` has refused anything, a floating holed block included.
/// - **A union.** `min` is commutative and associative, so a run may be folded in any order and
///   in place. A difference and an intersection are neither indifferent to order nor uniform in
///   sign across the list, and the measured shape is a union.
/// - **A pipe whose radius is a plain number.** A driven radius compiles to the offset expression
///   instead — a different body, with a divisor of its own — so it cannot share a loop body. An
///   extrude needs no condition of this sort: every number on it is already a parameter slot, and
///   `extrude_placement` resolves the plane and the sweep direction to nine more whichever
///   `Sweep` it holds, so any two extrudes differ only in what is in the buffer.
/// - **Consumed by this boolean and nothing else, and — for a pipe — its spine used as a spine
///   by nothing else.** Together these are what make the interning consecutive: a curve some
///   earlier node already interned would come back at its old index and the run would read the
///   wrong bindings. A profile is keyed by its *node*, so no second node can have interned one;
///   a second consumer is still refused, because then the member's own line is still read and the
///   run would be a second copy of it rather than a replacement. Checked on the graph rather than
///   on the interning, so that both passes reach the same answer without either of them having to
///   see the other's state.
/// - **`RUN_FLOOR` of them, or `WALL_RUN_FLOOR` of thin extrudes.** Below that there is nothing
///   to save — and for a wall "nothing" starts lower, which `RunKind::least` says why.
///
/// **One run, of one kind.** A loop has one body, so a stretch ends where the block changes: a
/// pipe beside an extrude ends both runs rather than joining them, and so does a thin extrude
/// beside a thick one, which is a different call with three bindings and four more slots. Where
/// a union holds two runs the longest wins and the earlier wins a tie — a decision taken here,
/// and so taken identically by both passes. **What that leaves** is a union of two long
/// stretches of different kinds, one of which is still written out: eight blades and eight
/// ribs in one union loop the blades or the ribs and not both.
fn looped_run(
    op: BooleanOp,
    graph: &Graph,
    inputs: &[Operand],
    floor: usize,
) -> Option<OperandRun> {
    if !matches!(op, BooleanOp::Union) {
        return None;
    }
    // Every consumer of every node, and every use of a curve as a spine, counted once for the
    // whole graph rather than per candidate.
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    let mut spine_uses: HashMap<CurveId, usize> = HashMap::new();
    for (_, node) in graph.nodes() {
        for input in node.inputs() {
            *consumers.entry(input).or_insert(0) += 1;
        }
        match node {
            Node::Pipe { curve, .. } => *spine_uses.entry(*curve).or_insert(0) += 1,
            Node::Sweep { spine, .. } => *spine_uses.entry(*spine).or_insert(0) += 1,
            Node::Attract { toward: fieldform_scene::Attractor::Curve(id), .. } => {
                *spine_uses.entry(*id).or_insert(0) += 1
            }
            _ => {}
        }
    }
    let plain_pipe = |slot: &Operand| -> Option<(CurveId, ParamRef)> {
        let id = slot.source()?;
        let Some(Node::Pipe { curve, radius }) = graph.node(id) else { return None };
        if radius.driver().is_some()
            || consumers.get(&id).copied().unwrap_or(0) != 1
            || spine_uses.get(curve).copied().unwrap_or(0) != 1
        {
            return None;
        }
        Some((*curve, radius.clone()))
    };
    // An extrude this boolean alone consumes. Nothing on the node itself can disqualify it: the
    // sweep, the draft and the symmetric flag are all parameter slots, so every extrude compiles
    // to the same call and any two of them share a loop body.
    // **Of a closed profile, or of a wall, and the two are different kinds of run.** A thin
    // extrude compiles to a different call, with three bindings and four more slots, so it
    // cannot share a loop body with a thick one: each ends the other's run, the way a pipe
    // beside an extrude does, and a stretch of either is a run of its own kind.
    let extrude_of = |slot: &Operand, thin: bool| -> Option<NodeId> {
        let id = slot.source()?;
        let Some(Node::Extrude { outline, rims, .. }) = graph.node(id) else { return None };
        if outline.is_wall() != thin || consumers.get(&id).copied().unwrap_or(0) != 1 {
            return None;
        }
        // **A rimmed extrude compiles to the `_rim_` call**, which the loop body has no slot
        // for, so it ends a run the way a pipe beside an extrude does.
        if !rims.is_plainly_hard() {
            return None;
        }
        Some(id)
    };
    let mut best: Option<OperandRun> = None;
    let mut at = 0usize;
    while at < inputs.len() {
        // Every stretch from here, of which at most one can be non-empty — a node is a pipe, a
        // thick extrude or a thin one and never two of them — so whichever starts at `at` is
        // the kind of the run there.
        let mut pipes = Vec::new();
        while let Some(found) = inputs.get(at + pipes.len()).and_then(|slot| plain_pipe(slot)) {
            pipes.push(found);
        }
        let stretch_of = |thin: bool| -> Vec<NodeId> {
            let mut found = Vec::new();
            while let Some(id) = inputs.get(at + found.len()).and_then(|s| extrude_of(s, thin)) {
                found.push(id);
            }
            found
        };
        let (extrudes, walls) = (stretch_of(false), stretch_of(true));
        let here: Option<(Vec<CurveUse>, RunKind)> = if !pipes.is_empty() {
            Some((
                pipes.iter().map(|(c, _)| CurveUse::Spine(*c)).collect(),
                RunKind::Pipes(pipes.iter().map(|(_, r)| r.clone()).collect()),
            ))
        } else if !extrudes.is_empty() {
            Some((
                extrudes.iter().map(|id| CurveUse::Profile(*id)).collect(),
                RunKind::Extrudes(extrudes),
            ))
        } else if !walls.is_empty() {
            // A member's three polylines together and in the order `wall_extrude_distance`
            // takes them, so the loop reads member `k` at `first + 3k` and the two after it.
            Some((
                walls
                    .iter()
                    .flat_map(|id| WALL_PARTS.map(|part| CurveUse::Wall(*id, part)))
                    .collect(),
                RunKind::Walls(walls),
            ))
        } else {
            None
        };
        // `max(1)` so an operand that is none of the kinds still advances the scan, and `max(2)`
        // below because one member is not a run whatever the floor says — and because the
        // measuring switch passes `usize::MAX` to mean never.
        let here = here.map(|(curves, kind)| OperandRun { at, curves, kind });
        let step = here.as_ref().map_or(1, OperandRun::len).max(1);
        if let Some(run) = here {
            if run.len() >= run.kind.least(floor).max(2)
                && best.as_ref().is_none_or(|b| run.len() > b.len())
            {
                best = Some(run);
            }
        }
        at += step;
    }
    best
}

fn operand(
    slot: &HashMap<(NodeId, usize), usize>,
    at: (impl Into<Operand>, usize),
) -> Result<String, CodegenError> {
    let at = (source_of(at.0), at.1);
    slot.get(&at)
        .map(|n| format!("n{n}"))
        .ok_or(CodegenError::DanglingNode(at.0))
}

/// The block a slot names -- and `Into`, so one reader serves both an [`Operand`] and the bare
/// [`NodeId`] a driver, a body or a critical mark is named by.
///
/// **An empty slot cannot arrive here**, which is arranged rather than hoped for: `generate_inner`
/// refuses a part whose plan holds anything [`Graph::incomplete`] names, before a line is written,
/// and that set is closed downstream -- so a node either pass reaches has every operand. That one
/// check is what lets the thirty-seven operand reads stay the plain lookups they were rather than
/// each growing a guard of its own. Said here rather than left to a missing key, which would come
/// out as `DanglingNode` and send whoever read it looking for a node that is present and whole;
/// the CPU's `Slots` index answers the same question in the same place, the same way.
fn source_of(at: impl Into<Operand>) -> NodeId {
    match at.into().source() {
        Some(id) => id,
        None => unreachable!("a part with an empty slot is refused before any line is written"),
    }
}

/// The refusal a hole gets, named from the slot that is empty.
///
/// **Not from the block that met it**, which is rarely the one to fill in: [`Graph::incomplete`]
/// is closed downstream, so the block the plan was refused for can be any number of blocks above
/// the empty slot. In id order, through [`Graph::first_empty_operand`], so one graph always refuses
/// with the same block however its nodes happen to be stored -- which is D7's ask of this crate
/// as much as it is a message somebody can act on. `met` stands in only if the hole has closed
/// between the walk and the refusal, which nothing does. `CompileError`'s `empty_slot` is this
/// function on the CPU side, worded to match.
fn empty_slot(graph: &Graph, met: NodeId) -> CodegenError {
    CodegenError::EmptySlot(graph.first_empty_operand().map_or(met, |(node, _)| node))
}

/// The open cage a body is, or reads, if any -- the same question `compile.rs` asks, so the two
/// refuse one part for one reason.
fn red_cage_behind(
    document: &fieldform_scene::Document,
    graph: &Graph,
    body: NodeId,
) -> Option<fieldform_scene::CageId> {
    graph.nodes().find_map(|(id, n)| match n {
        Node::Cage(cage)
            if document.cage_is_red(*cage) && (id == body || graph.downstream_of(id).contains(&body)) =>
        {
            Some(*cage)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod boolean_list_tests;

/// What the generator writes for a thin extrude, and what it keeps out of the text.
///
/// `gpu_matches_cpu` says whether the shader finds the right surface; these say whether the
/// program is *shaped* right — the bindings in the order the call reads them, the two passes
/// counting the same slots, and nothing about a wall in the source but the fact of one.
#[cfg(test)]
mod thin_extrude_tests;

/// What the generator writes for a profile solid's rims: a hard one the text it always was, a
/// rounded one the `_rim_` call with one slot more, counted the same by both passes.
#[cfg(test)]
mod rims_tests;

/// What the generator writes for the other two blocks an outline went onto: a thin revolve and
/// an open-section sweep. Shaped as a thin extrude's program is, and held to the same things.
#[cfg(test)]
mod thin_revolve_and_sweep_tests;
