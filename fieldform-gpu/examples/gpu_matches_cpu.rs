// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Checks that the shader we generate finds the same surface the CPU field describes.
//!
//! This is the correctness anchor for the whole GPU path. Two independent implementations now
//! answer "where is the surface": `GraphField` walking the graph in `f64` on the CPU, and a
//! generated WGSL function evaluating it in `f32` on the GPU with cages read from a baked
//! texture. They are written differently, run differently, and can drift apart silently —
//! a wrong shader still produces a confident picture.
//!
//! # How agreement is checked
//!
//! Not by comparing images. Two renderers will always differ in the last pixel of a silhouette,
//! and chasing that tells you nothing about whether the geometry is right.
//!
//! Instead: for every pixel the GPU says it hit, take the hit point and ask the **CPU** field
//! how far that point is from the surface. If the GPU found the real surface, the answer is
//! about zero. If it found something else — the wrong side of a boolean, a mis-addressed
//! texel, a blend with the wrong radius — the answer is not, and the size of the error says
//! which. One number, and it cannot be satisfied by a picture that merely looks plausible.
//!
//! # It fails, rather than only reporting
//!
//! For a long time this computed all of the above and then exited 0 whatever the numbers said,
//! so the anchor could only fail by being read. It now ends with a table of every scene's worst
//! beside the budget it is held to, and exits **1** if any scene is over its budget or returned
//! no hits at all, **2** if a `FIELDFORM_SCENE` filter matched nothing — a run that checked
//! nothing is not a pass. See [`failure_budget_of`] for where the budgets come from and why they
//! are deliberately loose. Every line that was printed before is still printed.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example gpu_matches_cpu`

use fieldform_cage::Cage;
use fieldform_field::{GraphField, ScalarField};
use fieldform_gpu::{Camera, Renderer};
use fieldform_scene::{Fillet, 
    BooleanOp, CageId, Continuity, Curve, CurveId, Document, Graph, LatticePhase, Node, NodeId,
    ParamRef, Part, PartId,
};
use nalgebra::Point3;

const CAGE_MM: f64 = 1000.0;
const PIPE_RADIUS_MM: f64 = 51.0;
const BLEND_RADIUS_MM: f64 = 12.0;
/// Curve tessellation and bake band, both in millimetres.
const TOLERANCE_MM: f64 = 0.25;

/// The tolerance one scene is compiled and baked at.
///
/// **Per scene, so that one of them can be coarse enough to bake a profile.** A profile answers
/// from a grid only when the tolerance allows the grid to stay inside it — which is the rule that
/// keeps a bake a display convenience — and at a quarter of a millimetre over a body this size it
/// never does. So the baked path would have shipped unmeasured, which is the one thing this
/// example exists to stop: the arithmetic in the shader's `sample_bake` has no other check on it.
/// The worst surface error, in millimetres, at which a scene is called a disagreement.
///
/// **Separate from [`tolerance_of`], which is what a scene is *built* at.** This is what its
/// answer is *held to*, and until this existed there was none: the example printed its residuals
/// and exited 0 whatever they said, so the correctness anchor for the whole GPU path could only
/// fail by being read.
///
/// The figures it is set against: `attractor-curve-spec.md` §"measured" records mean 0.0112 mm
/// and 99th percentile 0.0130 mm for the suite, and says the percentile "is the same 0.013 mm
/// every other scene in that suite reports, which is the marcher's own resolution". So the floor
/// under any honest budget is about a hundredth of a millimetre, and a disagreement worth the
/// name — a boolean taken the wrong way, a mis-addressed texel, a blend at the wrong radius — is
/// tens of millimetres on a body a metre across. There is a great deal of room between those two,
/// and these budgets sit in it deliberately.
///
/// **Set from a real run, 2026-09-27.** They were first chosen without one — the container this
/// was written in had no GPU — and the note here said every figure "should come down to what the
/// hardware actually shows" once a run printed the table at the end of `main`. It has, so they
/// have. Sixty scenes at 512², every one inside the budget it was then held to.
///
/// ```text
/// group                            scenes   worst mm   budget
/// the marcher's own resolution         50     0.0135     0.05
/// the cage bake                         5     0.2264     0.40
/// two sweep variants                    2     0.1502     0.30
/// the conformal map                     2     0.7483     1.00
/// the attractor's cut locus             1     0.8478     1.50
/// ```
///
/// **The floor is now measured rather than cited.** Fifty of the sixty scenes land between 0.0130
/// and 0.0135 mm — half a micron of spread across every boolean, lattice, pattern, extrude and
/// revolve here — which confirms empirically the 0.013 mm `attractor-curve-spec.md` calls "the
/// marcher's own resolution". 0.05 mm is about 3.7× that: room for another adapter's `f32`
/// rounding, and none at all for a boolean taken the wrong way.
///
/// Three groups keep more, now for measured reasons rather than guessed ones:
///
/// - **`Attracted`** measures 0.8478 mm here against the 0.8555 mm `attractor-curve-spec.md`
///   records, so that account holds on real hardware: "the polyline has a corner at x = −300: it
///   is the cut locus above, where the field genuinely jumps and the two marchers stop a step
///   apart. Not a shader disagreement — the same discontinuity, seen twice." A cut locus is a
///   property of the field, so no budget can make it go away and none should try.
/// - **The conformal scenes** stay at 1.0 deliberately. `ConformalInfill`'s 0.7483 mm is the
///   tightest margin in the table and so the first thing that will fire, but [`Scene`]'s own
///   medial-axis measurement says why a closest-point map cannot do better there: the answer is
///   ambiguous rather than wrong. Read that before treating a failure here as a defect.
/// - **The cage scenes** peak at 0.2264 mm, just inside the 0.25 mm band the cage is baked at
///   (`TOLERANCE_MM`) — and all five peak at the *same point in space* whatever boolean is over
///   them, while `PipeOnly`, the one scene with no cage in it, sits down at the floor with
///   everything else. So it is the bake, not the operations. 0.40 mm is 1.8× the measurement.
///
/// **What the run disproved: the baked profiles' 3.0 mm.** That came from [`tolerance_of`] — they
/// are built at a 2.0 mm band rather than 0.25 mm, and "a bake cannot answer more finely than the
/// band it was baked at". They measure 0.0133 and 0.0132 mm, at the floor with every closed-form
/// scene, 225× under the budget written for them. The tolerance governs where the grid may be
/// *used*, not how far off the answer it gives lands, so the special case is gone and they take
/// the default like anything else.
fn failure_budget_of(scene: Scene) -> f64 {
    match scene {
        // The cut locus above.
        Scene::Attracted => 1.5,
        // The medial axis of the closest-point map rather than the transcription — left loose.
        Scene::ConformalInfill | Scene::ConformalShell => 1.0,
        // The 0.25 mm band the cage is baked at, which all five of these peak just inside.
        Scene::CageOnly
        | Scene::Difference
        | Scene::Blend
        | Scene::BlendedDifference
        | Scene::G2Blend
        | Scene::G2BlendedDifference
        | Scene::ChamferBlend
        | Scene::ChamferedDifference
        | Scene::TwoBodies => 0.40,
        // Both sit ~10× the floor where the plain `Swept` does not. Not diagnosed, so the budget
        // covers what was measured rather than pretending to a reason.
        Scene::SweptFlush | Scene::SweptParallel => 0.30,
        // The law'd sweeps, measured rather than assumed: a scaled or turned section reads
        // where its own frame puts it, and the two sides sample the spine's polyline at the
        // same places but land a hair apart on a section that is moving.
        Scene::SweptScaled | Scene::SweptDrafted | Scene::SweptTwistedWindow => 0.30,
        // The tapering sweep's own budget, for its own reason; the rim adds nothing measured
        // (0.0833 against the bare sweep's 0.0837, at the same point).
        Scene::ChamferRimmedTaper => 0.30,
        // **A rim on a divided body is where a ray grazes it.** The two surfaces agree -- 0.0002 mm
        // at the 99th percentile -- and the worst is a few pixels on the rounded edge, each landed
        // inside: 0.0604 at G1, 0.0522 at G2, 0.0528 chamfered, so it is not the G2 divisor. The
        // same rims on an extrude and a loft, undivided, measure 0.013. This sweep reads 4.26x
        // short where only the divided reading holds, which is the whole rounded edge, and the
        // settle lands a little deep there. Measured, not argued: the budget is what was seen.
        // Before the in-tube reading kept the hard corner it was 0.32 -- two surfaces at once.
        Scene::RimmedSweep | Scene::EndRimmedSweep => 0.10,
        Scene::ThinSweptScaled => 0.30,
        _ => 0.05,
    }
}

/// The loft `node` is, built as the field builds it -- `fieldform_field::loft_built`, at the
/// default tolerance whatever the scene's own, since that is the one build the compile, the
/// stations in `params` and the uploaded polylines all come from.
fn loft_of(doc: &Document, graph: &Graph, node: NodeId) -> std::sync::Arc<fieldform_field::Lofted> {
    let Some(Node::Loft { rims: _, sections, spine, smooth }) = graph.node(node) else {
        unreachable!("a loft binding belongs to a loft");
    };
    fieldform_field::loft_built(doc, sections, *spine, *smooth).expect("a loft that compiled builds")
}

fn tolerance_of(scene: Scene) -> f64 {
    match scene {
        // Coarse enough that a profile, or a wall, answers from a grid.
        Scene::BakedProfile
        | Scene::DraftedBake
        | Scene::ThinBaked
        | Scene::ThinTubeBaked
        | Scene::ThinCutBaked
        | Scene::SpokedRibsBaked
        | Scene::ThinRevolvedBaked
        | Scene::ThinRevolvedAcrossBaked
        | Scene::ThinSweptBaked => 2.0,
        // **Fine enough that a wall cannot.** These are the scenes that hold the shader's walk
        // to the field's, and at the quarter millimetre everything else compiles to, whether a
        // wall bakes is an accident of its size: the S did and the loop did not. A scene that
        // slid onto a grid would go on agreeing while checking nothing it was written to check,
        // so the tolerance says which it is, and `main` asserts that it is. The chamfered wall
        // is the thin extrude's own scene and is walked as it is.
        Scene::ThinExtruded
        | Scene::ChamferRimmedWall
        | Scene::ThinLeaning
        | Scene::ThinTube
        | Scene::ThinRib
        | Scene::ThinCut
        | Scene::SpokedRibs
        | Scene::ThinRevolved
        | Scene::ThinRevolvedAcross
        | Scene::ThinRevolvedWedge
        | Scene::ThinSwept => 0.02,
        _ => TOLERANCE_MM,
    }
}

fn main() {
    let width = 512u32;
    let height = 512u32;

    // Every scene's worst, so the run ends with the table the budgets above should be set from.
    // Collected rather than asserted scene by scene: an assert that aborts on the first bad
    // scene hides every scene after it, and this is a diagnostic as much as a gate. It stays
    // one — nothing below removes a line that was printed before.
    let mut measured: Vec<(&'static str, f64, usize, f64)> = Vec::new();

    for scene in [
        Scene::CageOnly,
        Scene::Difference,
        Scene::Blend,
        Scene::BlendedDifference,
        Scene::G2Blend,
        Scene::G2BlendedDifference,
        Scene::ChamferBlend,
        Scene::ChamferedDifference,
        Scene::Hollow,
        Scene::CylindricalInfill,
        Scene::SphericalInfill,
        Scene::ConformalInfill,
        Scene::ConformalShell,
        Scene::SolidInfill,
        Scene::SchoenIwpInfill,
        Scene::NeoviusInfill,
        Scene::FischerKochSInfill,
        Scene::CubicBeamInfill,
        Scene::BccBeamInfill,
        Scene::FccBeamInfill,
        Scene::OctetBeamInfill,
        Scene::DiamondBeamInfill,
        Scene::Mirrored,
        Scene::MirroredAboutAPlane,
        Scene::Twisted,
        Scene::Attracted,
        Scene::Patterned,
        Scene::PatternedRow,
        Scene::PatternedRing,
        Scene::PatternedFillet,
        Scene::G2PatternedFillet,
        Scene::PatternedRun,
        Scene::PatternedArc,
        Scene::PatternedArcFillet,
        Scene::PatternedLongRun,
        Scene::PatternedLongArc,
        Scene::PatternedGrid,
        Scene::PatternedLongGrid,
        Scene::VaryingOffset,
        Scene::ProtectedBore,
        Scene::PipeOnly,
        Scene::TwoBodies,
        Scene::Primitives,
        Scene::RoundedBoxCut,
        Scene::G2BoxCut,
        Scene::ChamferedBoxCut,
        Scene::EdgedCylinders,
        Scene::EdgedCones,
        Scene::G2Cones,
        Scene::RimmedBlendedToChamferedBox,
        Scene::ManyWayDifference,
        Scene::ManyWayUnion,
        Scene::RoundSolids,
        Scene::SkewedBoxes,
        Scene::Morphed,
        Scene::MorphedDriven,
        Scene::BlendDriven,
        Scene::TwoLegChamfers,
        Scene::TwoLegRims,
        Scene::GyroidInfill,
        Scene::SchwarzPInfill,
        Scene::SchwarzDInfill,
        Scene::SphereInfill,
        Scene::WalledInfill,
        Scene::HoneycombInfill,
        Scene::VoronoiFoamInfill,
        Scene::VoronoiBeamInfill,
        Scene::GradedWall,
        Scene::GradedDensity,
        Scene::GradedStrut,
        Scene::Extruded,
        Scene::Leaning,
        Scene::Upright,
        Scene::BakedProfile,
        Scene::Drafted,
        Scene::DraftedBake,
        Scene::DraftedFlip,
        Scene::Symmetric,
        Scene::SymmetricDrafted,
        Scene::TwoSidedBegun,
        Scene::RimmedDrafted,
        Scene::RimmedTwoSided,
        Scene::ChamferRimmedWall,
        Scene::Revolved,
        Scene::ThinExtruded,
        Scene::ThinLeaning,
        Scene::ThinTube,
        Scene::ThinRib,
        Scene::ThinCut,
        Scene::Swept,
        Scene::SweptFlush,
        Scene::SweptParallel,
        Scene::SharpDraft,
        Scene::DraftedWallSweep,
        Scene::SweptScaled,
        Scene::SweptDrafted,
        Scene::SweptTwistedWindow,
        Scene::Lofted,
        Scene::RimmedLoft,
        Scene::EndRimmedTwoSided,
        Scene::StartRimmedLoft,
        Scene::EndRimmedSweep,
        Scene::RimmedWedge,
        Scene::StartRimmedThinWedge,
        Scene::RimmedSweep,
        Scene::ChamferRimmedTaper,
        Scene::LoftedG0,
        Scene::LoftedDerived,
        Scene::ThinSweptScaled,
        Scene::SpokedPipes,
        Scene::SpokedBlades,
        Scene::SpokedRibs,
        Scene::ThinBaked,
        Scene::ThinTubeBaked,
        Scene::ThinCutBaked,
        Scene::SpokedRibsBaked,
        Scene::ThinRevolved,
        Scene::ThinRevolvedAcross,
        Scene::ThinRevolvedWedge,
        Scene::RevolvedWedge,
        Scene::ThinSwept,
        Scene::ThinRevolvedBaked,
        Scene::ThinRevolvedAcrossBaked,
        Scene::ThinSweptBaked,
    ] {
        // `FIELDFORM_SCENE=bore` runs only the scenes whose label holds that text: the whole
        // list is minutes, and the one being worked on is seconds.
        if std::env::var("FIELDFORM_SCENE").is_ok_and(|only| !scene.label().contains(&only)) {
            continue;
        }
        println!("\n=== {} ===", scene.label());
        let (doc, graph, curve_points) = scene.build();
        let part = Part {
            id: PartId::new(),
            name: "test".into(),
            material: None,
            graph: graph.clone(),
            inputs: Vec::new(),
        };

        let cpu = GraphField::compile(&doc, &part, tolerance_of(scene)).expect("the graph compiles");
        // The CPU field's own divisors, not a second derivation of them. A driven morph is
        // bounded from its operands' intervals, which exist only in the compiled field, so
        // letting the shader work out its own would divide the two paths by different numbers
        // and put one surface in two places -- which is the thing this example exists to catch.
        let program = fieldform_gpu::generate_with(&doc, &graph, cpu.morph_divisors())
            .expect("the graph generates a shader");
        println!(
            "shader: {} cages, {} curves, {} parameters",
            program.baked.len(),
            program.curves.len(),
            program.params.len()
        );

        // The cage field the shader's texture is baked from. Built the same way GraphField
        // builds its own leaf, so any disagreement is the GPU path's and not the cage's.
        //
        // A graph can reference no cage at all — "Pipe only" does — so this follows what the
        // shader asked for rather than assuming there is one.
        let levels = fieldform_field::levels_for_tolerance(CAGE_MM * 3f64.sqrt(), tolerance_of(scene));
        let cage_fields: Vec<fieldform_field::CageField> = program
            .baked
            .iter()
            .map(|leaf| match leaf {
                fieldform_gpu::BakedLeaf::Cage(id) => {
                    fieldform_field::CageField::from_cage(doc.cage(*id).unwrap(), levels)
                        .expect("the cage builds")
                }
                // No scene here imports one, and a silent empty field would be a worse way to
                // find that out than a panic naming it.
                fieldform_gpu::BakedLeaf::Mesh(id) => panic!("no scene here uses mesh {id}"),
            })
            .collect();

        let bounds = bake_bounds();
        let baked: Vec<(&(dyn fieldform_field::ScalarField + Sync), _)> = cage_fields
            .iter()
            .map(|f| (f as &(dyn fieldform_field::ScalarField + Sync), bounds))
            .collect();
        // **Baked the way the field bakes**, at the same tolerance, because otherwise this
        // measurement would be reporting a difference in method rather than a mistake: a
        // profile answered from a grid on one side and from its boundary on the other differs
        // by the margin, everywhere, and that has nothing to do with whether the shader is
        // right. See `Profile::baked`.
        // **Zip truncates, and a truncated curve list is a silently wrong picture** rather
        // than a failure: every curve past the end simply never reaches the shader, so a body
        // that should be swept or warped renders as though nothing had been asked of it. That
        // is indistinguishable from a working scene unless you already know what it should
        // look like. Checked here so a scene that forgets to hand back a polyline says so.
        assert_eq!(
            program.curves.len(),
            curve_points.len(),
            "{}: the graph names {} curve(s) and the scene handed back {} polyline(s)",
            scene.label(),
            program.curves.len(),
            curve_points.len(),
        );
        let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
            .curves
            .iter()
            .zip(curve_points.iter())
            .map(|(use_of, points)| match use_of {
                // **An extrude's profile by its node, not by its position in the list** -- the
                // same correction the spine arm below records, and for a sharper reason. The
                // binding order is whatever codegen interned: for a *run* of extrudes it is the
                // order the union lists them, and for the same union written out it is the
                // emission order, which reaches a union's operands back to front. So no single
                // handback order is right for both arms, and the blade fan read **228 mm** off
                // with the loop switched off -- four correct blades each wearing a neighbour's
                // profile, which is precisely the mistake the fan is here to catch in the kernel.
                //
                // Derived with the same two tolerances the single-extrude scenes derive it with,
                // so those hand back exactly this and nothing about them changes.
                fieldform_gpu::CurveUse::Profile(node) => {
                    let derived = match graph.node(*node) {
                        Some(Node::Extrude { outline, .. }) => {
                            let world = doc.curves[&outline.curve()].tessellate(0.05);
                            let (origin, normal) = fieldform_field::plane_of(&world)
                                .unwrap_or((Point3::origin(), nalgebra::Vector3::z()));
                            let rotation = fieldform_field::rotation_from_z(normal);
                            let into = move |p: Point3<f64>| {
                                Point3::from(
                                    rotation.inverse_transform_vector(&(p.coords - origin.coords)),
                                )
                            };
                            fieldform_field::Profile::in_plane(&world, into, 1e-3).ok().map(
                                |flat| {
                                    flat.points()
                                        .iter()
                                        .map(|p| Point3::new(p.x, p.y, 0.0))
                                        .collect::<Vec<_>>()
                                },
                            )
                        }
                        // A sweep's section is flattened about the point the path carries it
                        // by -- the spine's start dropped onto the plane, not the centroid --
                        // and that is the compile's reading, so it is asked for here rather
                        // than fitted again.
                        Some(Node::Sweep { outline, spine, .. }) => {
                            let laid = fieldform_field::sweep_outline_section(
                                &doc,
                                outline,
                                *spine,
                                fieldform_field::PROFILE_TURN_DEGREES,
                            )
                            .expect("a sweep that compiled has a section");
                            let (origin, normal) = (laid.origin, laid.normal);
                            let rotation = fieldform_field::rotation_from_z(normal);
                            let into = move |p: Point3<f64>| {
                                Point3::from(
                                    rotation.inverse_transform_vector(&(p.coords - origin.coords)),
                                )
                            };
                            fieldform_field::Profile::in_plane(&laid.points, into, 1e-3).ok().map(
                                |flat| {
                                    flat.points()
                                        .iter()
                                        .map(|p| Point3::new(p.x, p.y, 0.0))
                                        .collect::<Vec<_>>()
                                },
                            )
                        }
                        // A revolve reads its profile in world (r, z), not flattened at all, and
                        // is one profile to a scene, so position says which.
                        _ => None,
                    };
                    let points = derived.unwrap_or_else(|| points.clone());
                    let flat: Vec<nalgebra::Point2<f64>> =
                        points.iter().map(|p| nalgebra::Point2::new(p.x, p.y)).collect();
                    // How far off the boundary the node reads, by the field's own rule. A
                    // drafted wall is a contour out from the profile rather than the profile
                    // itself, and a fillet reads its operands out to its radius -- so a bake
                    // built here to a different width would be measuring that, not the shader.
                    //
                    // **This node's own figure** where there is one to ask for. It used to be the
                    // first extrude the graph happened to hold, which is the same answer while a
                    // scene has one extrude and an arbitrary one as soon as it has four.
                    let reads = match graph.node(*node) {
                        Some(Node::Extrude { .. }) => {
                            fieldform_field::profile_reads_out_to(&doc, &graph, *node)
                        }
                        _ => graph
                            .nodes()
                            .find(|(_, n)| matches!(n, Node::Extrude { .. }))
                            .map(|(id, _)| fieldform_field::profile_reads_out_to(&doc, &graph, id))
                            .unwrap_or((0.0, 0.0)),
                    };
                    let bake = fieldform_field::Profile::new(flat).ok().and_then(|profile| {
                        let (lo, hi) = profile.bounds();
                        fieldform_field::bake_cells((hi - lo).max(), tolerance_of(scene), reads)
                            .map(|cells| profile.baked(cells, reads))
                        .and_then(|p| p.bake().cloned())
                    });
                    (points, bake)
                }
                // **A spine by its id, not by its position in the list.** The zip below pairs a
                // scene's polylines with the program's bindings positionally, and the binding
                // order is whatever codegen interned — which for a union of pipes is the order the
                // union lists them and for everything else is the emission order. The four-spoke
                // scene read **45 mm** off against the CPU until this was keyed by id: not a
                // kernel fault at all, four correct pipes with each other's spines. Every scene
                // here tessellates a spine at `TOLERANCE_MM`, so this reproduces exactly what it
                // handed back, and can no longer reproduce its neighbour's.
                //
                // At the tolerance the scene is compiled to, which for every scene that had a
                // spine before a sweep could carry a wall is `TOLERANCE_MM`: the field flattens
                // its path at the compile's tolerance, and a path flattened twice is two paths.
                fieldform_gpu::CurveUse::Spine(id) => {
                    (doc.curves[id].tessellate(tolerance_of(scene)), None)
                }
                // The miters of whatever section the node reads, from the kernel -- the same
                // call the viewport makes, so what is compared is the shader's arithmetic
                // against the field's rather than two derivations of a displacement.
                fieldform_gpu::CurveUse::Miters(node) => (
                    fieldform_field::outline_miters(
                        &doc,
                        &graph,
                        *node,
                        fieldform_field::PROFILE_TURN_DEGREES,
                    )
                    .expect("a block whose draft mitres has a section"),
                    None,
                ),
                fieldform_gpu::CurveUse::Point(_) => (points.clone(), None),
                // **A sweep's frame from its node, and from the very points its spine goes up
                // as** -- which is the arm above, so the two bindings index the same samples
                // whatever tolerance the scene is read at. A baked scene is its twin's document
                // at another tolerance, so a frame handed back by position would be the twin's.
                fieldform_gpu::CurveUse::Frame(node) => match graph.node(*node) {
                    Some(Node::Sweep { outline, spine, .. }) => {
                        let path = doc.curves[spine].tessellate(tolerance_of(scene));
                        (
                            fieldform_field::sweep_frame(&doc, outline, *spine, &path)
                                .expect("a spine with a direction"),
                            None,
                        )
                    }
                    Some(Node::Loft { .. }) => (
                        loft_of(&doc, &graph, *node)
                            .frame_normals()
                            .iter()
                            .map(|n| Point3::from(*n))
                            .collect(),
                        None,
                    ),
                    _ => (points.clone(), None),
                },
                // **A loft's centerline and sections from the loft itself**, built as the field
                // built it: the stations codegen wrote are arcs along this polyline and nothing
                // else, and a section is the boundary the field reads, projected square at its
                // own station. What is compared is the shader's interpolation against the
                // field's, over one centerline and one set of sections.
                fieldform_gpu::CurveUse::LoftSpine(node) => {
                    (loft_of(&doc, &graph, *node).spine.points().to_vec(), None)
                }
                fieldform_gpu::CurveUse::LoftSection(node, k) => (
                    loft_of(&doc, &graph, *node)
                        .section_points(*k)
                        .iter()
                        .map(|q| Point3::new(q.x, q.y, 0.0))
                        .collect(),
                    None,
                ),
                fieldform_gpu::CurveUse::LoftSlope(node, k) => (
                    loft_of(&doc, &graph, *node)
                        .section_slopes(*k)
                        .iter()
                        .map(|m| Point3::new(m.x, m.y, 0.0))
                        .collect(),
                    None,
                ),
                // **A wall's three polylines from the wall itself**, whichever block it is on. The
                // field built one to answer from; this builds it again from the same document
                // at the same fineness and hands the shader what `Wall::packed` says it reads.
                // So what is compared is the shader's arithmetic against the field's, on one
                // curve, one set of mitres and one list of bare edges — which is the whole of
                // what could differ between them.
                fieldform_gpu::CurveUse::Wall(node, part) => {
                    // Baked by the rule the field above was compiled under, at the tolerance it
                    // was compiled to, so the two read one grid or neither does -- and no wider
                    // than the field's own band, since the field this is compared against has
                    // no display reading off it.
                    let wall = fieldform_field::outline_wall_baked(
                        &doc,
                        &graph,
                        *node,
                        fieldform_field::PROFILE_TURN_DEGREES,
                        tolerance_of(scene),
                        0.0,
                    )
                    .expect("a thin extrude that compiled has a wall");
                    // A wall under a section law is walked on both sides -- see
                    // `fieldform_field::wall_bakes_under` -- so the scene that sets one is
                    // expected to have no grid, as a walked twin is.
                    let law_walks = matches!(
                        graph.node(*node),
                        Some(Node::Sweep { law, .. }) if !fieldform_field::wall_bakes_under(law)
                    );
                    assert_eq!(
                        wall.is_baked(),
                        scene.walked_twin().is_some() && !law_walks,
                        "{}: its wall is read from {}, which is not what the scene is for",
                        scene.label(),
                        if wall.is_baked() { "a grid" } else { "its curve" },
                    );
                    let grid = match part {
                        fieldform_field::WallPart::Chain => wall.bake().cloned(),
                        _ => None,
                    };
                    (wall.packed(*part), grid)
                }
            })
            .collect();
        // Through the same helper the viewport uses, so the shader tessellates the surface to
        // the depth the field did. A different depth would put the two sets of walls in
        // slightly different places and read as a shading error.
        let surfaces = fieldform_field::packed_surfaces(&doc, &program.surfaces, None)
            .expect("the scene's surfaces pack");
        let renderer = pollster::block_on(Renderer::new(
            &program,
            &baked,
            &curves,
            &surfaces,
            128,
            (bounds.1 - bounds.0).norm() * 0.15,
        ));
        let renderer = match renderer {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
                return;
            }
        };

        // Parameter values, in the order the shader asked for them.
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| match p {
                ParamRef::Literal(v) => *v as f32,
                ParamRef::Named(_) => unreachable!("this scene uses only literals"),
                // A ramp's numbers are generated constants, not buffer slots, so a driven
                // parameter never reaches the parameter list at all.
                ParamRef::Field(_) => unreachable!("a driven parameter is not a slot"),
            })
            .collect();
        renderer.set_params(&values);

        // Swept around the model rather than rendered from one angle. The artefact this is
        // chasing appears in "certain views" and comes and goes as the camera tumbles, so a
        // single viewpoint is exactly the test that would miss it.
        let bounds = model_bounds();
        let centre = Point3::from((bounds.0.coords + bounds.1.coords) * 0.5);
        let radius = (bounds.1 - bounds.0).norm() * 0.5;
        let half_h = (45f64.to_radians() * 0.5).tan();

        // `FIELDFORM_SKIP_GRID=32` marches every scene here through the viewport's skip grid
        // (plan section 5 V5), that many cells a side over the model's cube. The grid lets a ray
        // cross a cell without reading the field, on the strength of one reading at the cell's
        // centre; if any source here overestimated its distance, this is where the surface
        // would come out wrong against the CPU's.
        if let Some(cells) = std::env::var("FIELDFORM_SKIP_GRID").ok().and_then(|v| v.parse::<u32>().ok()) {
            let reach = nalgebra::Vector3::repeat(radius);
            // `FIELDFORM_SKIP_MASKS=0` fills the grid and leaves the masks unread, which is how a
            // difference is laid at the masks' door or taken from it: same shader, same grid.
            let masks = std::env::var("FIELDFORM_SKIP_MASKS").map_or(true, |v| v != "0");
            renderer.build_skip_grid(centre - reach, radius * 2.0, cells, masks);
        }

        let mut worst_angle = (0.0f64, 0.0f64, 0.0f64);
        let mut worst_place = (Point3::origin(), 0.0f64);
        for step in 0..12 {
            let yaw = step as f64 / 12.0 * std::f64::consts::TAU;
            let pitch = 0.45f64;
            let dir = nalgebra::Vector3::new(
                yaw.cos() * pitch.cos(),
                yaw.sin() * pitch.cos(),
                pitch.sin(),
            );
            let eye = centre + dir * (radius / half_h) * 1.05;
            let camera = Camera::looking_at(
                eye,
                centre,
                nalgebra::Vector3::z(),
                (centre, radius),
                45.0,
                width as f64 / height as f64,
            );
            let hits = renderer
                .render(&camera, width, height, 512)
                .expect("the render succeeds");
            let (worst, hit_count, at, signed) = residual(&cpu, &camera, &hits, width, height);
            if worst > worst_angle.1 {
                worst_angle = (yaw.to_degrees(), worst, hit_count as f64);
                worst_place = (at, signed);
            }
        }
        println!(
            "  swept 12 views: worst surface error {:.4} mm at yaw {:.0} deg ({} hits)",
            worst_angle.1, worst_angle.0, worst_angle.2 as usize
        );
        measured.push((
            scene.label(),
            worst_angle.1,
            worst_angle.2 as usize,
            failure_budget_of(scene),
        ));
        if worst_angle.1 > 0.05 {
            // Where, and which side. A positive reading means the shader stopped short of the
            // surface -- it believed there was something there and the field says there is not.
            println!(
                "    worst at ({:.2}, {:.2}, {:.2}), the CPU field reads {:+.4} mm there",
                worst_place.0.x, worst_place.0.y, worst_place.0.z, worst_place.1
            );
        }

        let mut camera = Camera::framing(model_bounds(), width as f64 / height as f64);
        // A sweep, so the hit threshold's effect on the reported error is measured rather than
        // assumed. `FIELDFORM_EPSILON_SWEEP=1` prints the whole curve for each scene.
        if std::env::var("FIELDFORM_EPSILON_SWEEP").is_ok() {
            let base = camera.epsilon;
            println!("=== {} ===", scene.label());
            // Budget alone first, at the threshold the app actually uses: if rays are running
            // out of steps rather than failing to converge, this moves and the epsilon sweep
            // below does not explain it.
            // The cross that decides the setting: a hundredfold finer threshold, on the step
            // budget the viewport actually uses. If it needs more steps, the budget has to
            // move with it; if it does not, only one number changes.
            camera.epsilon = base * 0.01;
            for budget in [512u32, 1024, 2048] {
                let started = std::time::Instant::now();
                let hits = renderer
                    .render(&camera, width, height, budget)
                    .expect("the render succeeds");
                let ms = started.elapsed().as_secs_f64() * 1e3;
                let (worst, n, _, _) = residual(&cpu, &camera, &hits, width, height);
                println!(
                    "  epsilon/100 at {budget:>5} steps: worst {worst:>8.4} mm, {ms:>6.1} ms, {n} hits"
                );
            }
            camera.epsilon = base;
            {
                let started = std::time::Instant::now();
                let hits = renderer
                    .render(&camera, width, height, 512)
                    .expect("the render succeeds");
                let ms = started.elapsed().as_secs_f64() * 1e3;
                let (worst, n, _, _) = residual(&cpu, &camera, &hits, width, height);
                println!(
                    "  stock epsilon at   512 steps: worst {worst:>8.4} mm, {ms:>6.1} ms, {n} hits"
                );
            }
            for factor in [1.0, 0.25, 0.0625, 0.015625] {
                camera.epsilon = base * factor;
                // More steps as the threshold tightens: a ray that has to get closer needs
                // more chances to, and a budget held fixed would report the budget instead.
                let budget = (512.0 / factor.sqrt()) as u32;
                let started = std::time::Instant::now();
                let hits = renderer
                    .render(&camera, width, height, budget)
                    .expect("the render succeeds");
                let ms = started.elapsed().as_secs_f64() * 1e3;
                let (worst, n, _, _) = residual(&cpu, &camera, &hits, width, height);
                println!(
                    "  epsilon {:>9.5} mm ({budget:>5} steps): worst {worst:>8.4} mm,                      {ms:>7.1} ms at {width}x{height}, {n} hits",
                    camera.epsilon,
                );
            }
            camera.epsilon = base;
            continue;
        }
        let hits = renderer
            .render(&camera, width, height, 512)
            .expect("the render succeeds");

        report(&cpu, &camera, &hits, width, height, scene.label());
    }

    verdict(&measured);
}

/// What the run found, and whether it passes — the part that was missing.
///
/// Prints every scene's worst beside the budget it was held to, so the table that should set
/// those budgets is the run's own output rather than something to be gathered by hand. Then
/// exits: `1` if any scene disagreed, `2` if there was nothing to check at all.
///
/// **Nothing to check is a failure, not a pass.** A `FIELDFORM_SCENE` filter that matches no
/// label, or a scene whose views return no hits, both produce a clean-looking run that measured
/// nothing — and a gate that passes when it did not run is worse than no gate, because it
/// reports the absence of evidence as evidence of absence.
fn verdict(measured: &[(&str, f64, usize, f64)]) {
    println!("\n=== worst surface error by scene ===");
    println!("{:<44} {:>10} {:>10}  {}", "scene", "worst mm", "budget", "hits");
    for (label, worst, hits, budget) in measured {
        let mark = if hits == &0 {
            " NO HITS"
        } else if worst > budget {
            " OVER"
        } else {
            ""
        };
        println!("{label:<44} {worst:>10.4} {budget:>10.2}  {hits:>7}{mark}");
    }

    if measured.is_empty() {
        eprintln!(
            "\nNothing ran. FIELDFORM_SCENE matched no scene label, so this checked nothing."
        );
        std::process::exit(2);
    }

    let blind: Vec<&str> = measured.iter().filter(|m| m.2 == 0).map(|m| m.0).collect();
    let over: Vec<&(&str, f64, usize, f64)> =
        measured.iter().filter(|m| m.2 > 0 && m.1 > m.3).collect();

    if blind.is_empty() && over.is_empty() {
        println!(
            "\n{} scene(s) agree with the CPU field, each inside its budget.",
            measured.len()
        );
        return;
    }
    for label in &blind {
        eprintln!("{label}: twelve views and not one hit — this scene measured nothing.");
    }
    for (label, worst, _, budget) in &over {
        eprintln!("{label}: worst {worst:.4} mm, over its budget of {budget:.2} mm.");
    }
    eprintln!(
        "\nThe shader and the CPU field disagree on {} scene(s), and {} measured nothing.",
        over.len(),
        blind.len(),
    );
    std::process::exit(1);
}

#[derive(Copy, Clone)]
enum Scene {
    /// A cage with no generator applied — the simplest graph there is, and the one that
    /// crashed the app.
    ///
    /// It compiles to a shader that never calls `pipe_distance` and reads no parameters, so a
    /// bind group layout *derived* from shader usage loses three of its six entries while the
    /// binding code still supplies all six. wgpu treats that mismatch as fatal. Every scene
    /// here had a pipe in it, so nothing exercised the case until the app did.
    CageOnly,
    /// A pipe cut through a cage — exercises a boolean and the negation inside it.
    Difference,
    /// A pipe with no cage at all.
    ///
    /// The mirror image of `CageOnly`, and the other end of the same bind-group problem: a
    /// shader that reads a curve and never touches the cage atlas. Every other scene here has
    /// a cage in it, which is exactly the shape of gap that has hidden four bugs already.
    PipeOnly,
    /// A cube with a sphere cut out of it, and no cage or curve anywhere.
    ///
    /// The closed-form leaves: exact everywhere, and the only nodes whose whole definition is
    /// numbers — so this is also the scene that checks the centre going through the parameter
    /// buffer in the same slot order on both passes, value and normal.
    Primitives,
    /// A block with three spheres cut from it in **one** subtraction, and four solids in one
    /// union — the list boolean, which folds as a statement per operand on both paths.
    ///
    /// The awkward configuration rather than the tidy one: the cutters overlap each other and
    /// two of them break the block's faces, so the fold has to pick the right operand's
    /// complement where several cutters compete, and the union's spheres overlap in pairs so
    /// the nearest changes hands mid-surface.
    ManyWayDifference,
    ManyWayUnion,
    /// A cube infilled with a lattice — one scene per cell, because the three surfaces are
    /// three separate hand-transcriptions into WGSL and a wrong sign in any one of them makes
    /// a pattern that still looks like a lattice.
    ///
    /// Also the case the whole field representation exists for: the meshed equivalent is
    /// billions of triangles, and this traces per pixel with nothing stored.
    GyroidInfill,
    SchwarzPInfill,
    SchwarzDInfill,
    /// A gyroid laid over an **open surface** rather than through a box — the conformal map.
    ///
    /// **The scene the whole storage-buffer path exists to be checked by.** Every other lattice
    /// here is closed-form arithmetic on both sides; this one asks the shader to walk a
    /// hierarchy over a tessellated mesh in its own storage buffers and come back with the same
    /// closest point the field found. `query_packed` already checks the walk against the field
    /// in Rust, so what is left for this to catch is the transcription into WGSL — a swapped
    /// offset, a stride read as the wrong type, a layout the two sides number differently.
    ///
    /// Domed rather than flat, and for a reason a flat panel would hide entirely: a facet normal
    /// and the surface's own agree exactly on a plane and nowhere else, and the conformal map
    /// carries that normal straight into its gradient. The first cut of this path used facet
    /// normals in the shader and matched the CPU perfectly on a panel.
    ///
    /// **Domed gently, and that is a measured choice rather than a timid one.** A closest-point
    /// map is multi-valued on the medial axis of its own offset — a point there is equidistant
    /// from two different places on the surface, and which one an implementation returns is
    /// decided by rounding. `fieldform-field/examples/conformal_agreement.rs` measures it with
    /// the GPU removed entirely, in `f64`, reading the shader's own arrays:
    ///
    /// ```text
    /// rise/span   worst s,t   worst mm   split
    ///    0.0000    3.33e-16     0.00e0   0.000%
    ///    0.0625     1.39e-3    1.68e-6   0.212%
    ///    0.1250      1.00e0    3.37e-6   0.263%
    ///    0.3125      1.00e0    6.76e-6   1.578%
    /// ```
    ///
    /// `s` and `t` reach **1.0 apart** — as far as they can be — while the offset still agrees
    /// to seven decimals: the two answers are the same distance away in different directions,
    /// which is what an ambiguity looks like and what an error does not. So a steeply domed
    /// scene would measure the medial axis rather than the transcription, and report a number
    /// that no amount of correct code could improve. This one stays at `rise/span = 0.0625`,
    /// where the map is single-valued over the slab, and measures what it is for.
    ///
    /// The limit itself is real and belongs in front of users: a conformal lattice is only
    /// well defined within its surface's own curvature radius, the way a cylindrical one is only
    /// a lattice outside its degenerate radius. The panel does not say so yet.
    ConformalInfill,
    /// The same lattice confined to a shell of its surface, and **nothing else in the graph**.
    ///
    /// Two things only this scene exercises. The shader takes an early exit when a point is
    /// further outside the surface's box than the shell reaches — answering from the box alone
    /// without walking a triangle — and the field has no such shortcut, so a disagreement here
    /// is that shortcut being wrong. And the band's face is a surface the cells do not describe:
    /// its normal is the surface's own, which is a branch in `lattice_normal` that the
    /// unconfined arm never reaches.
    ///
    /// No confining body, which is the point of it: a shelled conformal lattice is bounded, so
    /// it is a body on its own. Every other lattice scene here has to be intersected with one.
    ///
    /// **It read 8.35 mm for five days and that was not a disagreement.** Both sides carried the
    /// same discontinuity: past the panel's rim the signed distance flips sign across the
    /// surface's extension while still millimetres from the edge, the lattice's third argument
    /// jumped with it, and the marcher stepped across. `conformal-cell-map-spec.md` section 9.2
    /// has the measurement and the fix; the scene reads 0.24 mm now, mean 0.010. It also used to
    /// leave its confining slab in the graph as a stray body, which is why its image was blank.
    /// What had been ruled out was ruled out correctly -- the arithmetic *was* identical -- and
    /// the fact that pointed at the answer was the worst point itself, 14 mm beyond the edge.
    ConformalShell,
    /// A *sphere* infilled with a gyroid, which is the shape a curved boundary makes.
    ///
    /// Kept separate from the cube because the two ask different questions. A cube's faces cut
    /// the lattice along flat planes; a sphere cuts it along a curved one, so the seam between
    /// the two surfaces bends in every direction at once. If the seams a user reported are
    /// geometry rather than shading, this is the scene whose error metric says so.
    SphereInfill,
    /// A cube infilled with a lattice whose phase's own number follows a ramp up through it --
    /// one scene per phase, since the three are three expressions on each side: a wall
    /// converted into an isovalue, a density looked up in a table, and a radius taken off a
    /// distance. The shader writes the ramp's expression where the sizing slot was read and
    /// divides by a constant the field computed; a slot miscounted, or a divisor computed apart,
    /// shows here and nowhere else.
    GradedWall,
    GradedDensity,
    GradedStrut,
    /// A cube infilled with a gyroid whose wall sits off the cell's surface -- a walled TPMS.
    /// The shift is one more argument to `lattice_distance` and one more slot, so this is the
    /// scene that says the ninth slot is read where it was written, on both passes.
    WalledInfill,
    /// A cube cored with a honeycomb: hexagons in the ground plane, straight up through it. The
    /// one cell with no cell coordinates -- it is measured in millimetres on both sides, from
    /// two transcriptions of the same fold into one sextant -- and the one whose walls are exact.
    HoneycombInfill,
    /// A tube cored with a Voronoi foam, and one with its edge beams -- both in the awkward
    /// configuration `voronoi-lattice-spec.md` §4.7 names, and awkward in three ways at once.
    ///
    /// **Cylindrical**, so the hash index has to wrap on the angular axis or the pattern tears
    /// down the seam -- and the wrap is arithmetic neither side can borrow from the other, since
    /// the field takes `rem_euclid` and WGSL's `%` truncates toward zero. **Off-axis**, so the
    /// frequency varies across the sample and the two-step bound is exercised rather than
    /// collapsing to a constant. And at **jitter 1**, which is the only place the `R = 2`
    /// fallback fires at all (0.04% of probes, `voronoi_profile`), so the wider ring is taken on
    /// both sides or on neither.
    ///
    /// The integer hash is the part that would fail silently: a mismatch shows up as a
    /// completely different arrangement of cells, which is a large residual rather than a subtle
    /// one -- but `a_hash_is_the_same_number_wherever_it_is_computed` is what should catch that
    /// first, and this is what catches everything built on top of it.
    VoronoiFoamInfill,
    VoronoiBeamInfill,
    /// A closed profile given thickness, and one swept about an axis.
    ///
    /// The first blocks whose shape comes from a *curve* rather than from numbers, and the only
    /// ones whose polyline the shader is handed already flattened into the block's frame. If
    /// the two sides ever disagree about what that flattening means, this is where it shows.
    Extruded,
    /// The same profile swept at an angle instead of straight out.
    ///
    /// A lean is a shear, and a shear is the one case where the two sides do not run the same
    /// arithmetic: the shader takes the corner between the profile's distance and the slab's
    /// when the sweep is straight, and the correction factor otherwise, while the CPU never
    /// takes the corner at all. That is a deliberate difference, so it needs a measurement
    /// rather than an argument.
    Leaning,
    /// The same profile again, standing in the YZ plane instead of lying on the ground.
    ///
    /// **Every other profile scene lies flat**, where the plane fitted to it is the identity and
    /// a world direction and a local one are the same three numbers. That is precisely the case
    /// that cannot catch a rotation applied the wrong way round, or not at all — and the
    /// application draws on whatever plane the person picked.
    Upright,
    /// The same square profile again, at a tolerance coarse enough that it answers from a grid.
    ///
    /// **The only scene that exercises the bake.** Everything else here is compiled at a quarter
    /// of a millimetre, where a profile keeps its exact boundary — so without this the shader's
    /// grid sampling would have no check on it at all, and a mis-indexed row would show as a
    /// wrong shape nobody was looking at.
    BakedProfile,
    /// The same square profile, extruded with its walls leaning out.
    ///
    /// A draft moves every wall by an amount that depends on how far along the sweep the query
    /// is, and the reading is then divided by the secant of the angle to stay a lower bound.
    /// Two closed-form corrections that both sides have to make identically, and neither shows
    /// in the shape if only one of them does it — the zero set is untouched by a factor.
    Drafted,
    /// The same draft, answered from a **baked grid** rather than an exact profile.
    ///
    /// Drafted and baked were each covered and the pair was not, which is the combination the
    /// application actually runs: a display tolerance coarse enough to bake, and walls that
    /// lean. It matters because a draft moves the *visible surface off the profile's own zero
    /// set* -- at this depth and angle, several millimetres off it -- and a narrow band stores
    /// only what is near zero. So the shader reads the coarse level where nothing else here
    /// ever reads it.
    DraftedBake,
    /// The same draft again, on a **negative depth**: the flip.
    ///
    /// A flip is a minus sign on the height, and the draft has to follow it so the walls still
    /// open away from the plane rather than closing into it. Both sides do that with a
    /// `signum`, from separately written expressions, and a sign is exactly the sort of thing
    /// that comes out right in one of them.
    DraftedFlip,
    /// Half each way about the profile's plane, which folds the sweep rather than running it
    /// out of one side -- the one thing in the extrude that the shader and the field each do
    /// for themselves, so the one thing a fold with a wrong sign could hide in.
    Symmetric,
    /// The same, drafted, so the profile is the widest section and the walls open both ways.
    SymmetricDrafted,
    /// **Section 2.3 of `extrude-revolve-options-spec.md`**: two sides of different depths,
    /// drafted differently, begun a tenth of the cage off the plane, along a typed lean. Every
    /// one of the six depth slots doing something, and the lean on top, so the shader's
    /// `extrude_over` is held to the kernel's `plane_and_slab` on every arm at once.
    TwoSidedBegun,
    /// **`round-chamfer-spec.md` §4's rims**, each a scene above with its rims set and nothing
    /// else changed (`Scene::rimmed`): the drafted square rounded at G1, where the section's
    /// gradient leans out of square with the slab's; the two-sided, begun, leaning one rounded
    /// at G2, every depth slot and the lean at once; and the thin extrude's wall chamfered, the
    /// `wall_extrude_rim_distance` arm. The rim goes through `rim_join` on the GPU and
    /// `Rim::join` on the CPU, and these are what say the two are one join.
    RimmedDrafted,
    RimmedTwoSided,
    ChamferRimmedWall,
    /// The four unlike sections lofted at G1, their rims rounded at G1: the loft's rim through
    /// `loft_rim_value_normal`, on a loft whose section changes along the path -- the case the
    /// spec's argument needed a divisor for and the blend's bound does not.
    RimmedLoft,
    /// The box minus the sphere with every edge of the box rounded at G1: the inner box grown by
    /// a ball, a true distance, so the shader's n-norm arm is held against the exact one.
    RoundedBoxCut,
    /// The same at G2, the cube norm: a bound, its edges the G2 blend's curve.
    G2BoxCut,
    /// The same chamfered: the box intersected with a bevel across each edge.
    ChamferedBoxCut,
    /// Two cylinders, one tilted and chamfered, one upright at G2 -- the meridian reading, and a
    /// frame that is not the world's.
    EdgedCylinders,
    /// A rounded frustum and a chamfered pointed cone, one tilted: the meridian polygon's two
    /// exact cuts, with an apex cut flat.
    EdgedCones,
    /// The same two at G2: the family's join at each corner, a bound.
    G2Cones,
    /// `round-chamfer-spec.md` §6's own scene: a two-sided drafted extrude with its rims rounded,
    /// blended at G2 to a chamfered box.
    RimmedBlendedToChamferedBox,
    /// The two-sided leaning extrude with side one's face alone rounded at G2: a rim on one end,
    /// which joins the rounded end's own slab and takes the other hard (`rim_join_ends`), on the
    /// solid where the two ends are furthest from alike.
    EndRimmedTwoSided,
    /// The loft with its first section's rim alone rounded.
    StartRimmedLoft,
    /// The bent sweep with the end of its path alone chamfered, through the in-tube reading.
    EndRimmedSweep,
    /// The tube section turned two hundred degrees with both caps' rims rounded at G2: past a
    /// half turn, so the outside reading changes cap along the way round.
    RimmedWedge,
    /// The thin revolve turned two ways with the start cap's rim alone chamfered.
    StartRimmedThinWedge,
    /// The square swept along the bent path, its rims rounded at G2: the in-tube reading,
    /// where the section is read as it is and only the caps are divided, with the rim's join
    /// in both the divided and the own reading (`sweep_in_tube_rim`).
    RimmedSweep,
    /// The same sweep scaled to a third on the way, its rims chamfered: `sweep_law_rim_over`,
    /// the section shrinking along the path as the rim cuts its ends.
    ChamferRimmedTaper,
    Revolved,
    /// A wall to the left of an S-shaped curve that ends in a corner sharper than a right
    /// angle, square-ended, pulled out of a plane that lines up with nothing.
    ///
    /// **The awkward case, not the convenient one.** A wall's field has four things in it that
    /// a profile's does not, and each needs a place to show: the fans on the outside of the
    /// S's two bends, which are on opposite sides; the mitre, where the corner turns toward
    /// the wall past a right angle and each band is cut by the other run's line; the two
    /// square ends; and "left", which is only left once the plane's normal has been settled.
    /// The chain is two curves meeting end to end, so the walk that joins them is in the path
    /// too, and the plane is tilted so that none of it is an identity.
    ThinExtruded,
    /// The same chain with the wall to its right and round ends, pulled out at a lean and
    /// drafted — so the wall's reading goes through the shear and the slope a thick extrude's
    /// does, and the round ends and the outside of the sharp corner are both in the picture.
    ThinLeaning,
    /// A wall about a closed loop, taken both ways about its plane: a tube with a
    /// re-entrant corner. The one case where the wall is read through the loop's own signed
    /// distance, walked, and folded about the middle of the band.
    ThinTube,
    /// A straight line given a thickness and round ends, and pulled sideways. A straight line
    /// has no plane of its own, so this is the scene that says the field and the shader chose
    /// the same one — the plane the stated sweep leaves most squarely.
    ThinRib,
    /// A block with that first wall cut out of it, so the slot's faces are the wall's own
    /// surface met from the *inside*.
    ///
    /// **The only scene that marches through a wall.** Everything above reads one from
    /// outside, where the feet of the perpendiculars are the whole answer. A subtracted wall
    /// is stepped through by its depth, which is measured to the bare edges found when the
    /// wall was built and to every fan's round — a different stretch of the shader, and one
    /// that an overstated depth would show as a ray stepping out through the slot's wall.
    ThinCut,
    /// A square profile carried along a curve that bends in two planes.
    ///
    /// The one body with three bindings -- profile, spine, frame -- and the scene where a frame
    /// error and a mapping error both have somewhere to show: the section has to turn out of
    /// the plane it was drawn in and keep turning, and a normal read from the wrong sample, or a
    /// section read in the wrong pair of axes, puts the surface visibly elsewhere.
    Swept,
    /// The same sweep with its section started flush with the profile's plane, and with it held
    /// parallel the whole way.
    ///
    /// **The two held sections, where they differ most.** Both slide the point along the path
    /// onto the section that contains it rather than projecting it across the nearest one, and
    /// both shorten the reading by how far the section leans. A shader that skips either is a
    /// solid in a different place from the one the kernel meshes, and this scene is the only
    /// thing that would say so: the profile's plane is square to nothing here.
    SweptFlush,
    SweptParallel,
    /// **The section law, which the shader reads through six slots of its own.** Scaled down
    /// the whole way, drafted outward, and turned through a right angle over a window that is
    /// neither end of the path -- the three things `SweepLaw` does, on the sweep's own spine so
    /// that what differs from `Swept` above is the law and nothing else.
    ///
    /// Here because a law is arithmetic written twice, once in `Swept::section_at` and once in
    /// `sweep_law_on`, and a difference between them is a solid in two places. The drafted one
    /// also exercises the corner the slab is not allowed to take, which cost `SkewExtrusion`
    /// eight millimetres when it was taken anyway.
    /// **A drafted extrude whose corners are mitred rather than rounded.** The section is the
    /// L, so five convex corners and one reflex, and the draft is steep enough that the mitre
    /// stands well clear of the arc it replaces.
    ///
    /// Here because the mitre is arithmetic written twice -- `Profile::offset_by` and
    /// `profile_mitred` -- over a boundary each side displaces for itself, and a difference
    /// between them is a solid in two places. It also walks rather than reading a grid, which
    /// is the one thing a mitred draft cannot do.
    SharpDraft,
    /// The same on a wall round a closed loop, drafted — with its corners **rounded**, which
    /// is the only thing a wall's draft does (mitring a band's two faces is refused by name,
    /// `CompileError::SharpCornersOnAWall`).
    ///
    /// **This is the scene whose absence let a real bug ship.** The shader moved the band's
    /// middle by the draft where the kernel grows the band about it, so a drafted tube's wall
    /// slid outward instead of thickening — 6.9 mm apart down the bore, and invisible from
    /// outside because the outer face lands in the same place either way. 87 scenes agreed
    /// with it in: `SweptDrafted` carries a closed profile, and the thin sweeps carry a scale,
    /// so no scene drafted a wall along a path until this one.
    DraftedWallSweep,
    SweptScaled,
    SweptDrafted,
    SweptTwistedWindow,
    /// **Four dissimilar sections along a cubic bending in two planes, G1** -- `loft-spec.md`
    /// §9's scene: a square, a circle, a rounded rectangle and a weighted quadratic ellipse,
    /// each centred on the path and drawn flat, so every one is tilted to the path and
    /// projected square at its own station. Everything the shader reads -- the centerline,
    /// its frame, the four projected sections and the stations in `params` -- comes from the
    /// one build the field compiled from, so what this measures is the two interpolations
    /// against each other: the span search, the Hermite with its one-sided end slopes, the
    /// slab and the bound's divisor.
    Lofted,
    /// The same loft at G0, so the linear branch of `loft_on` is measured and not only parsed:
    /// one slot differs between the two, which is also the toggle ruled to refill a buffer.
    LoftedG0,
    /// **Two offset rings and no centerline** -- the spec's S-spine case (§9), and two things
    /// the scene above never reaches: a centerline that exists nowhere but in the loft's own
    /// build, uploaded as `CurveUse::LoftSpine`, and a loft of exactly two sections, where the
    /// span loop runs no times and both end slopes are one-sided.
    LoftedDerived,
    /// The same law on an open section, where a wall is scaled whole rather than keeping its
    /// thickness -- the one place the two kinds of wall part company under a taper.
    ThinSweptScaled,
    /// **Four pipes on their own spokes, in one union** — the shape `pipe_run_distance` walks as a
    /// loop instead of writing out a call an operand.
    ///
    /// Here because a loop that walks a run wrongly still compiles and simply puts material
    /// somewhere else, which is exactly what item 34 said of the pattern's window when the same
    /// change was made one level down. Nothing else in this list forms a run: `ManyWayUnion` is
    /// primitives, and every other pipe here is on its own.
    ///
    /// Four rather than two, so the run is longer than `RUN_FLOOR` and an off-by-one in
    /// `first + k` lands on a spine that exists and is the wrong one — which is the mistake a
    /// two-pipe scene could not tell from a correct one.
    SpokedPipes,
    /// **Four blades on their own spokes, in one union** — the shape `extrude_run_distance`
    /// walks, and the other kind of run.
    ///
    /// Here for the same reason the pipes are, and against a second mistake they could not catch:
    /// an extrude's loop reads twelve slots a member rather than one, so an off-by-one in the
    /// *slot* arithmetic is a blade with somebody else's height, draft or plane — all of which
    /// still compile and simply put material elsewhere.
    ///
    /// **Each blade a different size, which is what makes four of them worth anything.** A fan of
    /// identical blades evenly spaced is symmetric under a shift of one, so an off-by-one in
    /// `first + k` would turn the fan by a step and produce *the same union*: a wrong loop nothing
    /// could see. Different lengths, widths and heights break that symmetry.
    SpokedBlades,
    /// **Four thin blades on their own spokes, in one union** — the shape
    /// `wall_extrude_run_distance` walks, and the third kind of run.
    ///
    /// A wall's loop steps by three bindings and sixteen slots where a blade's steps by one and
    /// twelve, so there are two strides to get wrong and four more slots to read off by one.
    /// **So no two of these are alike in anything the loop reads**: each has its own thickness
    /// and height, three have a kink in them and the wall to a different side of it, two have
    /// round ends, one bends the other way so its plane faces down and its rotation is not the
    /// others', one is taken both ways about its plane, and one is a closed loop. A member
    /// wearing its neighbour's chain, edges or numbers is then a different solid.
    SpokedRibs,
    /// The first thin scene again, at a tolerance coarse enough that the wall answers from a
    /// grid of its own distance.
    ///
    /// **Four scenes, because a baked wall is read by a different stretch of the shader and
    /// from a different buffer**, and each of the four reaches it another way. This one is the
    /// plain case: an open one-sided wall with a mitre in it, seen from outside.
    ThinBaked,
    /// The loop, baked. The scene that would catch the grid being read *and then folded* — a
    /// wall's bake holds the band's distance already, and the loop's branch in the shader
    /// folds a curve's. Asked in the wrong order the tube comes out twice as thick or not at
    /// all.
    ThinTubeBaked,
    /// The cut, baked: the wall's grid read from inside the material, where the block's slot
    /// is the wall's own surface and the fine band has to be there on both sides of it.
    ThinCutBaked,
    /// The four thin blades, each from a grid: a run whose members each carry a bake on the
    /// first of their three bindings, so the loop's `first + 3k` has to land on a span with a
    /// grid every time and on the right one.
    SpokedRibsBaked,
    /// **A curve given a thickness and turned about an axis**: the S with its sharp corner, to
    /// its left, as a shell of revolution. The wall is laid in the half-plane about the axis
    /// rather than in a plane of its own, so this is the scene that says the shader and the
    /// field read it at the same `(r, z)` and call the same side left.
    ThinRevolved,
    /// A line close by the axis with more wall toward it than there is room for, so the band
    /// reaches across the axis and comes back as radii on this side. **The one scene that
    /// takes the second reading**, at `(-r, z)`: without it the middle of this solid is a hole.
    ThinRevolvedAcross,
    /// **Section 3.3 of `extrude-revolve-options-spec.md`**: the thin revolve's S, square-ended
    /// and to the left, turned 135 degrees two ways -- 90 one way and 45 the other -- so both
    /// caps, the wedge's inside and the reading past a cap are all in the picture.
    ThinRevolvedWedge,
    /// The rectangle a thick revolve turns into a tube, turned 200 degrees one way: a wedge
    /// past a half turn, where which cap is nearer changes along the way round.
    RevolvedWedge,
    /// **The spec's acceptance case** (`outline-wall-spec.md` section 7): a square-ended wall
    /// to the left of an S with one sharp corner, carried along a spine that bends in two
    /// planes. An open section, which a sweep could not take at all, with every awkward thing
    /// a wall has in it and a frame that has to turn it out of the plane it was drawn in.
    ThinSwept,
    /// Each of those three from a grid of the wall's own distance.
    ThinRevolvedBaked,
    ThinRevolvedAcrossBaked,
    ThinSweptBaked,
    /// The four round solids at once, one of them rotated by a placement.
    ///
    /// None of them carries an axis: a cylinder points along Z and is turned by a
    /// `Transform`. That decision is only sound if a rotated primitive shades and traces the
    /// same on both sides, so the capsule here is placed at an angle no axis happens to line
    /// up with — the CPU applies the inverse rotation in `f64` and the shader does it in
    /// `f32` from a different expression, and a mismatch in either would show up as the hit
    /// point missing the surface.
    RoundSolids,
    /// Two skewed boxes and a sphere: one box leaning every way at once and blended into the
    /// sphere, the other **left-handed** and placed by a turn and a scale about a pivot that is
    /// not the origin. Twelve slots in each pass, counted separately, so a slot dropped from one
    /// shows here; the left-handed box is the one whose determinant is negative, which is the sign
    /// the shader divides by; and the blend asks for the box's value *beside* its edges, where a
    /// box that was only right on its faces would shape the fillet wrongly.
    /// `skewed_box_probe` holds the arithmetic itself at points a ray rarely reaches.
    SkewedBoxes,
    /// Two round solids in the same place, averaged half way. The one node whose shader arm is
    /// written twice — value and normal — so a slot counted in one pass and not the other shows
    /// here and nowhere else.
    Morphed,
    /// The same, with the amount driven by a ramp. The divisor is a generated constant here and
    /// a field on the op there, so this is the scene that catches the two computing it apart.
    MorphedDriven,
    /// A sphere into a block, filleted by a radius that follows a ramp up Z
    /// (`round-chamfer-spec.md` §9.1). The radius is the ramp's own expression in the shader and
    /// an op the field reads; the divisor is `driven_blend_divisor` on both sides, and this is
    /// the scene that would show the two parting.
    BlendDriven,
    /// The same sphere into the same block, chamfered with two legs -- further along the block
    /// than the sphere -- and an upright extrude whose rims are a two-legged chamfer
    /// (`round-chamfer-spec.md` §9.2): the operands scaled before the join on both sides, and
    /// the rim's second leg riding in the vector's exponent slot.
    TwoLegChamfers,
    /// The two-sided extrude with rims that are a two-legged chamfer, by the rimmed-twin table.
    TwoLegRims,
    /// A cage and a pipe as two bodies, unjoined — what building a pipe leaves before any
    /// boolean is applied, and the only scene where the union of bodies is what is drawn
    /// rather than a boolean somebody asked for.
    TwoBodies,
    /// A pipe blended into a cage — exercises the smooth minimum, where the two
    /// implementations of `smooth_min` have to agree exactly or the seam moves.
    Blend,
    /// A pipe cut out of a cage with the cut filleted.
    ///
    /// The shader turns a smooth union into this by negating the operand *and* its gradient,
    /// in two separate places. A sign dropped from one of them leaves the silhouette right and
    /// the shading inverted, which the union scene cannot catch because it negates neither.
    BlendedDifference,
    /// The cage and the pipe blended at **G2**, which is the same expression with the exponent
    /// 3 instead of 2 — passed to both `smooth_min`s as a literal, so a path that assumed two
    /// anywhere (the closed form `blend_bound` has for the circle, the `n != 2` branch that
    /// replaces it) shows here as a seam in the wrong place. The cubic is also the first
    /// exponent whose gradient weights are not the operands' own values.
    G2Blend,
    /// A pipe cut out of a cage with a G2 fillet: the sign handling of `BlendedDifference` at the
    /// other exponent, where the gradient weights `x²` and `y²` are not `x` and `y`.
    G2BlendedDifference,
    /// The cage and the pipe joined by a **chamfer** -- the same expression at exponent one,
    /// with a closed-form bound of its own on both sides (`blend_bound`'s `n == 1` arm) and a
    /// normal that is the two operands' halves across the bevel. Exponent one is also where the
    /// shader's `pow(x, n − 1)` would be `pow(0, 0)` on a face, which WGSL leaves undefined, so a
    /// NaN normal there would show here as a black seam.
    ChamferBlend,
    /// A pipe cut out of a cage with a chamfered edge: the sign handling at exponent one.
    ChamferedDifference,
    /// A `Pattern` node repeating one body six times around Z.
    ///
    /// The case point contexts were built for, at the scale that matters: one node reached
    /// through six placements, so it is evaluated at six points. But the reason it needs a
    /// scene of its own now is that the two paths reach this shape by *different routes* — the
    /// CPU folds a `min` over pre-inverted isometries, while the shader lowers the node into
    /// transforms and unions before it sees it. Nothing but a residual would catch the two
    /// disagreeing, and a rotation about a point is exactly where they could: the CPU composes
    /// translate–rotate–translate, and the lowering leans on `Transform` turning about its own
    /// translation to say the same thing in one node.
    Patterned,
    /// The same node stepping a body along a line instead of around an axis.
    ///
    /// Worth separating because the linear arm is where the shader does arithmetic the CPU does
    /// not: the step is *multiplied out per instance* at generation time, since a shader has no
    /// loop to multiply in. An off-by-one in that multiple stacks the copies one step from
    /// where the CPU puts them — uniform, plausible, and visible only as a residual.
    PatternedRow,
    /// A small body ringed a long way from its axis, which is where the window is narrowest.
    ///
    /// Reported as "patterned a sphere 24 times around x=125 and it stopped looking like
    /// spheres -- more like their intersection; increasing X breaks it the same way". Both the
    /// count and the radius matter: the further out the ring, the fewer cells one body spans,
    /// so `pattern_reach` returns a smaller window and a fold that looks at too few neighbours
    /// clips each copy to its own wedge. The existing radial scene is six copies of a body
    /// close in, which is the opposite end of that.
    PatternedRing,
    /// A row of overlapping copies rounded into each other rather than merely unioned.
    ///
    /// **The n-ary smooth minimum, which the two paths accumulate separately.** The field runs
    /// a sum and a minimum over the window in `f64`; the shader runs the same two in `f32` in a
    /// generated loop. Nothing else in this sweep folds more than two operands through a blend,
    /// so nothing else would catch one of them weighting the copies differently — and a seam in
    /// the wrong place is exactly the sort of thing that still draws a confident picture.
    PatternedFillet,
    /// The same row of copies rounded at G2: the n-ary fold with the cubic's weights, on the
    /// path that sums a window of instances rather than folding two operands.
    G2PatternedFillet,
    /// A row whose step is its whole run, centred on the body, with two copies left out and an
    /// oblique step -- the three options that change where a copy of a row *is*, in a scene whose
    /// every copy is asymmetric, so a copy a step out of place shows. The shader is handed the gap
    /// already divided by `n - 1` and the steps to copy 0 as a bias; either one wrong puts the row
    /// somewhere plausible and wrong.
    PatternedRun,
    /// A ring that stops short of a turn about an axis off the origin, centred, with copies left
    /// out. Over an arc a point behind the ring belongs to the nearer end rather than wrapping
    /// round, which is the one thing a full turn never asks of the fold.
    PatternedArc,
    /// The same arc with its copies pitched closer than their diameter and rounded into each
    /// other, one skipped: the rounded sum has to leave a skipped copy out as well as the minimum.
    PatternedArcFillet,
    /// **Forty small copies, so the window is a few cells wide and not the whole pattern.** The
    /// scenes above have a handful of large bodies, which makes the loop walk every copy -- and a
    /// walk of every copy never asks the fold which copy a point belongs to, so it cannot see a
    /// fold that answers wrongly. This one is a centred row spaced by its whole run, where the
    /// answer depends on the origin and the bias both.
    PatternedLongRun,
    /// The same over an arc, where a point behind the ring has to belong to the nearer end: the one
    /// place the shader's cell differs from a full turn's, and a ray crossing the gap steps by what
    /// it is told, so a wrong end would let it through a copy.
    PatternedLongArc,
    /// **A grid at two oblique steps**, centred, with copies skipped, small enough that both
    /// directions are windowed. The nearest copy to a point past an edge is found by sliding along
    /// the edge, not by clamping each coordinate, and the second step leaves the first's plane so
    /// that the two coordinates are read off dual vectors and not off components.
    PatternedGrid,
    /// A long thin grid: three columns by thirty-six rows, so the columns are covered by their window
    /// and walked whole while the rows are walked by window — the two kinds of loop in one grid.
    PatternedLongGrid,
    /// A twisted bar: the first domain warp, and the first field whose value is *divided*.
    ///
    /// Both paths compute the divisor from the body's own reach from the axis, and they have to
    /// agree exactly — a shader dividing by a different number puts the surface in the same
    /// place while scaling the field around it, so the tracer stops short on one path and not
    /// the other. Nothing but a residual against the CPU would catch that.
    Twisted,
    /// A bar drawn toward the same curve the pipe scenes sweep.
    ///
    /// The second domain warp, and the one whose divisor is a *parameter* rather than a
    /// constant folded into the shader — so this is the scene that would catch the two passes
    /// counting parameter slots out of step, which shifts every slot after it and is silent
    /// because the shader still compiles and simply reads the wrong numbers.
    ///
    /// Aimed at the *bent* curve on purpose. A straight one would agree between the two paths
    /// even if the shader's closest-point search kept the wrong segment, because on a straight
    /// spine every segment gives the same answer; a bend is where picking the nearer one
    /// matters, and where a near-tie between two segments has to break the same way on both.
    Attracted,
    /// A body unioned with its own reflection, which is what Modify > Mirror produces.
    ///
    /// Worth its own scene because a mirror goes through the *point assignment* machinery — the
    /// same path a placement uses to change where its subtree is evaluated — and that code now
    /// has two kinds to tell apart in four places. A mirror emitted where a transform was
    /// expected would put the reflection somewhere plausible and shade it from the wrong side,
    /// which only a residual against the CPU would catch.
    Mirrored,
    /// The same reflection, about a **plane block that has been turned** rather than about a
    /// typed axis.
    ///
    /// The scene above aims its mirror with `Frame::At`, which is the arm that always worked,
    /// and that is why nothing here caught this: the shader resolved a frame by hand instead of
    /// asking the kernel, and had drifted. A plane's own turn was dropped; and since a block
    /// carrying a turn *is* that block inside a `Node::Transform` by the time either compiler
    /// sees it, a turned plane did not even reach the arm that dropped it — it fell through to
    /// one that returned the world origin facing `+Z`. So the reflection landed on the far side
    /// of a plane nobody had drawn.
    ///
    /// Turned about two axes and moved off the origin, so a shader that lost either half of the
    /// frame is measurably wrong rather than accidentally right. Measured, on the shader as it
    /// stood: **283.3 mm** of surface error on a 1000 mm scene, and normals 32.5 degrees out on
    /// average. Afterwards 0.0133 mm, which is this example's own floor.
    MirroredAboutAPlane,
    /// A cylindrical lattice inside a tube, which is what the cell map exists for.
    ///
    /// The scene that exercises the second map end to end. Two things could go wrong on the GPU
    /// alone and nowhere else: the coordinate change happens per pixel in `f32` where the CPU
    /// does it in `f64`, and the wall thickness varies with radius, so a shader that hoisted
    /// the isovalue would draw walls that thin toward the axis while the CPU's stay even. Both
    /// show up as a residual against the CPU field and nothing else would catch them.
    CylindricalInfill,
    /// The same tube with a spherical map, which is the second map the split promised.
    SphericalInfill,
    /// The same tube filled with the *solid* phase rather than a sheet.
    ///
    /// **Deliberately not a rectangular block**, because the two things the solid phase does
    /// differently both hide under one. Its isovalue does not vary with the local frequency,
    /// where a sheet's does — under a rectangular map the frequency is constant, so a shader
    /// that wrongly scaled the isovalue would agree with the CPU anyway. And its normal must
    /// not carry the sheet's sign flip, which shows as a residual only where the surface is
    /// curved enough for the two to point measurably apart.
    ///
    /// So: cylindrical, off the axis, against a bore — the configuration in which a mistake in
    /// either has somewhere to show.
    SolidInfill,
    /// The three cells added after the first family, each in a tube under a cylindrical map.
    ///
    /// **The awkward configuration on purpose, and not the cube the first three got.** Each cell
    /// is three separate hand-transcriptions into WGSL — the value, the gradient, and the bound
    /// — and a wrong sign in any one still draws something that looks like a lattice. A curved
    /// map is what gives a mistake somewhere to show: the coordinate change happens per pixel in
    /// `f32` against the CPU's `f64`, the isovalue varies with radius, and the gradient has to be
    /// carried back out of the map's own basis, which is where a mis-signed partial stops
    /// cancelling and starts bending the surface.
    SchoenIwpInfill,
    NeoviusInfill,
    FischerKochSInfill,
    /// The beam family, in the same tube under the same cylindrical map.
    ///
    /// **The arithmetic here is a different shape from every scene above**, which is why it needs
    /// scenes of its own rather than trusting the surfaces' agreement to carry over: there is no
    /// isovalue, no banding and no gradient bound to divide by. The value is a distance to a
    /// strut set in cell space, converted into millimetres by the largest local frequency, and
    /// the normal is a central difference rather than an analytic gradient. A curved map is what
    /// makes the conversion non-trivial, so this is where a mistake in it would show.
    CubicBeamInfill,
    BccBeamInfill,
    FccBeamInfill,
    OctetBeamInfill,
    /// Diamond, which has a shader arm of its own.
    ///
    /// **The only cell that does not fold**, so it reaches its answer by a different reduction —
    /// wrap and sort the coordinates — over a 24-segment table transcribed into WGSL, against six
    /// segments for the two above. That table is the largest hand-copied thing in the prelude and
    /// this scene is what checks it: one wrong strut moves the surface, and the residual says so.
    DiamondBeamInfill,
    /// A body whose offset varies through space, driven by a ramp off a plane.
    ///
    /// The first field-driven parameter, and the scene it needs because the two paths compute
    /// it differently in three places at once: the ramp is a *generated constant* in the
    /// shader and a resolved number on the CPU, the easing is a WGSL branch against a float
    /// code where the CPU matches on an enum, and both divide by a bound that neither derives
    /// on its own — they share `driven_offset_divisor` precisely so they cannot disagree.
    ///
    /// A smootherstep ramp on purpose: it is the steepest of the three, so a divisor computed
    /// from the wrong shape factor shows here and would not on a linear one.
    VaryingOffset,
    /// A filleted body with a bore marked critical, so every fillet tapers to nothing at it
    /// (`blend-integrity-spec.md` §6).
    ///
    /// **The body's own blends sit under a placement and the bore does not.** The gap a fillet
    /// tapers by is pointwise in *world* space, and a blend under a rotation only ever holds a
    /// local point — so a shader that worked the gap out from the point it had would taper
    /// about a bore in the wrong place, and the residual here is what would say so. The
    /// tapering pixels' normals are differenced rather than composed, which the normal column
    /// checks against the CPU's gradient.
    ProtectedBore,
    /// A hollowed cube: the cube minus an inward offset of itself.
    ///
    /// The scene the offset node needs, and it asks the one question that matters on the GPU
    /// side. The shader subtracts a constant from the value and passes the *gradient straight
    /// through* — dropping the subtraction from the gradient is correct here and wrong
    /// everywhere it looks similar, so a mistake shows as a wall of the right thickness shaded
    /// as if it were somewhere else. The two inner faces are also where a cavity's normal
    /// points opposite the outer one, which the difference arm has to negate.
    Hollow,
}

impl Scene {
    fn label(self) -> &'static str {
        match self {
            Scene::CageOnly => "cage alone",
            Scene::ConformalInfill => "a gyroid laid over an open domed surface",
            Scene::ConformalShell => "the same gyroid, confined to a shell of it",
            Scene::Difference => "cage minus pipe",
            Scene::PipeOnly => "pipe alone",
            Scene::TwoBodies => "cage and pipe, unjoined",
            Scene::Primitives => "box minus sphere",
            Scene::RoundedBoxCut => "box minus sphere, the box's edges rounded at G1",
            Scene::G2BoxCut => "box minus sphere, the box's edges at G2",
            Scene::ChamferedBoxCut => "box minus sphere, the box's edges chamfered",
            Scene::EdgedCylinders => "a chamfered tilted cylinder and a G2 upright one",
            Scene::EdgedCones => "a rounded frustum and a chamfered pointed cone",
            Scene::G2Cones => "a frustum and a pointed cone, their edges at G2",
            Scene::RimmedBlendedToChamferedBox => "a rimmed two-sided extrude blended G2 to a chamfered box",
            Scene::ManyWayDifference => "a block minus three cutters, in one subtraction",
            Scene::ManyWayUnion => "four solids in one union",
            Scene::RoundSolids => "round solids, sliced by a half-space",
            Scene::SkewedBoxes => "two skewed boxes and a sphere, one box blended and one turned",
            Scene::Morphed => "a sphere and a cuboid, morphed half way",
            Scene::MorphedDriven => "the same, with the amount driven by a ramp",
            Scene::BlendDriven => "a sphere filleted into a block, the radius driven by a ramp up Z",
            Scene::TwoLegChamfers => "a sphere chamfered into a block with two legs",
            Scene::TwoLegRims => "the two-sided extrude, its rims a chamfer of two legs",
            Scene::GyroidInfill => "cube infilled with a gyroid",
            Scene::SchwarzPInfill => "cube infilled with a Schwarz P",
            Scene::SchwarzDInfill => "cube infilled with a Schwarz D",
            Scene::SphereInfill => "sphere infilled with a gyroid",
            Scene::WalledInfill => "cube infilled with a gyroid whose wall sits off its surface",
            Scene::HoneycombInfill => "cube cored with a honeycomb",
            Scene::VoronoiFoamInfill => "tube cored with a Voronoi foam, fully jittered",
            Scene::VoronoiBeamInfill => "tube cored with Voronoi edge beams, fully jittered",
            Scene::GradedWall => "cube infilled with a gyroid whose wall follows a ramp",
            Scene::GradedDensity => "cube filled with a Schwarz P whose density follows a ramp",
            Scene::GradedStrut => "cube infilled with BCC beams whose radius follows a ramp",
            Scene::Blend => "cage blended with pipe",
            Scene::BlendedDifference => "cage minus pipe filleted",
            Scene::G2Blend => "cage blended with pipe, G2",
            Scene::G2BlendedDifference => "cage minus pipe, G2 fillet",
            Scene::ChamferBlend => "cage chamfered to pipe",
            Scene::ChamferedDifference => "cage minus pipe, chamfered edge",
            Scene::Hollow => "cube shelled to a wall",
            Scene::CylindricalInfill => "tube infilled around its axis",
            Scene::SphericalInfill => "tube infilled about a centre",
            Scene::SolidInfill => "tube filled with a solid gyroid",
            Scene::SchoenIwpInfill => "tube infilled with a Schoen IWP",
            Scene::NeoviusInfill => "tube infilled with a Neovius",
            Scene::FischerKochSInfill => "tube infilled with a Fischer-Koch S",
            Scene::CubicBeamInfill => "tube infilled with cubic beams",
            Scene::BccBeamInfill => "tube infilled with BCC beams",
            Scene::FccBeamInfill => "tube infilled with FCC beams",
            Scene::OctetBeamInfill => "tube infilled with an octet truss",
            Scene::DiamondBeamInfill => "tube infilled with diamond beams",
            Scene::Mirrored => "a body beside its own reflection",
            Scene::MirroredAboutAPlane => "a body reflected in a plane block that has been turned",
            Scene::Twisted => "a bar wound about its axis",
            Scene::Attracted => "a bar drawn toward a curve",
            Scene::Patterned => "one body patterned six times around Z",
            Scene::PatternedRow => "one body patterned five times along X",
            Scene::PatternedRing => "a sphere ringed 24 times far from its axis, copies only",
            Scene::PatternedFillet => "a row of copies rounded into each other",
            Scene::G2PatternedFillet => "a row of copies rounded into each other, G2",
            Scene::PatternedRun => "a row spaced by its whole run, centred, with copies skipped",
            Scene::PatternedArc => "a ring over part of a turn, with copies skipped",
            Scene::PatternedArcFillet => "an arc of copies rounded into each other, skipping one",
            Scene::PatternedLongRun => "forty small copies along a centred whole run, three skipped",
            Scene::PatternedLongArc => "forty small copies round an arc, one skipped",
            Scene::PatternedGrid => "a centred grid of small copies at two oblique steps, some skipped",
            Scene::PatternedLongGrid => "a long thin grid: one direction walked whole, the other by window",
            Scene::VaryingOffset => "a body whose offset varies with height",
            Scene::ProtectedBore => "a filleted body whose bore is held critical",
            Scene::Extruded => "a square profile extruded",
            Scene::Leaning => "a square profile extruded at a lean",
            Scene::Upright => "a profile standing in the YZ plane, extruded",
            Scene::BakedProfile => "a profile extruded, answered from a baked grid",
            Scene::Drafted => "a square profile extruded with a draft",
            Scene::DraftedBake => "a drafted extrude answered from a baked grid",
            Scene::DraftedFlip => "a drafted extrude flipped to the other side",
            Scene::Symmetric => "an extrude taken both ways about its plane",
            Scene::SymmetricDrafted => "the same, drafted, so it opens both ways",
            Scene::TwoSidedBegun => "an extrude of two sides, drafted apart, begun off its plane, leaning",
            Scene::RimmedDrafted => "the drafted square, its rims rounded at G1",
            Scene::RimmedTwoSided => "the two-sided leaning extrude, its rims rounded at G2",
            Scene::ChamferRimmedWall => "a thin extrude's wall, its rims chamfered",
            Scene::RimmedLoft => "the four-section loft, its rims rounded",
            Scene::EndRimmedTwoSided => "the two-sided leaning extrude, its end's rim alone rounded at G2",
            Scene::StartRimmedLoft => "the four-section loft, its start's rim alone rounded",
            Scene::EndRimmedSweep => "the bent sweep, its end's rim alone chamfered",
            Scene::RimmedWedge => "the tube section turned 200 degrees, its rims rounded at G2",
            Scene::StartRimmedThinWedge => "the thin revolve turned two ways, its start's rim alone chamfered",
            Scene::RimmedSweep => "the bent sweep, its rims rounded at G2",
            Scene::ChamferRimmedTaper => "the tapering sweep, its rims chamfered",
            Scene::Revolved => "a rectangle revolved into a tube",
            Scene::ThinExtruded => "a thin extrude to the left of an S with a sharp corner",
            Scene::ThinLeaning => "a thin extrude to the right, round-ended, leaning and drafted",
            Scene::ThinTube => "a thin extrude about a closed loop, both ways",
            Scene::ThinRib => "a thin extrude of a straight line, swept sideways",
            Scene::ThinCut => "a thin extrude cut out of a block",
            Scene::Swept => "a square profile swept along a curve bending in two planes",
            Scene::SweptFlush => "the same sweep, started flush with its profile's plane",
            Scene::SweptParallel => "the same sweep, its section held parallel throughout",
            Scene::SharpDraft => "an L drafted twelve degrees, its corners mitred",
            Scene::DraftedWallSweep => "a wall round a loop, drafted and carried along a path",
            Scene::SweptScaled => "the same sweep, its section scaled to a third on the way",
            Scene::SweptDrafted => "the same sweep, every wall drafted two degrees out",
            Scene::Lofted => "four unlike sections lofted smoothly along a path bending in two planes",
            Scene::LoftedG0 => "the same loft, straight from each section to the next",
            Scene::LoftedDerived => "two offset rings lofted along a centerline of the loft's own",
            Scene::SweptTwistedWindow => {
                "the same sweep, turned a right angle over the middle of its path"
            }
            Scene::ThinSweptScaled => "a thin open sweep whose section scales to a half",
            Scene::SpokedPipes => "four pipes on spokes, in one union",
            Scene::SpokedBlades => "four blades on spokes, in one union",
            Scene::SpokedRibs => "four thin blades on spokes, in one union",
            Scene::ThinBaked => "a thin extrude answered from a baked grid",
            Scene::ThinTubeBaked => "a thin extrude about a loop, answered from a baked grid",
            Scene::ThinCutBaked => "a thin extrude from a baked grid, cut out of a block",
            Scene::SpokedRibsBaked => "four thin blades in one union, each from a baked grid",
            Scene::ThinRevolved => "a thin revolve of an S with a sharp corner",
            Scene::ThinRevolvedAcross => "a thin revolve whose band crosses its axis",
            Scene::ThinRevolvedWedge => "a thin revolve turned 135 degrees two ways",
            Scene::RevolvedWedge => "a rectangle revolved 200 degrees into a tube section",
            Scene::ThinSwept => "a thin open S swept along a curve bending in two planes",
            Scene::ThinRevolvedBaked => "the thin revolve, answered from a baked grid",
            Scene::ThinRevolvedAcrossBaked => "the thin revolve across its axis, from a baked grid",
            Scene::ThinSweptBaked => "the thin open sweep, answered from a baked grid",
        }
    }

    /// The cell this scene infills with, if it is an infill scene.
    fn lattice_cell(self) -> Option<fieldform_scene::LatticeCell> {
        match self {
            Scene::GyroidInfill => Some(fieldform_scene::LatticeCell::Gyroid),
            Scene::SchwarzPInfill => Some(fieldform_scene::LatticeCell::SchwarzP),
            Scene::SchwarzDInfill => Some(fieldform_scene::LatticeCell::SchwarzD),
            Scene::SphereInfill => Some(fieldform_scene::LatticeCell::Gyroid),
            Scene::WalledInfill => Some(fieldform_scene::LatticeCell::Gyroid),
            Scene::HoneycombInfill => Some(fieldform_scene::LatticeCell::Honeycomb),
            _ => None,
        }
    }

    /// The scene this one is the baked reading of, if it is one.
    ///
    /// A baked thin scene is the same document as its twin and differs only in the tolerance
    /// it is compiled and uploaded at, which is what decides whether a wall answers from a
    /// grid. So it builds nothing of its own: one geometry, read two ways, is what makes the
    /// pair a comparison.
    fn walked_twin(self) -> Option<Scene> {
        match self {
            Scene::ThinBaked => Some(Scene::ThinExtruded),
            Scene::ThinTubeBaked => Some(Scene::ThinTube),
            Scene::ThinCutBaked => Some(Scene::ThinCut),
            Scene::SpokedRibsBaked => Some(Scene::SpokedRibs),
            Scene::ThinRevolvedBaked => Some(Scene::ThinRevolved),
            Scene::ThinRevolvedAcrossBaked => Some(Scene::ThinRevolvedAcross),
            Scene::ThinSweptBaked => Some(Scene::ThinSwept),
            _ => None,
        }
    }

    /// The scene a rimmed one is, and the rims it sets on every extrude in it -- a fifth of the
    /// cage's tenth for the square's, a third of the wall's thickness for the chamfer.
    fn rimmed(self) -> Option<(Scene, fieldform_scene::Rims)> {
        let at = |reach: f64, continuity: Continuity, at: fieldform_scene::RimEnds| fieldform_scene::Rims {
            radius: ParamRef::Literal(reach),
            continuity,
            at,
            second: None,
        };
        let rims = |reach: f64, continuity: Continuity| at(reach, continuity, fieldform_scene::RimEnds::Both);
        use fieldform_scene::RimEnds::{End, Start};
        match self {
            Scene::EndRimmedTwoSided => Some((Scene::TwoSidedBegun, at(0.05 * CAGE_MM, Continuity::G2, End))),
            // A two-legged chamfer's rims: twice as far in from the side as back from the end.
            Scene::TwoLegRims => Some((
                Scene::TwoSidedBegun,
                fieldform_scene::Rims {
                    second: Some(ParamRef::Literal(0.1 * CAGE_MM)),
                    ..rims(0.05 * CAGE_MM, Continuity::Chamfer)
                },
            )),
            Scene::StartRimmedLoft => Some((Scene::Lofted, at(0.02 * CAGE_MM, Continuity::G1, Start))),
            Scene::EndRimmedSweep => Some((Scene::Swept, at(0.02 * CAGE_MM, Continuity::Chamfer, End))),
            Scene::RimmedWedge => Some((Scene::RevolvedWedge, rims(0.02 * CAGE_MM, Continuity::G2))),
            Scene::StartRimmedThinWedge => Some((Scene::ThinRevolvedWedge, at(0.004 * CAGE_MM, Continuity::Chamfer, Start))),
            Scene::RimmedDrafted => Some((Scene::Drafted, rims(0.06 * CAGE_MM, Continuity::G1))),
            Scene::RimmedTwoSided => Some((Scene::TwoSidedBegun, rims(0.05 * CAGE_MM, Continuity::G2))),
            Scene::ChamferRimmedWall => Some((Scene::ThinExtruded, rims(0.006 * CAGE_MM, Continuity::Chamfer))),
            Scene::RimmedLoft => Some((Scene::Lofted, rims(0.02 * CAGE_MM, Continuity::G1))),
            Scene::RimmedSweep => Some((Scene::Swept, rims(0.02 * CAGE_MM, Continuity::G2))),
            Scene::ChamferRimmedTaper => Some((Scene::SweptScaled, rims(0.015 * CAGE_MM, Continuity::Chamfer))),
            _ => None,
        }
    }

    fn build(self) -> (Document, Graph, Vec<Vec<Point3<f64>>>) {
        if let Some(twin) = self.walked_twin() {
            return twin.build();
        }
        {
            let edge = |reach: f64, continuity: Continuity| fieldform_scene::Fillet {
                radius: ParamRef::Literal(reach * CAGE_MM),
                continuity,
                second: None,
            };
            let cut = match self {
                Scene::RoundedBoxCut => Some(edge(0.08, Continuity::G1)),
                Scene::G2BoxCut => Some(edge(0.08, Continuity::G2)),
                Scene::ChamferedBoxCut => Some(edge(0.06, Continuity::Chamfer)),
                _ => None,
            };
            if let Some(cut) = cut {
                let (doc, mut graph, curves) = Scene::Primitives.build();
                let boxes: Vec<NodeId> = graph
                    .nodes()
                    .filter(|(_, n)| matches!(n, Node::Cuboid { .. }))
                    .map(|(id, _)| id)
                    .collect();
                for id in boxes {
                    if let Some(Node::Cuboid { edge, .. }) = graph.node_mut(id) {
                        *edge = cut;
                    }
                }
                return (doc, graph, curves);
            }
            if matches!(self, Scene::EdgedCones | Scene::G2Cones) {
                let mut graph = Graph::empty();
                let at = |x: f64, y: f64, z: f64| {
                    fieldform_scene::Place::At([
                        ParamRef::Literal(x * CAGE_MM),
                        ParamRef::Literal(y * CAGE_MM),
                        ParamRef::Literal(z * CAGE_MM),
                    ])
                };
                let g2 = matches!(self, Scene::G2Cones);
                let frustum = graph.add(Node::Cone {
                    ends: fieldform_scene::Ends::between(at(-0.3, 0.0, -0.35), at(-0.3, 0.0, 0.25)),
                    base_radius: ParamRef::Literal(0.25 * CAGE_MM),
                    top_radius: ParamRef::Literal(0.12 * CAGE_MM),
                    edge: edge(0.05, if g2 { Continuity::G2 } else { Continuity::G1 }),
                });
                let pointed = graph.add(Node::Cone {
                    ends: fieldform_scene::Ends::between(at(0.25, -0.1, -0.35), at(0.4, 0.15, 0.4)),
                    base_radius: ParamRef::Literal(0.22 * CAGE_MM),
                    top_radius: ParamRef::Literal(0.0),
                    edge: edge(0.06, if g2 { Continuity::G2 } else { Continuity::Chamfer }),
                });
                graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![frustum.into(), pointed.into()] });
                return (Document::new(), graph, Vec::new());
            }
            if matches!(self, Scene::EdgedCylinders) {
                let mut graph = Graph::empty();
                let at = |x: f64, y: f64, z: f64| {
                    fieldform_scene::Place::At([
                        ParamRef::Literal(x * CAGE_MM),
                        ParamRef::Literal(y * CAGE_MM),
                        ParamRef::Literal(z * CAGE_MM),
                    ])
                };
                let tilted = graph.add(Node::Cylinder {
                    ends: fieldform_scene::Ends::between(at(-0.45, -0.1, -0.3), at(-0.1, 0.15, 0.35)),
                    radius: ParamRef::Literal(0.18 * CAGE_MM),
                    edge: edge(0.05, Continuity::Chamfer),
                });
                let upright = graph.add(Node::Cylinder {
                    ends: fieldform_scene::Ends::between(at(0.3, 0.0, -0.35), at(0.3, 0.0, 0.3)),
                    radius: ParamRef::Literal(0.2 * CAGE_MM),
                    edge: edge(0.07, Continuity::G2),
                });
                graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![tilted.into(), upright.into()] });
                return (Document::new(), graph, Vec::new());
            }
            if matches!(self, Scene::RimmedBlendedToChamferedBox) {
                let (doc, mut graph, curves) = Scene::RimmedTwoSided.build();
                let extrude = graph
                    .nodes()
                    .find(|(_, n)| matches!(n, Node::Extrude { .. }))
                    .map(|(id, _)| id)
                    .expect("the two-sided extrude");
                // Across the extrude at the plane it starts from, sticking out either side, so
                // the blend has a join to round all the way round the box's waist.
                let block = graph.add(Node::Cuboid {
                    centre: fieldform_scene::Place::At([
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.1 * CAGE_MM),
                    ]),
                    size: [
                        // Longer than the 0.6-wide square, so it stands out of both sides.
                        ParamRef::Literal(1.0 * CAGE_MM),
                        ParamRef::Literal(0.1 * CAGE_MM),
                        ParamRef::Literal(0.12 * CAGE_MM),
                    ],
                    turn: fieldform_scene::no_turn(),
                    edge: edge(0.02, Continuity::Chamfer),
                });
                graph.add(Node::Blend {
                    op: BooleanOp::Union,
                    a: extrude.into(),
                    b: block.into(),
                    fillet: edge(0.03, Continuity::G2),
                });
                return (doc, graph, curves);
            }
        }
        if let Some((base, set)) = self.rimmed() {
            let (doc, mut graph, curves) = base.build();
            let solids: Vec<NodeId> = graph
                .nodes()
                .filter(|(_, node)| matches!(node, Node::Extrude { .. } | Node::Loft { .. } | Node::Sweep { .. } | Node::Revolve { .. }))
                .map(|(id, _)| id)
                .collect();
            assert!(!solids.is_empty(), "{}: a rimmed scene rims a profile solid", self.label());
            for id in solids {
                match graph.node_mut(id) {
                    Some(Node::Extrude { rims, .. } | Node::Loft { rims, .. } | Node::Sweep { rims, .. } | Node::Revolve { rims, .. }) => {
                        *rims = set
                    }
                    _ => {}
                }
            }
            return (doc, graph, curves);
        }
        let mut doc = Document::new();
        let cage_id = CageId::new();
        doc.cages.insert(cage_id, Cage::cube_of(CAGE_MM).into());

        let curve_id = CurveId::new();
        let s = CAGE_MM;
        let curve = Curve::polyline(
            curve_id,
            "spine",
            vec![
                Point3::new(-s, -0.18 * s, -0.10 * s),
                Point3::new(-0.3 * s, -0.05 * s, 0.02 * s),
                Point3::new(0.3 * s, 0.10 * s, 0.05 * s),
                Point3::new(s, 0.22 * s, 0.12 * s),
            ],
            false,
        )
        .expect("a valid polyline");
        let spine = curve.tessellate(TOLERANCE_MM);
        doc.curves.insert(curve_id, curve);

        // A pipe on its own is a graph with no cage node in it at all, not a graph whose cage
        // is merely unused: an unconsumed node is a body now and would be drawn.
        if matches!(self, Scene::ManyWayDifference | Scene::ManyWayUnion) {
            let mut graph = Graph::empty();
            let at = |x: f64, y: f64, z: f64| {
                [
                    ParamRef::Literal(x * CAGE_MM),
                    ParamRef::Literal(y * CAGE_MM),
                    ParamRef::Literal(z * CAGE_MM),
                ]
            };
            if matches!(self, Scene::ManyWayDifference) {
                let block = graph.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At(at(0.0, 0.0, 0.0)),
                    size: [
                        ParamRef::Literal(CAGE_MM),
                        ParamRef::Literal(CAGE_MM * 0.6),
                        ParamRef::Literal(CAGE_MM * 0.5),
                    ],
                    turn: fieldform_scene::no_turn(),
                });
                // Two cutters breaking faces, one overlapping both and buried.
                let cutters = [
                    graph.add(Node::Sphere { centre: fieldform_scene::Place::At(at(0.35, 0.0, 0.2)), radius: ParamRef::Literal(0.22 * CAGE_MM) }),
                    graph.add(Node::Sphere { centre: fieldform_scene::Place::At(at(-0.3, 0.15, -0.1)), radius: ParamRef::Literal(0.25 * CAGE_MM) }),
                    graph.add(Node::Sphere { centre: fieldform_scene::Place::At(at(0.05, -0.05, 0.0)), radius: ParamRef::Literal(0.18 * CAGE_MM) }),
                ];
                graph.add(Node::Boolean {
                    op: BooleanOp::Difference,
                    inputs: vec![block.into(), cutters[0].into(), cutters[1].into(), cutters[2].into()],
                });
            } else {
                let solids = [
                    graph.add(Node::Sphere { centre: fieldform_scene::Place::At(at(-0.3, 0.0, 0.0)), radius: ParamRef::Literal(0.22 * CAGE_MM) }),
                    graph.add(Node::Sphere { centre: fieldform_scene::Place::At(at(-0.05, 0.1, 0.05)), radius: ParamRef::Literal(0.2 * CAGE_MM) }),
                    graph.add(Node::Cuboid { edge: Default::default(),
                        centre: fieldform_scene::Place::At(at(0.25, -0.05, 0.0)),
                        size: [
                            ParamRef::Literal(0.4 * CAGE_MM),
                            ParamRef::Literal(0.3 * CAGE_MM),
                            ParamRef::Literal(0.35 * CAGE_MM),
                        ],
                        turn: fieldform_scene::no_turn(),
                    }),
                    graph.add(Node::Sphere { centre: fieldform_scene::Place::At(at(0.4, 0.15, 0.1)), radius: ParamRef::Literal(0.16 * CAGE_MM) }),
                ];
                graph.add(Node::Boolean { op: BooleanOp::Union, inputs: solids.iter().copied().map(Into::into).collect() });
            }
            return (doc, graph, Vec::new());
        }

        // **A run**: four pipes on their own spokes, each read by the union alone, which is what
        // `looped_pipe_run` admits. Its own branch, as the many-way booleans have, because a pipe
        // on its own is a graph with no cage node in it rather than one whose cage goes unused.
        if matches!(self, Scene::SpokedPipes) {
            let mut graph = Graph::empty();
            let mut points = Vec::new();
            let mut pipes = Vec::new();
            for i in 0..4 {
                let turn = std::f64::consts::TAU * f64::from(i) / 4.0;
                let (c, sn) = (turn.cos(), turn.sin());
                let id = CurveId::new();
                let at = |r: f64, z: f64| {
                    Point3::new(r * c * CAGE_MM, r * sn * CAGE_MM, z * CAGE_MM)
                };
                // Bent, and each one bent differently, so a run that read the wrong spine puts
                // a pipe somewhere a correct one never goes.
                let spoke = Curve::polyline(
                    id,
                    "spoke",
                    vec![
                        at(0.08, 0.02 * f64::from(i)),
                        at(0.45, 0.14 - 0.06 * f64::from(i)),
                        at(0.92, -0.05 + 0.04 * f64::from(i)),
                    ],
                    false,
                )
                .expect("a bent line");
                points.push(spoke.tessellate(TOLERANCE_MM));
                doc.curves.insert(id, spoke);
                pipes.push(graph.add(Node::Pipe {
                    curve: id,
                    // Each radius different, so a run reading the wrong slot is a wrong thickness
                    // rather than the same picture.
                    radius: ParamRef::Literal((0.07 + 0.015 * f64::from(i)) * CAGE_MM),
                }));
            }
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: pipes.into_iter().map(Into::into).collect() });
            return (doc, graph, points);
        }

        // **A run of the other kind**: four blades on their own spokes, extruded in Z and
        // unioned, which is what `extrude_run_distance` walks. Its own branch for the same reason
        // the pipes' is — a profile is a graph with no cage node in it.
        if matches!(self, Scene::SpokedBlades) {
            let mut graph = Graph::empty();
            let mut flats = Vec::new();
            let mut ids = Vec::new();
            for i in 0..4 {
                let k = f64::from(i);
                let turn = std::f64::consts::TAU * k / 4.0;
                let (c, sn) = (turn.cos(), turn.sin());
                let (inner, outer) = (0.10 * CAGE_MM, (0.50 + 0.09 * k) * CAGE_MM);
                let half = (0.05 + 0.012 * k) * CAGE_MM;
                // Along the spoke by `r`, across it by `w`, all in the ground plane — so each
                // blade's frame is its own centroid and its normal is +Z.
                let at = |r: f64, w: f64| Point3::new(r * c - w * sn, r * sn + w * c, 0.0);
                let id = CurveId::new();
                let blade = Curve::polyline(
                    id,
                    "blade",
                    vec![at(inner, -half), at(outer, -half), at(outer, half), at(inner, half)],
                    true,
                )
                .expect("a rectangle");
                // Pre-flattened into the blade's own frame, as every profile here is handed back.
                // The harness pairs these with the bindings by position, and a run's bindings are
                // in the order the union lists its operands — which is the order of this loop.
                let world = blade.tessellate(TOLERANCE_MM);
                let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                let rotation = fieldform_field::rotation_from_z(normal);
                let into = move |p: Point3<f64>| {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                };
                let flat =
                    fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM).expect("planar");
                flats.push(
                    flat.points()
                        .iter()
                        .map(|p| Point3::new(p.x, p.y, 0.0))
                        .collect::<Vec<_>>(),
                );
                doc.curves.insert(id, blade);
                ids.push(graph.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(id),
                    sweep: fieldform_scene::Sweep::Normal,
                    // Each a different height, so a wrong slot is a blade of somebody else's
                    // depth rather than the same picture.
                    height: ParamRef::Literal((0.25 + 0.07 * k) * CAGE_MM),
                    start: ParamRef::Literal(0.0),
                    sides: fieldform_scene::Sides::One,
                    draft: ParamRef::Literal(0.0),
                    corners: Default::default(),
                }));
            }
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.into_iter().map(Into::into).collect() });
            return (doc, graph, flats);
        }

        // **A run of the third kind**: four thin blades, which is what
        // `wall_extrude_run_distance` walks. No two alike — see `Scene::SpokedRibs` for why.
        if matches!(self, Scene::SpokedRibs) {
            use fieldform_scene::{Outline, WallEnds, WallSide};
            let mut graph = Graph::empty();
            let mut ids = Vec::new();
            let each = [
                (WallSide::Centre, WallEnds::Square),
                (WallSide::Left, WallEnds::Round),
                (WallSide::Right, WallEnds::Round),
                (WallSide::Left, WallEnds::Square),
            ];
            for (i, (side, ends)) in each.into_iter().enumerate() {
                let k = i as f64;
                let turn = std::f64::consts::TAU * k / 4.0;
                let (c, sn) = (turn.cos(), turn.sin());
                let (inner, outer) = (0.12 * CAGE_MM, (0.55 + 0.08 * k) * CAGE_MM);
                // Along the spoke by `r`, across it by `w`, in the ground plane.
                let at = |r: f64, w: f64| Point3::new(r * c - w * sn, r * sn + w * c, 0.0);
                let id = CurveId::new();
                let rib = if i == 3 {
                    // A loop, so one member of the run is read through the closed branch.
                    let half = 0.07 * CAGE_MM;
                    Curve::polyline(
                        id,
                        "rib",
                        vec![at(inner, -half), at(outer, -half), at(outer, half), at(inner, half)],
                        true,
                    )
                } else {
                    // A kink part-way out, to the right of the spoke — except the second, which
                    // kinks to the left, so its chain winds the other way and its plane faces
                    // down: the one member whose rotation is not the identity.
                    let kink = (0.06 + 0.02 * k) * CAGE_MM * if i == 1 { 1.0 } else { -1.0 };
                    Curve::polyline(
                        id,
                        "rib",
                        vec![at(inner, 0.0), at(0.5 * (inner + outer), kink), at(outer, 0.0)],
                        false,
                    )
                }
                .expect("a polyline");
                doc.curves.insert(id, rib);
                ids.push(graph.add(Node::Extrude { rims: Default::default(),
                    outline: Outline::Wall {
                        curve: id,
                        thickness: ParamRef::Literal((0.04 + 0.012 * k) * CAGE_MM),
                        side,
                        ends,
                    },
                    sweep: fieldform_scene::Sweep::Normal,
                    height: ParamRef::Literal((0.22 + 0.06 * k) * CAGE_MM),
                    start: ParamRef::Literal(0.0),
                    sides: fieldform_scene::Sides::symmetric_if(i == 2),
                    draft: ParamRef::Literal(0.0),
                    corners: Default::default(),
                }));
            }
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.into_iter().map(Into::into).collect() });
            // A wall's three polylines are built by the harness from the wall itself, so these
            // are only the places for them: three a member, as the shader binds them.
            return (doc, graph, vec![Vec::new(); 12]);
        }

        if matches!(self, Scene::Primitives) {
            let mut graph = Graph::empty();
            // Three different sides, because an equal-sided box cannot catch a per-axis
            // mistake: every wrong half-extent is the same wrong half-extent on a cube, and a
            // swapped axis in the shader's normal looks identical to the right one.
            let cube = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                size: [
                    ParamRef::Literal(CAGE_MM),
                    ParamRef::Literal(CAGE_MM * 0.55),
                    ParamRef::Literal(CAGE_MM * 0.8),
                ],
                turn: fieldform_scene::no_turn(),
            });
            let sphere = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(0.2 * CAGE_MM),
                    ParamRef::Literal(0.1 * CAGE_MM),
                    ParamRef::Literal(-0.15 * CAGE_MM),
                ]),
                radius: ParamRef::Literal(0.4 * CAGE_MM),
            });
            graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![cube.into(), sphere.into()] });
            return (doc, graph, Vec::new());
        }

        // A curve given a thickness and pulled into a solid. See the four scenes' own notes for
        // what each is there to catch. The three polylines a wall is read from are handed back
        // empty: they are not something a scene can write down, and `main` builds them from the
        // wall itself, by node.
        if matches!(
            self,
            Scene::ThinExtruded
                | Scene::ThinLeaning
                | Scene::ThinTube
                | Scene::ThinRib
                | Scene::ThinCut
        ) {
            use fieldform_scene::{Outline, Sweep, WallEnds, WallSide};
            let h = 0.30 * CAGE_MM;
            // A plane that lines up with no axis, so neither the fitted plane nor the rotation
            // into it is ever the identity.
            let turn = nalgebra::Rotation3::from_axis_angle(
                &nalgebra::Unit::new_normalize(nalgebra::Vector3::new(1.0, 2.0, 0.5)),
                0.7,
            );
            let shift = nalgebra::Vector3::new(40.0, -60.0, -150.0);
            let at = |x: f64, y: f64| turn * Point3::new(x * h, y * h, 0.0) + shift;
            let lit = ParamRef::Literal;

            let curve = CurveId::new();
            match self {
                Scene::ThinTube => {
                    // An L, so the loop has a corner that turns the other way from the rest.
                    let corners = [
                        (-0.9, -0.7),
                        (0.8, -0.7),
                        (0.8, -0.1),
                        (-0.2, -0.1),
                        (-0.2, 0.8),
                        (-0.9, 0.8),
                    ];
                    let points = corners.iter().map(|&(x, y)| at(x, y)).collect();
                    doc.curves.insert(
                        curve,
                        Curve::polyline(curve, "Loop", points, true).expect("closed"),
                    );
                }
                Scene::ThinRib => {
                    doc.curves.insert(
                        curve,
                        Curve::polyline(curve, "Line", vec![at(-0.9, -0.3), at(0.8, 0.5)], false)
                            .expect("a line"),
                    );
                }
                _ => {
                    // An S as one cubic, which bends one way and then the other, and a
                    // straight run from its far end that doubles back 130 degrees to the left.
                    let control = vec![at(-1.0, -0.5), at(-0.3, -0.5), at(-0.3, 0.3), at(0.4, 0.3)];
                    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
                    doc.curves.insert(
                        curve,
                        Curve::new(curve, "S", 3, control, vec![1.0; 4], knots, false)
                            .expect("a cubic"),
                    );
                    let back = CurveId::new();
                    doc.curves.insert(
                        back,
                        Curve::polyline(back, "Back", vec![at(0.4, 0.3), at(-0.1, 0.9)], false)
                            .expect("a line"),
                    );
                }
            }

            let (side, ends) = match self {
                Scene::ThinExtruded | Scene::ThinCut => (WallSide::Left, WallEnds::Square),
                Scene::ThinLeaning => (WallSide::Right, WallEnds::Round),
                _ => (WallSide::Centre, WallEnds::Round),
            };
            let sweep = match self {
                Scene::ThinLeaning => {
                    Sweep::Toward { xyz: [lit(1.0), lit(0.0), lit(2.0)] }
                }
                // Across the line and out of the page it was drawn on, more or less.
                Scene::ThinRib => Sweep::Toward { xyz: [lit(-0.3), lit(1.0), lit(0.4)] },
                _ => Sweep::Normal,
            };
            let mut graph = Graph::empty();
            let wall = graph.add(Node::Extrude { rims: Default::default(),
                outline: Outline::Wall { curve, thickness: lit(0.12 * h), side, ends },
                sweep,
                height: lit(0.4 * CAGE_MM),
                start: lit(0.0),
                sides: fieldform_scene::Sides::symmetric_if(matches!(self, Scene::ThinTube)),
                draft: lit(if matches!(self, Scene::ThinLeaning) { 6.0 } else { 0.0 }),
                corners: Default::default(),
            });
            if matches!(self, Scene::ThinCut) {
                // A block the wall runs clean through, square to the world while the wall is
                // not, so the slot crosses it at an angle and comes out through three faces.
                let middle = turn * Point3::new(-0.3 * h, 0.1 * h, 0.2 * CAGE_MM) + shift;
                let block = graph.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([lit(middle.x), lit(middle.y), lit(middle.z)]),
                    size: [lit(1.1 * h), lit(0.9 * h), lit(0.8 * h)],
                    turn: fieldform_scene::no_turn(),
                });
                graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), wall.into()] });
            }
            return (doc, graph, vec![Vec::new(); 3]);
        }

        // A profile turned into a solid, both ways. The first blocks whose shape comes from a
        // *curve* rather than from numbers, and the first whose polyline the shader has to be
        // handed pre-flattened — so this is the scene that says the two sides agree about what
        // "in the frame" means, which is the part most likely to drift.
        if matches!(self, Scene::Extruded
            | Scene::Leaning
            | Scene::Upright
            | Scene::BakedProfile
            | Scene::Drafted
            | Scene::DraftedBake
            | Scene::DraftedFlip
            | Scene::Symmetric
            | Scene::SymmetricDrafted
            | Scene::TwoSidedBegun
            | Scene::Revolved
            | Scene::RevolvedWedge) {
            let profile = CurveId::new();
            let h = 0.30 * CAGE_MM;
            let straight =
                matches!(
                    self,
                    Scene::Extruded
                        | Scene::Leaning
                        | Scene::Upright
                        | Scene::BakedProfile
                        | Scene::Drafted
                        | Scene::DraftedBake
                        | Scene::DraftedFlip
                        | Scene::Symmetric
                        | Scene::SymmetricDrafted
                        | Scene::TwoSidedBegun
                );
            let (points, frame) = if matches!(self, Scene::Upright) {
                // The same square, stood up in the YZ plane: its normal is X, so nothing here
                // is the identity any more.
                (
                    vec![
                        Point3::new(0.0, -h, -h),
                        Point3::new(0.0, h, -h),
                        Point3::new(0.0, h, h),
                        Point3::new(0.0, -h, h),
                    ],
                    fieldform_scene::Frame::default(),
                )
            } else if straight {
                // A square in the ground plane, extruded up.
                (
                    vec![
                        Point3::new(-h, -h, 0.0),
                        Point3::new(h, -h, 0.0),
                        Point3::new(h, h, 0.0),
                        Point3::new(-h, h, 0.0),
                    ],
                    fieldform_scene::Frame::default(),
                )
            } else {
                // A rectangle clear of the axis, revolved into a tube.
                (
                    vec![
                        Point3::new(0.5 * h, 0.0, -h),
                        Point3::new(1.5 * h, 0.0, -h),
                        Point3::new(1.5 * h, 0.0, h),
                        Point3::new(0.5 * h, 0.0, h),
                    ],
                    fieldform_scene::Frame::default(),
                )
            };
            doc.curves.insert(
                profile,
                fieldform_scene::Curve::polyline(profile, "Profile", points, true)
                    .expect("closed"),
            );
            let mut graph = Graph::empty();
            if straight {
                graph.add(Node::Extrude { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(profile),
                    // Straight out of the profile's own plane, which the compile fits for
                    // itself, or leaned over in X by half its rise.
                    sweep: if matches!(self, Scene::Upright) {
                        // Typed, and typed as the profile's own normal -- which is what the
                        // panel seeds when the option is chosen, so it must land in exactly
                        // the same place as `Normal` does.
                        fieldform_scene::Sweep::Toward {
                            xyz: [
                                ParamRef::Literal(1.0),
                                ParamRef::Literal(0.0),
                                ParamRef::Literal(0.0),
                            ],
                        }
                    } else if matches!(self, Scene::Leaning | Scene::TwoSidedBegun) {
                        fieldform_scene::Sweep::Toward {
                            xyz: [
                                ParamRef::Literal(1.0),
                                ParamRef::Literal(0.0),
                                ParamRef::Literal(2.0),
                            ],
                        }
                    } else {
                        fieldform_scene::Sweep::Normal
                    },
                    height: if matches!(self, Scene::DraftedFlip) {
                        ParamRef::Literal(-0.5 * CAGE_MM)
                    } else {
                        ParamRef::Literal(0.5 * CAGE_MM)
                    },
                    // Half each way for the two symmetric scenes, which is the one case where
                    // the shader folds the sweep about the plane rather than running it out of
                    // one side. Nothing else here would notice a fold whose sign was wrong.
                    // Begun a tenth of the cage off the plane, in the one scene that begins
                    // anywhere but at it.
                    start: ParamRef::Literal(if matches!(self, Scene::TwoSidedBegun) {
                        0.1 * CAGE_MM
                    } else {
                        0.0
                    }),
                    // Two sides with their own numbers in that scene; half each way for the two
                    // symmetric ones, which is the case where the shader folds the sweep about
                    // the plane rather than running it out of one side.
                    sides: if matches!(self, Scene::TwoSidedBegun) {
                        fieldform_scene::Sides::Two {
                            height: ParamRef::Literal(0.3 * CAGE_MM),
                            draft: ParamRef::Literal(-10.0),
                        }
                    } else {
                        fieldform_scene::Sides::symmetric_if(matches!(
                            self,
                            Scene::Symmetric | Scene::SymmetricDrafted
                        ))
                    },
                    draft: if matches!(
                        self,
                        Scene::Drafted
                            | Scene::DraftedBake
                            | Scene::DraftedFlip
                            | Scene::SymmetricDrafted
                            | Scene::TwoSidedBegun
                    ) {
                        ParamRef::Literal(20.0)
                    } else {
                        ParamRef::Literal(0.0)
                    },
                    corners: Default::default(),
                });
            } else {
                // All the way round, or two hundred degrees of it: past a half turn, so which
                // cap is nearer changes along the way round outside the wedge.
                let wedge = matches!(self, Scene::RevolvedWedge);
                graph.add(Node::Revolve { rims: Default::default(),
                    outline: fieldform_scene::Outline::closed(profile),
                    frame,
                    angle: ParamRef::Literal(if wedge { 200.0 } else { 360.0 }),
                    sides: fieldform_scene::AngleSides::One,
                });
            }
            // The profile flattened into its own frame, which is what the shader is handed.
            // Built through `Profile` rather than by repeating the arithmetic, for the same
            // reason the application does: a second flattening is a second answer.
            let world = doc.curves[&profile].tessellate(0.05);
            // **An extrude's profile is read in its own fitted plane; a revolve's is not.**
            // This said `|p| p` for both, which is right only while the profile lies in the
            // ground plane -- so the flattening the shader is handed was never checked against
            // a rotation until the upright scene. A revolve keeps the identity because its
            // profile is read in world (r, z) about its frame's axis, not flattened at all.
            let (origin, normal) = fieldform_field::plane_of(&world)
                .unwrap_or((Point3::origin(), nalgebra::Vector3::z()));
            let rotation = fieldform_field::rotation_from_z(normal);
            let into = move |p: Point3<f64>| {
                if straight {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                } else {
                    p
                }
            };
            let flat = if straight {
                fieldform_field::Profile::in_plane(&world, into, 1e-3).expect("planar")
            } else {
                fieldform_field::Profile::about_axis(&world, into)
                    .expect("clear of the axis")
            };
            let spine: Vec<Point3<f64>> = flat
                .points()
                .iter()
                .map(|p| Point3::new(p.x, p.y, 0.0))
                .collect();
            return (doc, graph, vec![spine]);
        }

        // A curve given a thickness and turned about an axis. In the half-plane y = 0 about
        // the world's Z, which is what a default frame is, so `(r, z)` here is `(x, z)` there.
        if matches!(self, Scene::ThinRevolved | Scene::ThinRevolvedAcross | Scene::ThinRevolvedWedge) {
            use fieldform_scene::{Outline, WallEnds, WallSide};
            let h = 0.30 * CAGE_MM;
            let at = |r: f64, z: f64| Point3::new(r * h, 0.0, z * h);
            let lit = ParamRef::Literal;
            let curve = CurveId::new();
            let (thickness, side, ends) = if matches!(self, Scene::ThinRevolvedAcross) {
                // Drawn upward, so its left is toward the axis, and half again as much wall
                // as the widest gap between it and the axis.
                doc.curves.insert(
                    curve,
                    Curve::polyline(curve, "Line", vec![at(0.10, -0.6), at(0.16, 0.7)], false)
                        .expect("a line"),
                );
                (0.5 * h, WallSide::Left, WallEnds::Round)
            } else {
                // The S of the thin extrudes, stood in the half-plane clear of the axis, with
                // the straight run from its far end doubling back past a right angle.
                let control = vec![at(0.6, -0.6), at(1.3, -0.6), at(1.3, 0.2), at(2.0, 0.2)];
                let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
                doc.curves.insert(
                    curve,
                    Curve::new(curve, "S", 3, control, vec![1.0; 4], knots, false)
                        .expect("a cubic"),
                );
                let back = CurveId::new();
                doc.curves.insert(
                    back,
                    Curve::polyline(back, "Back", vec![at(2.0, 0.2), at(1.5, 0.8)], false)
                        .expect("a line"),
                );
                (0.12 * h, WallSide::Left, WallEnds::Square)
            };
            let mut graph = Graph::empty();
            let wedge = matches!(self, Scene::ThinRevolvedWedge);
            graph.add(Node::Revolve { rims: Default::default(),
                outline: Outline::Wall { curve, thickness: lit(thickness), side, ends },
                frame: fieldform_scene::Frame::default(),
                angle: lit(if wedge { 90.0 } else { 360.0 }),
                sides: if wedge {
                    fieldform_scene::AngleSides::Two { angle: lit(45.0) }
                } else {
                    fieldform_scene::AngleSides::One
                },
            });
            return (doc, graph, vec![Vec::new(); 3]);
        }

        // A drafted solid whose corners are **mitred**: the section as drawn, and the miter
        // vectors that say where each of its corners goes as the draft opens. Both are derived
        // by `main` from the node, so what is handed back here is their places.
        if matches!(self, Scene::SharpDraft) {
            use fieldform_scene::{Outline, WallEnds, WallSide};
            let h = 0.22 * CAGE_MM;
            let at = |x: f64, y: f64| Point3::new(x * h, y * h, 0.0);
            // An L: five convex corners and one reflex, so the reflex case is measured too.
            let curve = CurveId::new();
            doc.curves.insert(
                curve,
                Curve::polyline(
                    curve,
                    "L",
                    vec![
                        at(-1.0, -1.0),
                        at(1.0, -1.0),
                        at(1.0, -0.3),
                        at(-0.1, -0.3),
                        at(-0.1, 1.0),
                        at(-1.0, 1.0),
                    ],
                    true,
                )
                .expect("closed"),
            );
            // Only a closed profile now: mitring a wall is refused by name.
            let wall = false;
            let mut graph = Graph::empty();
            let node = graph.add(Node::Extrude { rims: Default::default(),
                outline: match wall {
                    true => Outline::Wall {
                        curve,
                        thickness: ParamRef::Literal(0.06 * CAGE_MM),
                        side: WallSide::Centre,
                        ends: WallEnds::Square,
                    },
                    false => Outline::closed(curve),
                },
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(0.55 * CAGE_MM),
                start: ParamRef::Literal(-0.3 * CAGE_MM),
                sides: fieldform_scene::Sides::One,
                // Steep enough that the mitre stands well clear of the arc it replaces.
                draft: ParamRef::Literal(12.0),
                corners: fieldform_scene::Corners::Sharp,
            });
            let _ = node;
            // A profile's own polyline flattened, and a place for the miters, which `main`
            // derives from the node. A wall hands back its three parts instead.
            let world = doc.curves[&curve].tessellate(0.05);
            let flat: Vec<Point3<f64>> =
                world.iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
            let handed = match wall {
                true => vec![flat.clone(), flat.clone(), flat.clone(), flat],
                false => vec![flat.clone(), flat],
            };
            return (doc, graph, handed);
        }

        if matches!(self, Scene::LoftedDerived) {
            // Two rings of one radius, level, a quarter of the cube apart in height and offset
            // sideways by nearly as much, so the centerline the loft derives is an S setting
            // off square to each -- and the section is read in the S's own frame.
            let ring = |centre: Point3<f64>, r: f64| -> Vec<Point3<f64>> {
                (0..64)
                    .map(|i| {
                        let a = std::f64::consts::TAU * i as f64 / 64.0;
                        centre + nalgebra::Vector3::new(r * a.cos(), r * a.sin(), 0.0)
                    })
                    .collect()
            };
            let (low, high) = (CurveId::new(), CurveId::new());
            doc.curves.insert(
                low,
                Curve::polyline(low, "Low", ring(Point3::new(-0.12 * s, 0.0, -0.25 * s), 0.10 * s), true)
                    .expect("closed"),
            );
            doc.curves.insert(
                high,
                Curve::polyline(high, "High", ring(Point3::new(0.12 * s, 0.0, 0.25 * s), 0.10 * s), true)
                    .expect("closed"),
            );
            let sections = vec![
                fieldform_scene::Outline::closed(low),
                fieldform_scene::Outline::closed(high),
            ];
            let mut graph = Graph::empty();
            let node = graph.add(Node::Loft { rims: Default::default(),
                sections: sections.clone(),
                spine: None,
                smooth: fieldform_scene::LoftSmooth::G1,
            });
            let built = loft_of(&doc, &graph, node);
            assert!(
                built.bend_share() < 0.5,
                "the S bends to {:.2} of its fold, which is not the gentle case this scene is for",
                built.bend_share()
            );
            let mut handed = vec![
                built.spine.points().to_vec(),
                built.frame_normals().iter().map(|n| Point3::from(*n)).collect(),
            ];
            for k in 0..sections.len() {
                handed.push(built.section_points(k).iter().map(|q| Point3::new(q.x, q.y, 0.0)).collect());
            }
            for k in 0..sections.len() {
                handed.push(built.section_slopes(k).iter().map(|m| Point3::new(m.x, m.y, 0.0)).collect());
            }
            return (doc, graph, handed);
        }

        if matches!(self, Scene::Lofted | Scene::LoftedG0) {
            // The path: a single-span cubic setting off upward and bending in two planes --
            // **genuinely two**: its control points are far from coplanar, so the frame turns
            // about the tangent as well as with it, and a shader that got that turn wrong
            // would read sections 90 degrees apart in their asymmetry. The first path here lay
            // within 2 mm of a plane over 836 mm and could not have told; found in review.
            // And gently: the loft reports how far toward folding it is, asserted below.
            let path = CurveId::new();
            let bend = Curve::new(
                path,
                "Path",
                3,
                vec![
                    Point3::new(0.0, 0.0, -0.30 * s),
                    Point3::new(-0.10 * s, 0.02 * s, 0.0),
                    Point3::new(0.06 * s, 0.14 * s, 0.12 * s),
                    Point3::new(0.30 * s, 0.10 * s, 0.34 * s),
                ],
                vec![1.0; 4],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("a single-span cubic");
            // Four sections, each drawn flat and centred on the path at a quarter of its
            // parameter -- stationed by the loft at the arc nearest its centroid, so they fall
            // in order. Flat rather than square to the path on purpose: a sketch is drawn on a
            // plane, and the projection is the loft's to make on both sides.
            let flat = |centre: Point3<f64>, shape: Vec<Point3<f64>>| -> Vec<Point3<f64>> {
                shape.into_iter().map(|p| centre + p.coords).collect()
            };
            let w = std::f64::consts::FRAC_1_SQRT_2;
            // The rational quadratic circle's control polygon, scaled to `a` by `b`: an
            // affine image of a NURBS circle under the same weights is an exact ellipse.
            let oval = |a: f64, b: f64| -> Vec<Point3<f64>> {
                vec![
                    Point3::new(a, 0.0, 0.0),
                    Point3::new(a, b, 0.0),
                    Point3::new(0.0, b, 0.0),
                    Point3::new(-a, b, 0.0),
                    Point3::new(-a, 0.0, 0.0),
                    Point3::new(-a, -b, 0.0),
                    Point3::new(0.0, -b, 0.0),
                    Point3::new(a, -b, 0.0),
                    Point3::new(a, 0.0, 0.0),
                ]
            };
            let oval_weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
            let oval_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
            // A rounded rectangle as a polyline: four sides and four quarter arcs of six
            // chords each, so its corners are round to the eye and its sides exactly straight.
            let rounded = |half_x: f64, half_y: f64, r: f64| -> Vec<Point3<f64>> {
                let mut out = Vec::new();
                let corners = [(1.0, 1.0, 0.0), (-1.0, 1.0, 90.0), (-1.0, -1.0, 180.0), (1.0, -1.0, 270.0)];
                for (sx, sy, from) in corners {
                    let (cx, cy) = (sx * (half_x - r), sy * (half_y - r));
                    for i in 0..=6 {
                        let a = (from + 90.0 * i as f64 / 6.0f64).to_radians();
                        out.push(Point3::new(cx + r * a.cos(), cy + r * a.sin(), 0.0));
                    }
                }
                out
            };
            let at = |t: f64| bend.evaluate(t);
            let square = CurveId::new();
            let circle = CurveId::new();
            let rect = CurveId::new();
            let ellipse = CurveId::new();
            let h = 0.10 * s;
            doc.curves.insert(
                square,
                Curve::polyline(
                    square,
                    "Square",
                    flat(
                        at(0.0),
                        vec![
                            Point3::new(-h, -h, 0.0),
                            Point3::new(h, -h, 0.0),
                            Point3::new(h, h, 0.0),
                            Point3::new(-h, h, 0.0),
                        ],
                    ),
                    true,
                )
                .expect("closed"),
            );
            doc.curves.insert(
                circle,
                Curve::new(
                    circle,
                    "Circle",
                    2,
                    flat(at(1.0 / 3.0), oval(0.09 * s, 0.09 * s)),
                    oval_weights.clone(),
                    oval_knots.clone(),
                    true,
                )
                .expect("a rational quadratic circle"),
            );
            doc.curves.insert(
                rect,
                Curve::polyline(
                    rect,
                    "Rounded rectangle",
                    flat(at(2.0 / 3.0), rounded(0.12 * s, 0.07 * s, 0.03 * s)),
                    true,
                )
                .expect("closed"),
            );
            doc.curves.insert(
                ellipse,
                Curve::new(
                    ellipse,
                    "Ellipse",
                    2,
                    flat(at(1.0), oval(0.11 * s, 0.06 * s)),
                    oval_weights,
                    oval_knots,
                    true,
                )
                .expect("a rational quadratic ellipse"),
            );
            doc.curves.insert(path, bend);
            let sections: Vec<fieldform_scene::Outline> = [square, circle, rect, ellipse]
                .into_iter()
                .map(fieldform_scene::Outline::closed)
                .collect();
            let mut graph = Graph::empty();
            let node = graph.add(Node::Loft { rims: Default::default(),
                sections: sections.clone(),
                spine: Some(path),
                smooth: match self {
                    Scene::LoftedG0 => fieldform_scene::LoftSmooth::G0,
                    _ => fieldform_scene::LoftSmooth::G1,
                },
            });
            // Six bindings in the order codegen interns them -- the centerline, its frame,
            // then each section -- all of which `main` derives from the node again; only their
            // places are handed back.
            let built = loft_of(&doc, &graph, node);
            assert!(
                built.bend_share() < 0.5,
                "the path bends to {:.2} of the loft's fold, which is not the gentle case this \
                 scene is for",
                built.bend_share()
            );
            let mut handed = vec![
                built.spine.points().to_vec(),
                built.frame_normals().iter().map(|n| Point3::from(*n)).collect(),
            ];
            for k in 0..sections.len() {
                handed.push(built.section_points(k).iter().map(|q| Point3::new(q.x, q.y, 0.0)).collect());
            }
            // And at G1 the rails' slopes, one binding per section after the sections.
            if !matches!(self, Scene::LoftedG0) {
                for k in 0..sections.len() {
                    handed.push(built.section_slopes(k).iter().map(|m| Point3::new(m.x, m.y, 0.0)).collect());
                }
            }
            return (doc, graph, handed);
        }

        // A wall round a *closed* loop, drafted, carried along a path -- the combination no
        // scene covered, and the one a real shader-versus-field split hid in. The draft grows
        // the band on both faces, so the disagreement to catch is down the bore.
        if matches!(self, Scene::DraftedWallSweep) {
            use fieldform_scene::{Outline, WallEnds, WallSide};
            let h = 0.16 * CAGE_MM;
            let loop_id = CurveId::new();
            doc.curves.insert(
                loop_id,
                Curve::polyline(
                    loop_id,
                    "Loop",
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
            // Setting off from under the section, as every sweep scene's path does.
            let path = CurveId::new();
            let bend = Curve::new(
                path,
                "Path",
                3,
                vec![
                    Point3::new(0.0, 0.0, -0.30 * s),
                    Point3::new(-0.10 * s, -0.10 * s, 0.00 * s),
                    Point3::new(0.10 * s, 0.08 * s, 0.14 * s),
                    Point3::new(0.26 * s, 0.22 * s, 0.34 * s),
                ],
                vec![1.0; 4],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("a single-span cubic");
            let spine_points = bend.tessellate(TOLERANCE_MM);
            doc.curves.insert(path, bend);
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(),
                outline: Outline::Wall {
                    curve: loop_id,
                    thickness: ParamRef::Literal(0.05 * CAGE_MM),
                    side: WallSide::Centre,
                    ends: WallEnds::Square,
                },
                spine: path,
                section: fieldform_scene::Section::Square,
                law: fieldform_scene::SweepLaw {
                    // Gentle: the band grows by the offset on each side, so the reach the
                    // fold test measures is the loop plus the whole draft, and this path bends.
                    taper: fieldform_scene::Taper::Draft(ParamRef::Literal(0.8)),
                    ..Default::default()
                },
            });
            // The wall's three polylines, which `main` derives from the node, then the spine
            // and its frame. Only their places are handed back.
            let flat: Vec<Point3<f64>> = doc.curves[&loop_id]
                .tessellate(0.05)
                .iter()
                .map(|p| Point3::new(p.x, p.y, 0.0))
                .collect();
            let frame = fieldform_field::sweep_frame(
                &doc,
                &Outline::Wall {
                    curve: loop_id,
                    thickness: ParamRef::Literal(0.05 * CAGE_MM),
                    side: WallSide::Centre,
                    ends: WallEnds::Square,
                },
                path,
                &spine_points,
            )
            .expect("a spine with a direction");
            return (
                doc,
                graph,
                vec![flat.clone(), flat.clone(), flat, spine_points, frame],
            );
        }

        // An open section carried along a curve: the wall's three polylines, then the spine and
        // its frame, which `main` derives by node. Only their places are handed back.
        if matches!(self, Scene::ThinSwept | Scene::ThinSweptScaled) {
            use fieldform_scene::{Outline, WallEnds, WallSide};
            let hs = 0.07 * CAGE_MM;
            let at = |x: f64, y: f64| Point3::new(x * hs, y * hs, 0.0);
            let curve = CurveId::new();
            let control = vec![at(-1.0, -0.5), at(-0.3, -0.5), at(-0.3, 0.3), at(0.4, 0.3)];
            let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
            doc.curves.insert(
                curve,
                Curve::new(curve, "S", 3, control, vec![1.0; 4], knots, false).expect("a cubic"),
            );
            let back = CurveId::new();
            doc.curves.insert(
                back,
                Curve::polyline(back, "Back", vec![at(0.4, 0.3), at(-0.1, 0.9)], false)
                    .expect("a line"),
            );
            // The thick sweep's own path: smooth, for the reason given there.
            let path = CurveId::new();
            let bend = Curve::new(
                path,
                "Path",
                3,
                vec![
                    // **Setting off from under the section**, 0.3 of the cube below its plane,
                    // since 2026-10-02 a section is carried by the point its path sets off from
                    // dropped onto its plane: here the section's own middle, with the 0.3 as
                    // the offset that carries the solid up to start at the section. The path
                    // used to set off 0.43 of the cube to one side as well, which under that
                    // rule carries the section on an arm wider than the bend -- a fold, refused.
                    Point3::new(0.0, 0.0, -0.30 * s),
                    Point3::new(-0.15 * s, -0.15 * s, 0.05 * s),
                    Point3::new(0.15 * s, 0.10 * s, 0.10 * s),
                    Point3::new(0.35 * s, 0.30 * s, 0.35 * s),
                ],
                vec![1.0; 4],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("a single-span cubic");
            doc.curves.insert(path, bend);
            let mut graph = Graph::empty();
            graph.add(Node::Sweep { rims: Default::default(),
                outline: Outline::Wall {
                    curve,
                    thickness: ParamRef::Literal(0.25 * hs),
                    side: WallSide::Left,
                    ends: WallEnds::Square,
                },
                spine: path,
                section: fieldform_scene::Section::Square,
                // Scaled to a half on the way, where the scaled scene asks for one: an open
                // wall scales whole, thickness and all, which is the arm of
                // `Wall::distance_tapered` the closed ones never reach.
                law: match self {
                    Scene::ThinSweptScaled => fieldform_scene::SweepLaw {
                        taper: fieldform_scene::Taper::Scale(ParamRef::Literal(0.5)),
                        ..Default::default()
                    },
                    _ => Default::default(),
                },
            });
            return (doc, graph, vec![Vec::new(); 5]);
        }

        // A profile carried along a curve, handed back in the shader's binding order --
        // profile, spine, frame -- with the profile flattened as an extrude's is and the frame
        // computed from the very points the spine goes up as, which is what `sweep_frame` asks
        // for and what `CurveUse::Frame` promises the shader.
        if matches!(
            self,
            Scene::Swept
                | Scene::SweptFlush
                | Scene::SweptParallel
                | Scene::SweptScaled
                | Scene::SweptDrafted
                | Scene::SweptTwistedWindow
        ) {
            let h = 0.1 * CAGE_MM;
            let profile = CurveId::new();
            doc.curves.insert(
                profile,
                Curve::polyline(
                    profile,
                    "Section",
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
            // **A smooth spine, on purpose.** This was a polyline through the same four points,
            // and it measured 1.1 mm -- at a point exactly equidistant from the two legs of the
            // middle corner. A corner is a tie between two segments, the CPU breaks it in f64
            // and the shader in f32, and the two branches read the section at frames 43 mm
            // apart along the curve. That is the field's own seam at a sharp corner (recorded
            // in `sweep-spec.md` section 9), and it is what this example must not be measuring:
            // on a smooth spine flattened to the tolerance, the joints are the tolerance apart.
            let path = CurveId::new();
            let bend = Curve::new(
                path,
                "Path",
                3,
                vec![
                    // **Setting off from under the section**, 0.3 of the cube below its plane,
                    // since 2026-10-02 a section is carried by the point its path sets off from
                    // dropped onto its plane: here the section's own middle, with the 0.3 as
                    // the offset that carries the solid up to start at the section. The path
                    // used to set off 0.43 of the cube to one side as well, which under that
                    // rule carries the section on an arm wider than the bend -- a fold, refused.
                    Point3::new(0.0, 0.0, -0.30 * s),
                    Point3::new(-0.15 * s, -0.15 * s, 0.05 * s),
                    Point3::new(0.15 * s, 0.10 * s, 0.10 * s),
                    Point3::new(0.35 * s, 0.30 * s, 0.35 * s),
                ],
                vec![1.0; 4],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("a single-span cubic");
            let spine_points = bend.tessellate(TOLERANCE_MM);
            doc.curves.insert(path, bend);
            let mut graph = Graph::empty();
            // **The held sections measured against the same shape.** A tilt the shader reads
            // differently from the kernel is a solid in two places, and this is the only thing
            // that would say so.
            let lit = ParamRef::Literal;
            graph.add(Node::Sweep { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(profile),
                spine: path,
                section: match self {
                    Scene::SweptFlush => fieldform_scene::Section::Skewed,
                    Scene::SweptParallel => fieldform_scene::Section::Parallel,
                    _ => fieldform_scene::Section::Square,
                },
                law: match self {
                    Scene::SweptScaled => fieldform_scene::SweepLaw {
                        taper: fieldform_scene::Taper::Scale(lit(0.34)),
                        shape: fieldform_scene::RampShape::Smooth,
                        ..Default::default()
                    },
                    Scene::SweptDrafted => fieldform_scene::SweepLaw {
                        taper: fieldform_scene::Taper::Draft(lit(2.0)),
                        ..Default::default()
                    },
                    Scene::SweptTwistedWindow => fieldform_scene::SweepLaw {
                        twist: lit(90.0),
                        extent: [lit(20.0), lit(75.0)],
                        ..Default::default()
                    },
                    _ => Default::default(),
                },
            });

            let world = doc.curves[&profile].tessellate(0.05);
            let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
            let rotation = fieldform_field::rotation_from_z(normal);
            let into = move |p: Point3<f64>| {
                Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
            };
            let flat = fieldform_field::Profile::in_plane(&world, into, 1e-3).expect("planar");
            let section: Vec<Point3<f64>> =
                flat.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
            let frame = fieldform_field::sweep_frame(
                &doc,
                &fieldform_scene::Outline::closed(profile),
                path,
                &spine_points,
            )
            .expect("a spine with a direction");
            return (doc, graph, vec![section, spine_points, frame]);
        }

        // A graded lattice: the ramp reads a plane through the middle of the cube, so the
        // phase's number runs from its low end at the cube's floor to its high end at the top.
        let graded: Option<(fieldform_scene::LatticeCell, fn(ParamRef) -> LatticePhase, f64, f64)> =
            match self {
                Scene::GradedWall => Some((
                    fieldform_scene::LatticeCell::Gyroid,
                    |thickness| LatticePhase::sheet(thickness),
                    CAGE_MM * 0.01,
                    CAGE_MM * 0.04,
                )),
                Scene::GradedDensity => Some((
                    fieldform_scene::LatticeCell::SchwarzP,
                    |density| LatticePhase::Solid { density },
                    0.15,
                    0.7,
                )),
                Scene::GradedStrut => Some((
                    fieldform_scene::LatticeCell::Bcc,
                    |radius| LatticePhase::Strut { radius },
                    CAGE_MM * 0.008,
                    CAGE_MM * 0.03,
                )),
                _ => None,
            };
        if let Some((cell, phase, low, high)) = graded {
            let mut graph = Graph::empty();
            let bounding = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                size: [ParamRef::Literal(CAGE_MM * 0.7); 3],
                turn: fieldform_scene::no_turn(),
            });
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let ramp = graph.add(Node::Ramp {
                input: plane.into(),
                in_min: ParamRef::Literal(-CAGE_MM * 0.35),
                in_max: ParamRef::Literal(CAGE_MM * 0.35),
                out_min: ParamRef::Literal(low),
                out_max: ParamRef::Literal(high),
                shape: fieldform_scene::RampShape::Linear,
            });
            let lattice = graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell,
                map: fieldform_scene::CellMap::Rectangular {
                    cell: ParamRef::Literal(CAGE_MM * 0.25),
                },
                phase: phase(ParamRef::Field(ramp)),
            });
            graph.add(Node::Boolean {
                op: BooleanOp::Intersection,
                inputs: vec![bounding.into(), lattice.into()],
            });
            return (doc, graph, Vec::new());
        }

        if let Some(cell) = self.lattice_cell() {
            let mut graph = Graph::empty();
            // A curved boundary for one scene and a flat one for the rest, since a seam
            // that only appears against curvature would hide in a box.
            let bounding = if matches!(self, Scene::SphereInfill) {
                graph.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    radius: ParamRef::Literal(CAGE_MM * 0.4),
                })
            } else {
                graph.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                    size: [ParamRef::Literal(CAGE_MM * 0.7); 3],
                    turn: fieldform_scene::no_turn(),
                })
            };
            let lattice = graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell,
                map: fieldform_scene::CellMap::Rectangular {
                    cell: ParamRef::Literal(CAGE_MM * 0.25),
                },
                phase: if matches!(self, Scene::WalledInfill) {
                    // Half the wall's own thickness off the surface, well inside what a gyroid
                    // on cells this size can hold.
                    LatticePhase::Sheet {
                        thickness: ParamRef::Literal(CAGE_MM * 0.03),
                        offset: ParamRef::Literal(CAGE_MM * 0.015),
                    }
                } else {
                    LatticePhase::sheet(ParamRef::Literal(CAGE_MM * 0.03))
                },
            });
            // Intersection, never a blend: the lattice's field is a lower bound, and the graph
            // refuses to blend one. That refusal is the reason this composes only this way.
            graph.add(Node::Boolean {
                op: BooleanOp::Intersection,
                inputs: vec![bounding.into(), lattice.into()],
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedFillet) {
            let mut graph = Graph::empty();
            // Pitched closer than their own diameter, so the copies overlap and there is a
            // seam to round. A radius large enough to move the surface a long way further than
            // either path's own numerical noise.
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                radius: ParamRef::Literal(CAGE_MM * 0.12),
            });
            graph.add(Node::Pattern {
                options: Default::default(),
                input: ball.into(),
                count: ParamRef::Literal(5.0),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [
                        ParamRef::Literal(CAGE_MM * 0.19),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                    ],
                },
                join: true,
                fillet: Fillet::of(ParamRef::Literal(CAGE_MM * 0.05)),
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::G2PatternedFillet) {
            let mut graph = Graph::empty();
            // The PatternedFillet scene exactly, at the cubic: pitched closer than their diameter
            // so there is a seam, and a radius big enough to move the surface a long way.
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                radius: ParamRef::Literal(CAGE_MM * 0.12),
            });
            graph.add(Node::Pattern {
                options: Default::default(),
                input: ball.into(),
                count: ParamRef::Literal(5.0),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [
                        ParamRef::Literal(CAGE_MM * 0.19),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                    ],
                },
                join: true,
                fillet: Fillet {
                    radius: ParamRef::Literal(CAGE_MM * 0.05),
                    continuity: Continuity::G2,
                    second: None,
                },
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedRun) {
            let mut graph = Graph::empty();
            // Off the origin and unequal on every side, so a copy shifted by a step shows.
            let lug = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(CAGE_MM * 0.02),
                    ParamRef::Literal(CAGE_MM * -0.03),
                    ParamRef::Literal(0.0),
                ]),
                size: [
                    ParamRef::Literal(CAGE_MM * 0.09),
                    ParamRef::Literal(CAGE_MM * 0.16),
                    ParamRef::Literal(CAGE_MM * 0.1),
                ],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Pattern {
                input: lug.into(),
                count: ParamRef::Literal(8.0),
                kind: fieldform_scene::PatternKind::Linear {
                    // The whole run, on every axis: 0.84 across, 0.21 up, -0.14 deep.
                    step: [
                        ParamRef::Literal(CAGE_MM * 0.84),
                        ParamRef::Literal(CAGE_MM * 0.21),
                        ParamRef::Literal(CAGE_MM * -0.14),
                    ],
                },
                // Eight is even, so the centred row straddles the body and no copy stands on it:
                // the block holds every copy whatever this says, which is the rule being checked.
                join: false,
                fillet: Fillet::hard(),
                options: fieldform_scene::PatternOptions {
                    spacing: fieldform_scene::Spacing::Extent,
                    symmetric: true,
                    skip: vec![2, 5],
                    ..Default::default()
                },
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedGrid | Scene::PatternedLongGrid) {
            let mut graph = Graph::empty();
            let long = matches!(self, Scene::PatternedLongGrid);
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(CAGE_MM * 0.01),
                    ParamRef::Literal(CAGE_MM * -0.01),
                    ParamRef::Literal(0.0),
                ]),
                radius: ParamRef::Literal(CAGE_MM * if long { 0.016 } else { 0.022 }),
            });
            let (columns, rows) = if long { (3.0, 36.0) } else { (12.0, 9.0) };
            graph.add(Node::Pattern {
                input: ball.into(),
                count: ParamRef::Literal(columns),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [
                        ParamRef::Literal(CAGE_MM * 0.075),
                        ParamRef::Literal(CAGE_MM * 0.02),
                        ParamRef::Literal(0.0),
                    ],
                },
                join: true,
                fillet: Fillet::hard(),
                options: fieldform_scene::PatternOptions {
                    symmetric: true,
                    second: Some(fieldform_scene::PatternSecond {
                        // Out of the first step's plane, and at about sixty degrees to it.
                        step: [
                            ParamRef::Literal(CAGE_MM * 0.03),
                            ParamRef::Literal(CAGE_MM * if long { 0.045 } else { 0.07 }),
                            ParamRef::Literal(CAGE_MM * 0.015),
                        ],
                        count: ParamRef::Literal(rows),
                    }),
                    // Twelve by nine is even both ways, so nothing stands on the body.
                    skip: if long { vec![4, 40, 77] } else { vec![5, 30, 31, 90] },
                    ..Default::default()
                },
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedLongRun) {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(CAGE_MM * 0.01),
                    ParamRef::Literal(CAGE_MM * -0.02),
                    ParamRef::Literal(0.0),
                ]),
                radius: ParamRef::Literal(CAGE_MM * 0.018),
            });
            graph.add(Node::Pattern {
                input: ball.into(),
                count: ParamRef::Literal(40.0),
                kind: fieldform_scene::PatternKind::Linear {
                    step: [
                        ParamRef::Literal(CAGE_MM * 0.9),
                        ParamRef::Literal(CAGE_MM * 0.2),
                        ParamRef::Literal(CAGE_MM * 0.05),
                    ],
                },
                join: true,
                fillet: Fillet::hard(),
                options: fieldform_scene::PatternOptions {
                    spacing: fieldform_scene::Spacing::Extent,
                    symmetric: true,
                    // Forty is even, so nothing stands on the body and all of these may go.
                    skip: vec![10, 11, 25],
                    ..Default::default()
                },
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedLongArc) {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(CAGE_MM * 0.3),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(0.0),
                ]),
                radius: ParamRef::Literal(CAGE_MM * 0.022),
            });
            graph.add(Node::Pattern {
                input: ball.into(),
                count: ParamRef::Literal(40.0),
                kind: fieldform_scene::PatternKind::Radial {
                    frame: fieldform_scene::Frame::at_along(
                        fieldform_scene::Axis::Z,
                        [
                            ParamRef::Literal(CAGE_MM * 0.03),
                            ParamRef::Literal(CAGE_MM * -0.02),
                            ParamRef::Literal(0.0),
                        ],
                    ),
                },
                join: true,
                fillet: Fillet::hard(),
                options: fieldform_scene::PatternOptions {
                    angle: ParamRef::Literal(270.0),
                    symmetric: true,
                    skip: vec![7],
                    ..Default::default()
                },
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedArc | Scene::PatternedArcFillet) {
            let mut graph = Graph::empty();
            let fillet = matches!(self, Scene::PatternedArcFillet);
            // A sphere for the rounded one, so the seam is the thing measured; a lug for the hard
            // one, so a copy turned the wrong way shows.
            let body = if fillet {
                graph.add(Node::Sphere {
                    centre: fieldform_scene::Place::At([ParamRef::Literal(CAGE_MM * 0.32), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
                    radius: ParamRef::Literal(CAGE_MM * 0.075),
                })
            } else {
                graph.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([
                        ParamRef::Literal(CAGE_MM * 0.3),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                    ]),
                    size: [
                        ParamRef::Literal(CAGE_MM * 0.14),
                        ParamRef::Literal(CAGE_MM * 0.07),
                        ParamRef::Literal(CAGE_MM * 0.2),
                    ],
                    turn: fieldform_scene::no_turn(),
                })
            };
            graph.add(Node::Pattern {
                input: body.into(),
                count: ParamRef::Literal(if fillet { 8.0 } else { 9.0 }),
                kind: fieldform_scene::PatternKind::Radial {
                    // About a point off the origin, the arm where the two paths could agree on
                    // the angle and still disagree on where the copies land.
                    frame: fieldform_scene::Frame::at_along(
                        fieldform_scene::Axis::Z,
                        [
                            ParamRef::Literal(CAGE_MM * 0.04),
                            ParamRef::Literal(CAGE_MM * -0.03),
                            ParamRef::Literal(0.0),
                        ],
                    ),
                },
                join: true,
                fillet: if fillet {
                    Fillet::of(ParamRef::Literal(CAGE_MM * 0.03))
                } else {
                    Fillet::hard()
                },
                options: fieldform_scene::PatternOptions {
                    // Over less than a turn, centred on the body: the arc runs from minus to plus
                    // half of it, so the gap is behind the ring and not at the body.
                    angle: ParamRef::Literal(if fillet { 250.0 } else { 200.0 }),
                    symmetric: true,
                    // The copy on the body is the middle rounded down — 4 of nine, 3 of eight,
                    // since a centred count is the total and the bias a whole number of steps —
                    // and is not skippable; any other is. Copy 2 of eight is the body's
                    // neighbour, so the gap still opens beside the body, between two seams.
                    skip: if fillet { vec![2] } else { vec![1, 6] },
                    ..Default::default()
                },
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::PatternedRing) {
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                radius: ParamRef::Literal(20.0),
            });
            graph.add(Node::Pattern {
                options: Default::default(),
                input: ball.into(),
                count: ParamRef::Literal(24.0),
                kind: fieldform_scene::PatternKind::Radial {
                    frame: fieldform_scene::Frame::at_along(
                        fieldform_scene::Axis::Z,
                        [
                            ParamRef::Literal(125.0),
                            ParamRef::Literal(0.0),
                            ParamRef::Literal(0.0),
                        ],
                    ),
                },
                // **Copies only**, which is what the tool now builds -- so the scene checking
                // the two paths agree is a scene of the thing people will have. The skip has to
                // happen on both sides, or the residual finds instance zero.
                join: false,
                fillet: Fillet::hard(),
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::Patterned | Scene::PatternedRow) {
            let mut graph = Graph::empty();
            // Off the axis and asymmetric, so a copy landing at the wrong angle shows.
            let lug = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(CAGE_MM * 0.32),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(0.0),
                ]),
                size: [
                    ParamRef::Literal(CAGE_MM * 0.3),
                    ParamRef::Literal(CAGE_MM * 0.12),
                    ParamRef::Literal(CAGE_MM * 0.4),
                ],
                turn: fieldform_scene::no_turn(),
            });
            let (count, kind) = if matches!(self, Scene::Patterned) {
                // About a point off the origin, which is the arm where the two paths could
                // agree on the angle and still disagree on where the copies land.
                (
                    6,
                    fieldform_scene::PatternKind::Radial { frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Z, [
                            ParamRef::Literal(CAGE_MM * 0.05),
                            ParamRef::Literal(CAGE_MM * -0.04),
                            ParamRef::Literal(0.0),
                        ]) },
                )
            } else {
                // A step on every axis: a shader that multiplied out only the first would put
                // the row in a straight line, which is what a plausible failure looks like.
                (
                    5,
                    fieldform_scene::PatternKind::Linear {
                        step: [
                            ParamRef::Literal(CAGE_MM * 0.26),
                            ParamRef::Literal(CAGE_MM * 0.07),
                            ParamRef::Literal(CAGE_MM * -0.05),
                        ],
                    },
                )
            };
            graph.add(Node::Pattern { options: Default::default(), input: lug.into(), count: ParamRef::Literal(f64::from(count)), kind, join: true, fillet: Fillet::hard() });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::VaryingOffset) {
            let mut graph = Graph::empty();
            // A plane low in the part, so its reading over the body's height spans the ramp's
            // whole input range rather than sitting clamped at one end — a scene where the
            // easing never leaves its flat region would agree however wrong the shape was.
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At([
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(-CAGE_MM * 0.3),
                ]),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let ramp = graph.add(Node::Ramp {
                input: plane.into(),
                in_min: ParamRef::Literal(0.0),
                in_max: ParamRef::Literal(CAGE_MM * 0.6),
                // Signed both ways, so the body is eroded at the bottom and grown at the top:
                // a sign dropped from either path shows as the taper running backwards.
                out_min: ParamRef::Literal(-CAGE_MM * 0.06),
                out_max: ParamRef::Literal(CAGE_MM * 0.12),
                shape: fieldform_scene::RampShape::Smoother,
            });
            let body = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                size: [
                    ParamRef::Literal(CAGE_MM * 0.5),
                    ParamRef::Literal(CAGE_MM * 0.5),
                    ParamRef::Literal(CAGE_MM * 0.6),
                ],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Offset { input: body.into(), distance: ParamRef::Field(ramp) });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::ProtectedBore) {
            let lit = ParamRef::Literal;
            let mm = CAGE_MM;
            let fillet = |k: f64| Fillet { radius: lit(k * mm), continuity: Continuity::G1, second: None };
            let mut graph = Graph::empty();
            let slab = |graph: &mut Graph, x: f64| {
                graph.add(Node::Cuboid { edge: Default::default(),
                    centre: fieldform_scene::Place::At([lit(x * mm), lit(0.0), lit(-0.1 * mm)]),
                    size: [lit(0.5 * mm), lit(0.4 * mm), lit(0.2 * mm)],
                    turn: fieldform_scene::no_turn(),
                })
            };
            let (a, b) = (slab(&mut graph, -0.15), slab(&mut graph, 0.15));
            let body = graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet(0.04) });
            let boss = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.12 * mm), lit(0.03 * mm), lit(0.0)]),
                radius: lit(0.09 * mm),
            });
            let bossed = graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: body.into(),
                b: boss.into(),
                fillet: fillet(0.05),
            });
            // Turned on two axes, so nothing under it is aligned with the world the gap is
            // measured in.
            let placed = graph.add(Node::Transform {
                input: bossed.into(),
                translation: [lit(0.02 * mm), lit(-0.01 * mm), lit(0.03 * mm)],
                rotation: [lit(17.0), lit(-11.0), lit(29.0)],
                scale: lit(1.0),
                about: fieldform_scene::Place::origin(),
            });
            let bore = graph.add(Node::Cylinder { edge: Default::default(),
                ends: fieldform_scene::Ends::between(
                    fieldform_scene::Place::At([lit(0.03 * mm), lit(-0.02 * mm), lit(-0.5 * mm)]),
                    fieldform_scene::Place::At([lit(0.03 * mm), lit(-0.02 * mm), lit(0.5 * mm)]),
                ),
                radius: lit(0.06 * mm),
            });
            graph.add(Node::Blend {
                op: BooleanOp::Difference,
                a: placed.into(),
                b: bore.into(),
                fillet: fillet(0.035),
            });
            graph.mark_critical(fieldform_scene::Critical {
                clearance: lit(0.01 * mm),
                ..fieldform_scene::Critical::new(bore)
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::Attracted) {
            let mut graph = Graph::empty();
            // Long across the curve rather than along it, so the pull varies over the body:
            // a bar sitting entirely inside the falloff would be displaced almost uniformly,
            // which is a translation and would look the same whatever the falloff shape was.
            let bar = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(CAGE_MM * 0.45),
                    ParamRef::Literal(0.0),
                ]),
                size: [
                    ParamRef::Literal(CAGE_MM * 0.8),
                    ParamRef::Literal(CAGE_MM * 0.7),
                    ParamRef::Literal(CAGE_MM * 0.35),
                ],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Attract {
                input: bar.into(),
                toward: fieldform_scene::Attractor::Curve(curve_id),
                // Strong enough to be unmistakable and clear of folding, and a reach that
                // covers part of the bar rather than all of it.
                strength: ParamRef::Literal(0.9),
                falloff: ParamRef::Literal(CAGE_MM * 0.5),
            });
            // The spine, because the shader is handed one polyline per curve the graph names
            // and an attractor names one exactly as a pipe does. Returning nothing here is
            // what an empty list means, and it does not fail: the span arrives with a count of
            // zero, `attract_point` finds no curve to be drawn toward, and the body renders
            // unwarped and perfectly plausible. It cost an afternoon.
            return (doc, graph, vec![spine]);
        }

        if matches!(self, Scene::Twisted) {
            let mut graph = Graph::empty();
            // Rectangular in section, so the winding is visible: a body with a circular section
            // twists into itself and would look identical however wrong the rate was.
            let bar = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                size: [
                    ParamRef::Literal(CAGE_MM * 0.55),
                    ParamRef::Literal(CAGE_MM * 0.18),
                    ParamRef::Literal(CAGE_MM * 0.9),
                ],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Twist {
                input: bar.into(),
                // About a quarter turn over the bar's height.
                degrees_per_mm: ParamRef::Literal(90.0 / (CAGE_MM * 0.9)),
                // Deliberately not the Z axis through the origin, so a shader that ignored
                // either the axis or the point would still look plausible and be measurably
                // wrong against the CPU.
                frame: fieldform_scene::Frame::at_along(fieldform_scene::Axis::Y, [
                    ParamRef::Literal(CAGE_MM * 0.05),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(-CAGE_MM * 0.07),
                ]),
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::Mirrored | Scene::MirroredAboutAPlane) {
            let mut graph = Graph::empty();
            // Off-centre and asymmetric, so a reflection that landed in the wrong place or
            // flipped the wrong axis could not pass by symmetry.
            let cone = graph.add(Node::Cone { edge: Default::default(),
                ends: fieldform_scene::Ends::between(
                    fieldform_scene::Place::At([
                        ParamRef::Literal(CAGE_MM * 0.15),
                        ParamRef::Literal(CAGE_MM * 0.1),
                        ParamRef::Literal(-CAGE_MM * 0.3),
                    ]),
                    fieldform_scene::Place::At([
                        ParamRef::Literal(CAGE_MM * 0.45),
                        ParamRef::Literal(-CAGE_MM * 0.05),
                        ParamRef::Literal(CAGE_MM * 0.25),
                    ])),
                base_radius: ParamRef::Literal(CAGE_MM * 0.2),
                top_radius: ParamRef::Literal(CAGE_MM * 0.06),
            });
            // One node, sharing the cone. A joined mirror evaluates its input at two points
            // — where the body is and where its reflection comes from — which is the case
            // shader generation used to refuse, and the reason a mirror was two blocks. The
            // union is inside the node now, so this scene is the regression test for both.
            let frame = match self {
                Scene::MirroredAboutAPlane => {
                    let plane = graph.add(Node::Plane {
                        origin: fieldform_scene::Place::At([
                            ParamRef::Literal(CAGE_MM * 0.08),
                            ParamRef::Literal(-CAGE_MM * 0.04),
                            ParamRef::Literal(CAGE_MM * 0.06),
                        ]),
                        flipped: false,
                        turn: fieldform_scene::Turn::By([
                            ParamRef::Literal(18.0),
                            ParamRef::Literal(-62.0),
                            ParamRef::Literal(0.0),
                        ]),
                    });
                    fieldform_scene::Frame::From { node: plane }
                }
                _ => fieldform_scene::Frame::at_along(
                    fieldform_scene::Axis::X,
                    [ParamRef::Literal(0.0); 3],
                ),
            };
            graph.add(Node::Mirror { input: cone.into(), frame, join: true });
            return (doc, graph, Vec::new());
        }

        if matches!(
            self,
            Scene::CylindricalInfill
                | Scene::SphericalInfill
                | Scene::SolidInfill
                | Scene::SchoenIwpInfill
                | Scene::NeoviusInfill
                | Scene::FischerKochSInfill
                | Scene::CubicBeamInfill
                | Scene::BccBeamInfill
                | Scene::FccBeamInfill
                | Scene::OctetBeamInfill
                | Scene::DiamondBeamInfill
                | Scene::VoronoiFoamInfill
                | Scene::VoronoiBeamInfill
        ) {
            let mut graph = Graph::empty();
            // A tube: a cylinder with a smaller one taken out, so the lattice has a genuine
            // radial range to stretch across rather than a solid core it would only fill.
            let outer = graph.add(Node::Cylinder { edge: Default::default(),
                ends: fieldform_scene::Ends::between(
                    fieldform_scene::Place::At([
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(-CAGE_MM * 0.35),
                    ]),
                    fieldform_scene::Place::At([
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(CAGE_MM * 0.35),
                    ])),
                radius: ParamRef::Literal(CAGE_MM * 0.4),
            });
            let bore = graph.add(Node::Cylinder { edge: Default::default(),
                ends: fieldform_scene::Ends::between(
                    fieldform_scene::Place::At([
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(-CAGE_MM * 0.4),
                    ]),
                    fieldform_scene::Place::At([
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(0.0),
                        ParamRef::Literal(CAGE_MM * 0.4),
                    ])),
                radius: ParamRef::Literal(CAGE_MM * 0.18),
            });
            let tube = graph.add(Node::Boolean {
                op: BooleanOp::Difference,
                inputs: vec![outer.into(), bore.into()],
            });
            let lattice = graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell: match self {
                    Scene::SchoenIwpInfill => fieldform_scene::LatticeCell::SchoenIwp,
                    Scene::NeoviusInfill => fieldform_scene::LatticeCell::Neovius,
                    Scene::FischerKochSInfill => fieldform_scene::LatticeCell::FischerKochS,
                    Scene::CubicBeamInfill => fieldform_scene::LatticeCell::Cubic,
                    Scene::BccBeamInfill => fieldform_scene::LatticeCell::Bcc,
                    Scene::FccBeamInfill => fieldform_scene::LatticeCell::Fcc,
                    Scene::OctetBeamInfill => fieldform_scene::LatticeCell::Octet,
                    Scene::DiamondBeamInfill => fieldform_scene::LatticeCell::Diamond,
                    // Full jitter, which is where the seeds reach the edge of their own cells
                    // and the wider ring can fire.
                    Scene::VoronoiFoamInfill => fieldform_scene::LatticeCell::VoronoiFoam {
                        jitter: ParamRef::Literal(1.0),
                        seed: 0x5EED_1234,
                    },
                    Scene::VoronoiBeamInfill => fieldform_scene::LatticeCell::VoronoiBeam {
                        jitter: ParamRef::Literal(1.0),
                        seed: 0x5EED_1234,
                    },
                    _ => fieldform_scene::LatticeCell::Gyroid,
                },
                map: if matches!(self, Scene::SphericalInfill) {
                    fieldform_scene::CellMap::Spherical {
                        cell_radius: ParamRef::Literal(CAGE_MM * 0.09),
                        azimuth_count: 20,
                        polar_count: 10,
                    }
                } else {
                    fieldform_scene::CellMap::Cylindrical {
                        cell_radius: ParamRef::Literal(CAGE_MM * 0.09),
                        cell_height: ParamRef::Literal(CAGE_MM * 0.09),
                        arc_count: 24,
                    }
                },
                phase: match self {
                    // A density rather than a wall, and nothing else about the scene changes —
                    // same cell, same map, same tube — so a residual here is the phase and
                    // cannot be anything else.
                    Scene::SolidInfill => {
                        LatticePhase::Solid { density: ParamRef::Literal(0.35) }
                    }
                    // A radius in millimetres, used directly — no isovalue anywhere on this
                    // path, which is the whole of what these two scenes are checking.
                    Scene::CubicBeamInfill
                    | Scene::BccBeamInfill
                    | Scene::FccBeamInfill
                    | Scene::OctetBeamInfill
                    | Scene::DiamondBeamInfill
                    | Scene::VoronoiBeamInfill => {
                        LatticePhase::Strut { radius: ParamRef::Literal(CAGE_MM * 0.012) }
                    }
                    _ => LatticePhase::sheet(ParamRef::Literal(CAGE_MM * 0.02)),
                },
            });
            // Intersection, never a blend: a lattice's field is a lower bound and the graph
            // refuses to blend one.
            graph.add(Node::Boolean {
                op: BooleanOp::Intersection,
                inputs: vec![tube.into(), lattice.into()],
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::ConformalInfill | Scene::ConformalShell) {
            // A domed panel: a quad grid bent so no two faces share a plane, which is what makes
            // the interpolated normal differ from any triangle's.
            let n = 4usize;
            let span = CAGE_MM * 0.8;
            let rise = CAGE_MM * 0.05;
            let step = span / n as f64;
            let mut positions = Vec::new();
            for j in 0..=n {
                for i in 0..=n {
                    let (x, y) = (i as f64 * step - span * 0.5, j as f64 * step - span * 0.5);
                    let (u, v) = (x / (span * 0.5), y / (span * 0.5));
                    positions.push(Point3::new(x, y, rise * (1.0 - u * u) * (1.0 - v * v)));
                }
            }
            let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
            let mut faces = Vec::new();
            for j in 0..n {
                for i in 0..n {
                    faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
                }
            }
            let surface = doc.add_quad_mesh(
                fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a domed panel"),
                "dome",
            );

            let mut graph = Graph::empty();
            let shelled = matches!(self, Scene::ConformalShell);
            // The slab the lattice is confined to. A conformal lattice is unbounded like any
            // other -- it is a pattern through all of space, banded about a surface -- so it
            // becomes a part by being intersected with one.
            //
            // **Built only where it is used.** A node nothing consumes is a body, so adding the
            // slab and then returning early for the shelled scene left a solid block unioned
            // with the lattice that scene exists to show alone. Its own notes said the image
            // was blank and that it was not building what it thought it was; this is what.
            let slab = (!shelled).then(|| graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(0.0),
                    ParamRef::Literal(rise * 0.5),
                ]),
                size: [
                    ParamRef::Literal(span * 0.9),
                    ParamRef::Literal(span * 0.9),
                    ParamRef::Literal(rise * 2.0),
                ],
                turn: fieldform_scene::no_turn(),
            }));
            let lattice = graph.add(Node::Lattice {
                frame: fieldform_scene::Frame::default(),
                cell: fieldform_scene::LatticeCell::Gyroid,
                map: fieldform_scene::CellMap::Conformal {
                    surface,
                    cells_per_face: 2,
                    depth: ParamRef::Literal(CAGE_MM * 0.12),
                    // Unconfined in the infill scene on purpose: that one measures the *tiling*
                    // against a slab, which is the arm with no shell in it.
                    // **Thin, and for the reason the dome is gentle.** A shell wraps the
                    // panel's rim as well as its faces, and the rim is where a closest-point
                    // map is weakest -- `s` and `t` are clamped there and the nearest point is
                    // an edge. A shell of 160 mm around a 50 mm rise is mostly rim, and the
                    // scene then measures that rather than the shader: 8.3 mm against 0.7 for
                    // the unconfined arm. This one hugs the surface.
                    thickness: shelled.then(|| ParamRef::Literal(CAGE_MM * 0.05)),
                },
                phase: LatticePhase::sheet(ParamRef::Literal(CAGE_MM * 0.02)),
            });
            if shelled {
                // Alone, because a shelled conformal lattice has a box of its own. Intersecting
                // it with the slab would hide exactly the faces this scene is here to check.
                return (doc, graph, Vec::new());
            }
            graph.add(Node::Boolean {
                op: BooleanOp::Intersection,
                inputs: vec![(slab.expect("built above for the unshelled scene")).into(), lattice.into()],
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::Hollow) {
            let mut graph = Graph::empty();
            let cube = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
                size: [ParamRef::Literal(CAGE_MM * 0.7); 3],
                turn: fieldform_scene::no_turn(),
            });
            // Through `Node::Shell` rather than by hand, because the node is what the app
            // builds now and the lowering that turns it back into an offset and a difference
            // is the thing that could be wrong. The body is *shared* by both arms — one node
            // read twice at the same point, so one SSA slot rather than the whole cube twice.
            //
            // A wall thick enough to trace through at this resolution and thin enough that the
            // cavity is most of the body — so a shader that ignored the hollowing entirely
            // would draw a solid cube and be caught by the residual rather than by a subtlety.
            graph.add(Node::Shell {
                input: cube.into(),
                thickness: ParamRef::Literal(CAGE_MM * 0.08),
            });
            return (doc, graph, Vec::new());
        }

        if matches!(self, Scene::TwoLegChamfers) {
            let mut graph = Graph::empty();
            let mm = |f: f64| ParamRef::Literal(f * CAGE_MM);
            // The sphere into the block of `BlendDriven`, chamfered with a leg twice as long along
            // the block as along the sphere, so the bevel's two intercepts are read apart.
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([mm(0.25), mm(0.0), mm(0.0)]),
                radius: mm(0.35),
            });
            let block = graph.add(Node::Cuboid {
                edge: Default::default(),
                centre: fieldform_scene::Place::At([mm(-0.2), mm(0.0), mm(0.0)]),
                size: [mm(0.6), mm(0.5), mm(0.5)],
                turn: fieldform_scene::no_turn(),
            });
            graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: ball.into(),
                b: block.into(),
                fillet: Fillet {
                    radius: mm(0.05),
                    continuity: Continuity::Chamfer,
                    second: Some(mm(0.1)),
                },
            });
            return (doc, graph, Vec::new());
        }
        if matches!(self, Scene::BlendDriven) {
            let mut graph = Graph::empty();
            let mm = |f: f64| ParamRef::Literal(f * CAGE_MM);
            // A sphere standing into a block, so the seam runs right round and up both sides;
            // the ramp reads a plane under the pair, tight at the bottom and generous at the
            // top, so the divisor the two paths must share is a real number rather than one.
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([mm(0.25), mm(0.0), mm(0.0)]),
                radius: mm(0.35),
            });
            let block = graph.add(Node::Cuboid {
                edge: Default::default(),
                centre: fieldform_scene::Place::At([mm(-0.2), mm(0.0), mm(0.0)]),
                size: [mm(0.6), mm(0.5), mm(0.5)],
                turn: fieldform_scene::no_turn(),
            });
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At([mm(0.0), mm(0.0), mm(-0.35)]),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let ramp = graph.add(Node::Ramp {
                input: plane.into(),
                in_min: mm(0.0),
                in_max: mm(0.7),
                out_min: mm(0.02),
                out_max: mm(0.14),
                shape: fieldform_scene::RampShape::Smooth,
            });
            graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: ball.into(),
                b: block.into(),
                fillet: Fillet { radius: ParamRef::Field(ramp), continuity: Continuity::G1, second: None },
            });
            return (doc, graph, Vec::new());
        }
        if matches!(self, Scene::Morphed | Scene::MorphedDriven) {
            let mut graph = Graph::empty();
            let mm = |f: f64| ParamRef::Literal(f * CAGE_MM);
            // **In the same place**, which is the arrangement a constant morph is for: two
            // bodies apart from each other average to nothing in between, and a scene of
            // nothing measures nothing.
            let centre = fieldform_scene::Place::At([mm(0.0), mm(0.0), mm(0.0)]);
            let ball = graph.add(Node::Sphere { centre, radius: mm(0.45) });
            // Sized so neither operand dominates: a cuboid whose corners reach past the
            // sphere and whose faces fall inside it, so the averaged surface is somewhere new
            // rather than nearly one of the two.
            let block = graph.add(Node::Cuboid { edge: Default::default(),
                centre,
                size: [mm(0.7), mm(0.7), mm(0.7)],
                turn: fieldform_scene::no_turn(),
            });
            // Not a half, deliberately: an even mix would be symmetric in the two operands and
            // could not tell a swapped pair from a correct one.
            let t = match self {
                // Driven by distance from a point off centre, so the transition is a region
                // rather than a plane and the ramp's own input is exercised.
                Scene::MorphedDriven => {
                    let point = graph.add(Node::Point {
                        at: [mm(0.15), mm(-0.1), mm(0.05)],
                    });
                    let ramp = graph.add(Node::Ramp {
                        input: point.into(),
                        in_min: mm(0.1),
                        in_max: mm(0.5),
                        out_min: ParamRef::Literal(0.0),
                        out_max: ParamRef::Literal(1.0),
                        shape: fieldform_scene::RampShape::Smooth,
                    });
                    ParamRef::Field(ramp)
                }
                _ => ParamRef::Literal(0.35),
            };
            graph.add(Node::Morph { a: ball.into(), b: block.into(), t });
            return (doc, graph, Vec::new());
        }
        if matches!(self, Scene::SkewedBoxes) {
            let mut graph = Graph::empty();
            let at = |x: f64, y: f64, z: f64| {
                [
                    ParamRef::Literal(x * CAGE_MM),
                    ParamRef::Literal(y * CAGE_MM),
                    ParamRef::Literal(z * CAGE_MM),
                ]
            };
            let mm = |f: f64| ParamRef::Literal(f * CAGE_MM);
            let edge = |x: f64, y: f64, z: f64| [mm(x), mm(y), mm(z)];
            // Edges that line up with nothing and are not at right angles to one another, so a
            // shader that quietly treated the box as square would be wrong everywhere but the
            // control, and the sign of the determinant is positive.
            let leaning = graph.add(Node::SkewedBox {
                centre: fieldform_scene::Place::At(at(-0.3, -0.1, 0.0)),
                edges: [edge(0.5, 0.0, 0.0), edge(0.2, 0.4, 0.0), edge(0.15, 0.1, 0.35)],
            });
            // Swapped so the determinant is negative: -0.034 in these units, and the box is as
            // solid as the first.
            let left = graph.add(Node::SkewedBox {
                centre: fieldform_scene::Place::At(at(0.0, 0.0, 0.0)),
                edges: [edge(0.3, 0.1, 0.05), edge(0.1, 0.0, 0.35), edge(-0.1, 0.3, 0.05)],
            });
            let placed = graph.add(Node::Transform {
                input: left.into(),
                translation: at(0.42, 0.25, 0.0),
                rotation: [
                    ParamRef::Literal(31.0),
                    ParamRef::Literal(-22.0),
                    ParamRef::Literal(17.0),
                ],
                scale: ParamRef::Literal(1.3),
                // About a point that is not the origin, for the reason `RoundSolids` gives.
                about: fieldform_scene::Place::At(at(0.1, 0.2, -0.1)),
            });
            let ball = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At(at(-0.1, 0.1, 0.35)),
                radius: mm(0.22),
            });
            let rounded = graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: leaning.into(),
                b: ball.into(),
                fillet: Fillet::of(ParamRef::Literal(40.0)),
            });
            graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![rounded.into(), placed.into()] });
            return (doc, graph, Vec::new());
        }
        if matches!(self, Scene::RoundSolids) {
            let mut graph = Graph::empty();
            let at = |x: f64, y: f64, z: f64| {
                [
                    ParamRef::Literal(x * CAGE_MM),
                    ParamRef::Literal(y * CAGE_MM),
                    ParamRef::Literal(z * CAGE_MM),
                ]
            };
            let mm = |f: f64| ParamRef::Literal(f * CAGE_MM);
            // Ends that share no axis with anything, so the segment frame is exercised in a
            // general orientation rather than one that happens to line up with X, Y or Z. A
            // frame built wrongly still looks right on an axis-aligned shape.
            let cylinder = graph.add(Node::Cylinder { edge: Default::default(),
                ends: fieldform_scene::Ends::between(fieldform_scene::Place::At(at(-0.55, -0.1, -0.25)), fieldform_scene::Place::At(at(-0.3, 0.15, 0.3))),
                radius: mm(0.15),
            });
            // Truncated rather than pointed, so both radii are exercised — an apex is the one
            // case where getting the top radius wrong still looks like a cone.
            let cone = graph.add(Node::Cone { edge: Default::default(),
                ends: fieldform_scene::Ends::between(fieldform_scene::Place::At(at(0.35, -0.2, -0.25)), fieldform_scene::Place::At(at(0.55, 0.2, 0.3))),
                base_radius: mm(0.22),
                top_radius: mm(0.07),
            });
            let torus = graph.add(Node::Torus {
                centre: fieldform_scene::Place::At(at(0.0, 0.45, 0.0)),
                major_radius: mm(0.25),
                minor_radius: mm(0.08),
                turn: fieldform_scene::no_turn(),
            });
            // Different radii at the two ends, so the tangent cone between the caps is
            // exercised and not just the equal-radius tube it degenerates to.
            let capsule = graph.add(Node::Capsule {
                ends: fieldform_scene::Ends::between(fieldform_scene::Place::At(at(-0.15, 0.0, -0.2)), fieldform_scene::Place::At(at(0.2, 0.05, 0.25))),
                start_radius: mm(0.13),
                end_radius: mm(0.05),
            });
            // Angles that line up with nothing, so a transposed rotation or a swapped axis
            // cannot pass by symmetry.
            let placed = graph.add(Node::Transform {
                input: capsule.into(),
                translation: at(0.0, -0.45, 0.05),
                rotation: [
                    ParamRef::Literal(37.0),
                    ParamRef::Literal(24.0),
                    ParamRef::Literal(-16.0),
                ],
                scale: ParamRef::Literal(1.4),
                // **Turned and scaled about a point that is not the origin**, which is the one
                // arrangement where the shader and the field each have to fold the pivot into
                // the translation for themselves. With the pivot at the origin the two forms
                // are the same expression and this measurement would say nothing.
                about: fieldform_scene::Place::At(at(0.18, -0.3, 0.12)),
            });
            // A half-space cutting the lot: the case the extent work was for, and the one
            // that proves an infinite operand still traces to a finite surface.
            // Flipped, so the parameter that carries which side is solid is exercised: an
            // unflipped plane would look the same whether the sign reached the shader or not.
            let plane = graph.add(Node::Plane {
                origin: fieldform_scene::Place::At(at(0.0, 0.0, -0.22)),
                flipped: true,
                turn: fieldform_scene::no_turn(),
            });
            let a = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![cylinder.into(), cone.into()] });
            let b = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), torus.into()] });
            let all = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![b.into(), placed.into()] });
            graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![all.into(), plane.into()] });
            return (doc, graph, Vec::new());
        }

        let mut graph = if matches!(self, Scene::PipeOnly) {
            Graph::empty()
        } else {
            Graph::from_cage(cage_id)
        };
        if matches!(self, Scene::CageOnly) {
            return (doc, graph, Vec::new());
        }

        // Absent only for the cageless scene, which never reaches the arms that use it.
        let cage_node = graph.roots().first().copied().unwrap_or_else(NodeId::new);
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(PIPE_RADIUS_MM),
        });
        match self {
            // Returned above with a graph of its own, so it never reaches this pipe or this cage.
            Scene::SpokedPipes
            | Scene::SpokedBlades
            | Scene::SpokedRibs
            // Each returned its twin's document before anything here was built.
            | Scene::ThinBaked
            | Scene::ThinTubeBaked
            | Scene::ThinCutBaked
            | Scene::SpokedRibsBaked
            | Scene::ThinRevolvedBaked
            | Scene::ThinRevolvedAcrossBaked
            | Scene::ThinSweptBaked
            // And these returned a graph of their own, as the other thin scenes do.
            | Scene::ThinRevolved
            | Scene::ThinRevolvedAcross
            | Scene::ThinRevolvedWedge
            | Scene::RevolvedWedge
            | Scene::ThinSwept
            | Scene::ThinSweptScaled
            | Scene::SharpDraft
            | Scene::DraftedWallSweep
            | Scene::CageOnly
            | Scene::ConformalInfill
            | Scene::ConformalShell
            | Scene::Primitives
            | Scene::RoundedBoxCut
            | Scene::G2BoxCut
            | Scene::ChamferedBoxCut
            | Scene::EdgedCylinders
            | Scene::EdgedCones
            | Scene::G2Cones
            | Scene::RimmedBlendedToChamferedBox
            | Scene::ManyWayDifference
            | Scene::ManyWayUnion
            | Scene::RoundSolids
            | Scene::SkewedBoxes
            | Scene::Morphed
            | Scene::MorphedDriven
            | Scene::BlendDriven
            | Scene::TwoLegChamfers
            | Scene::TwoLegRims
            | Scene::GyroidInfill
            | Scene::SchwarzPInfill
            | Scene::SchwarzDInfill
            | Scene::SphereInfill
            | Scene::WalledInfill
            | Scene::HoneycombInfill
            | Scene::GradedWall
            | Scene::GradedDensity
            | Scene::GradedStrut
            // Both returned above, before any of this: their geometry comes from a curve, so
            // they build nothing from the cage and the pipe this arm is about.
            | Scene::Extruded
            | Scene::Leaning
            | Scene::Upright
            | Scene::BakedProfile
            | Scene::Drafted
            | Scene::DraftedBake
            | Scene::DraftedFlip
            | Scene::Symmetric
            | Scene::SymmetricDrafted
            | Scene::TwoSidedBegun
            | Scene::RimmedDrafted
            | Scene::RimmedTwoSided
            | Scene::ChamferRimmedWall
            | Scene::RimmedLoft
            | Scene::EndRimmedTwoSided
            | Scene::StartRimmedLoft
            | Scene::EndRimmedSweep
            | Scene::RimmedWedge
            | Scene::StartRimmedThinWedge
            | Scene::RimmedSweep
            | Scene::ChamferRimmedTaper
            | Scene::Revolved
            | Scene::ThinExtruded
            | Scene::ThinLeaning
            | Scene::ThinTube
            | Scene::ThinRib
            | Scene::ThinCut
            | Scene::Swept
            | Scene::SweptFlush
            | Scene::SweptParallel
            | Scene::SweptScaled
            | Scene::SweptDrafted
            | Scene::SweptTwistedWindow
            | Scene::Lofted
            | Scene::LoftedG0
            | Scene::LoftedDerived
            | Scene::Hollow
            | Scene::CylindricalInfill
            | Scene::SphericalInfill
            | Scene::SolidInfill
            | Scene::SchoenIwpInfill
            | Scene::NeoviusInfill
            | Scene::FischerKochSInfill
            | Scene::CubicBeamInfill
            | Scene::BccBeamInfill
            | Scene::FccBeamInfill
            | Scene::OctetBeamInfill
            | Scene::DiamondBeamInfill
            | Scene::VoronoiFoamInfill
            | Scene::VoronoiBeamInfill
            | Scene::Mirrored
            | Scene::MirroredAboutAPlane
            | Scene::Twisted
            | Scene::Attracted
            | Scene::Patterned
            | Scene::PatternedRow
            | Scene::PatternedRing
            | Scene::PatternedFillet
            | Scene::G2PatternedFillet
            | Scene::PatternedRun
            | Scene::PatternedArc
            | Scene::PatternedArcFillet
            | Scene::PatternedLongRun
            | Scene::PatternedLongArc
            | Scene::PatternedGrid
            | Scene::PatternedLongGrid
            | Scene::VaryingOffset
            | Scene::ProtectedBore => unreachable!("returned above"),
            Scene::PipeOnly => pipe,
            // Nothing joins them: both are bodies, and the part's field is their union.
            Scene::TwoBodies => pipe,
            Scene::Difference => graph.add(Node::Boolean {
                op: BooleanOp::Difference,
                inputs: vec![cage_node.into(), pipe.into()],
            }),
            Scene::Blend => graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: cage_node.into(),
                b: pipe.into(),
                fillet: Fillet { radius: ParamRef::Literal(BLEND_RADIUS_MM), continuity: Continuity::G1, second: None },
            }),
            Scene::BlendedDifference => graph.add(Node::Blend {
                op: BooleanOp::Difference,
                a: cage_node.into(),
                b: pipe.into(),
                fillet: Fillet { radius: ParamRef::Literal(BLEND_RADIUS_MM), continuity: Continuity::G1, second: None },
            }),
            Scene::G2Blend => graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: cage_node.into(),
                b: pipe.into(),
                fillet: Fillet { radius: ParamRef::Literal(BLEND_RADIUS_MM), continuity: Continuity::G2, second: None },
            }),
            Scene::G2BlendedDifference => graph.add(Node::Blend {
                op: BooleanOp::Difference,
                a: cage_node.into(),
                b: pipe.into(),
                fillet: Fillet { radius: ParamRef::Literal(BLEND_RADIUS_MM), continuity: Continuity::G2, second: None },
            }),
            Scene::ChamferBlend => graph.add(Node::Blend {
                op: BooleanOp::Union,
                a: cage_node.into(),
                b: pipe.into(),
                fillet: Fillet { radius: ParamRef::Literal(BLEND_RADIUS_MM), continuity: Continuity::Chamfer, second: None },
            }),
            Scene::ChamferedDifference => graph.add(Node::Blend {
                op: BooleanOp::Difference,
                a: cage_node.into(),
                b: pipe.into(),
                fillet: Fillet { radius: ParamRef::Literal(BLEND_RADIUS_MM), continuity: Continuity::Chamfer, second: None },
            }),
        };

        (doc, graph, vec![spine])
    }
}

/// The box the cage is baked over: the cage's own extent with a margin.
///
/// Not the composed bounds. Only the cage is baked — the pipe is closed-form and evaluated
/// exactly in the shader — so stretching the box to cover a curve that runs well outside the
/// cage spends most of the grid on empty space and coarsens the one thing in it. The margin
/// keeps the surface clear of the faces, which is what the out-of-box bound assumes.
fn bake_bounds() -> (Point3<f64>, Point3<f64>) {
    let h = CAGE_MM * 0.5 * 1.15;
    (Point3::new(-h, -h, -h), Point3::new(h, h, h))
}

/// What the camera frames: the cage, plus the pipe running out either side of it.
fn model_bounds() -> (Point3<f64>, Point3<f64>) {
    let h = CAGE_MM * 0.75;
    (Point3::new(-h, -h, -h), Point3::new(h, h, h))
}

/// Worst distance from the CPU surface over every GPU hit, and how many there were.
fn residual(
    cpu: &GraphField,
    camera: &Camera,
    hits: &[fieldform_gpu::Hit],
    width: u32,
    height: u32,
) -> (f64, usize, Point3<f64>, f64) {
    let mut worst: f64 = 0.0;
    let mut count = 0;
    let mut worst_at = Point3::origin();
    let mut worst_signed = 0.0;
    for (i, hit) in hits.iter().enumerate() {
        if !hit.is_hit() {
            continue;
        }
        let (px, py) = ((i as u32 % width), (i as u32 / width));
        let u = (px as f64 + 0.5) / width as f64 * 2.0 - 1.0;
        let v = 1.0 - (py as f64 + 0.5) / height as f64 * 2.0;
        let dir = (camera.forward
            + camera.right * (u * camera.half_height * camera.aspect)
            + camera.up * (v * camera.half_height))
            .normalize();
        let at = camera.eye + dir * hit.depth;
        let signed = cpu.distance(at);
        if signed.abs() > worst {
            worst = signed.abs();
            worst_at = at;
            worst_signed = signed;
        }
        count += 1;
    }
    (worst, count, worst_at, worst_signed)
}

/// Asks the CPU field how far each GPU hit point is from the surface.
fn report(
    cpu: &GraphField,
    camera: &Camera,
    hits: &[fieldform_gpu::Hit],
    width: u32,
    height: u32,
    label: &str,
) {
    let mut residuals: Vec<f64> = Vec::new();
    // How far the shader's carried normal is from the field's own gradient, in degrees.
    let mut angles: Vec<f64> = Vec::new();
    let mut pixels: Vec<[u8; 3]> = Vec::with_capacity(hits.len());
    const BACKGROUND: [u8; 3] = [24, 26, 30];
    let light = (camera.forward * -1.0 + camera.up * 0.6 + camera.right * 0.35).normalize();

    for (i, hit) in hits.iter().enumerate() {
        if !hit.is_hit() {
            pixels.push(BACKGROUND);
            continue;
        }
        let (px, py) = ((i as u32 % width), (i as u32 / width));
        let u = (px as f64 + 0.5) / width as f64 * 2.0 - 1.0;
        let v = 1.0 - (py as f64 + 0.5) / height as f64 * 2.0;
        let dir = (camera.forward
            + camera.right * (u * camera.half_height * camera.aspect)
            + camera.up * (v * camera.half_height))
            .normalize();
        let p = camera.eye + dir * hit.depth;

        residuals.push(cpu.distance(p).abs());

        // **And which way it faces, which is a different question.** The residual above asks
        // where the surface is; a normal wrong by degrees puts it in exactly the right place
        // and lights it as a different shape. That gap let a pattern's normal be turned by a
        // whole number of instances too far -- and by a different number either side of a cell
        // boundary, so the shading creased along the middle of every fillet while an STL of the
        // same body came out smooth. Nothing here looked, because nothing here was asking.
        //
        // The field's own gradient by central difference is the independent answer. A step of
        // a hundredth of a millimetre is small against any feature and large against `f32`.
        let h = 0.01;
        let grad = nalgebra::Vector3::new(
            cpu.distance(p + nalgebra::Vector3::new(h, 0.0, 0.0))
                - cpu.distance(p - nalgebra::Vector3::new(h, 0.0, 0.0)),
            cpu.distance(p + nalgebra::Vector3::new(0.0, h, 0.0))
                - cpu.distance(p - nalgebra::Vector3::new(0.0, h, 0.0)),
            cpu.distance(p + nalgebra::Vector3::new(0.0, 0.0, h))
                - cpu.distance(p - nalgebra::Vector3::new(0.0, 0.0, h)),
        );
        if let Some(grad) = grad.try_normalize(1e-9) {
            angles.push(
                hit.normal.dot(&grad).clamp(-1.0, 1.0).acos().to_degrees(),
            );
        }

        let shade = 0.15 + 0.85 * hit.normal.dot(&light).max(0.0);
        let c = (shade.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        pixels.push([c, c, c]);
    }

    // A label is prose and may hold a colon, which the development machine's file system reads
    // as the start of an alternate data stream: the picture lands hidden behind an empty file
    // named for the first half of the label. So every character a file name cannot hold
    // becomes a dash too.
    let file_safe: String = label
        .chars()
        .map(|c| if c == ' ' || r#"<>:"/\|?*"#.contains(c) { '-' } else { c })
        .collect();
    let name = format!("gpu-cpu-{file_safe}.png");
    let mut buf = Vec::with_capacity(pixels.len() * 3);
    for p in &pixels {
        buf.extend_from_slice(p);
    }
    image::save_buffer(&name, &buf, width, height, image::ColorType::Rgb8).expect("writable");

    if residuals.is_empty() {
        println!("  the GPU hit nothing at all — the shader is not finding the model");
        return;
    }
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let p99 = residuals[residuals.len() * 99 / 100];
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "  {} hits of {} pixels ({:.1}%)",
        residuals.len(),
        hits.len(),
        residuals.len() as f64 / hits.len() as f64 * 100.0
    );
    if !angles.is_empty() {
        println!(
            "  the shader's normal against the field's gradient: mean {:.3} deg, \
             99th pct {:.3} deg, worst {:.3} deg",
            angles.iter().sum::<f64>() / angles.len() as f64,
            angles[angles.len() * 99 / 100],
            angles[angles.len() - 1],
        );
    }
    println!(
        "  distance from the CPU surface at those points: mean {mean:.4} mm, \
         99th pct {p99:.4} mm, worst {:.4} mm",
        residuals.last().unwrap()
    );
    println!("  wrote {name}");
}
