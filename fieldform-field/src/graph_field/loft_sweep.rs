// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a loft and a sweep are built from, read off the graph: sections, law, frame and stretch.

use super::{CompileError, rotation_from_z, spine_start, sweep_outline_section, wall_in_plane};
use fieldform_scene::Document;
use nalgebra::{Point3, Vector3};

/// Whether a wall carried under `law` may answer from a grid.
///
/// **A grid holds one wall, and a law makes a different one at every station.** A profile's
/// bake survives a law: a scale reads it as `s · d(q/s)` and a draft as a contour of it, which
/// are the same grid asked differently. A wall's does not, because a wall is *not* a function
/// of its curve's distance alone -- an open chain's mitres and end faces are built for one
/// thickness on one curve, and a loop's band is a fold whose middle the draft moves, where a
/// grid of the folded wall has nothing left to fold. So a wall under a law is walked, on both
/// sides: the shader's `wall_sweep_law_on` reads the loop's own distance and folds it itself,
/// and would read a wall's grid as a curve's if one were there.
///
/// One rule, asked by the compile and by `outline_wall_baked`, so the field and the polylines
/// the shader walks cannot disagree about whether there is a grid.
pub fn wall_bakes_under(law: &fieldform_scene::SweepLaw) -> bool {
    law.is_plain()
}

/// The law a sweep's section follows, with its numbers asked of the document: the kernel's
/// [`crate::SectionLaw`] from the block's [`fieldform_scene::SweepLaw`]. Degrees become
/// radians and a draft its tangent here, percentages of the path become fractions, and a
/// number that cannot be resolved is refused as any other is. What the law then refuses -- a
/// scale of nought, a draft that closes the section, a twist on a parallel section, an empty
/// window -- it refuses by name when the leaf is built.
pub fn sweep_law(
    document: &Document,
    law: &fieldform_scene::SweepLaw,
) -> Result<crate::SectionLaw, CompileError> {
    let resolve = |r: &fieldform_scene::ParamRef| {
        document.resolve(r).ok_or(CompileError::UnresolvedRadius)
    };
    let taper = match &law.taper {
        fieldform_scene::Taper::None => crate::Taper::None,
        fieldform_scene::Taper::Scale(v) => crate::Taper::Scale { end: resolve(v)? },
        fieldform_scene::Taper::Draft(v) => {
            crate::Taper::Draft { tangent: resolve(v)?.to_radians().tan() }
        }
    };
    Ok(crate::SectionLaw {
        taper,
        corners: law.corners,
        twist: resolve(&law.twist)?.to_radians(),
        shape: law.shape,
        window: (resolve(&law.extent[0])? / 100.0, resolve(&law.extent[1])? / 100.0),
    })
}

/// What a loft is built from, read once from the graph: each section's world polyline in the
/// order the block lists them, the centerline's polyline — named or derived — and the direction
/// the first section's own `+u` axis points, which seeds the frame as a sweep's profile does.
#[derive(Clone, Debug, PartialEq)]
pub struct LoftParts {
    pub sections: Vec<Vec<Point3<f64>>>,
    pub spine: Vec<Point3<f64>>,
    pub seed: Vector3<f64>,
    /// Whether the centerline was built from the sections rather than named. The panel draws
    /// a derived one dashed and offers to make it a real curve (`loft-spec.md` §3).
    pub derived: bool,
}

/// **One reading for the compile, the extent, the divisor and the shader**, so the four cannot
/// lay the same loft in different places — the reason `sweep_outline_section` exists for a
/// sweep.
///
/// Each section is walked as a profile is, with the centerline left out of the walk: a path
/// drawn through its sections touches them, and a walk that did not know to skip it would
/// wander off down it. A wall is refused by name, a missing curve by name.
pub fn loft_parts(
    document: &Document,
    sections: &[fieldform_scene::Outline],
    spine: Option<fieldform_scene::CurveId>,
    turn_degrees: f64,
    tolerance_mm: f64,
) -> Result<LoftParts, CompileError> {
    // **Walked once per change, not once per reader.** A rebuild asks for a loft's parts from
    // the step budget, from codegen, from the viewport's supplier, from the extent and from the
    // panel, and the walk -- tessellating each section's curves at a degree of turn and
    // deriving the centerline -- was the whole of the build's cost: 48 ms on the owner's three
    // sections, five times a rebuild, every frame of a drag (`document_cost.rs`). The key is
    // the curves themselves, found by the ends-only walk and compared whole: a curve dragged is
    // a different key, and nothing else about the document reaches the walk.
    static WALKED: std::sync::Mutex<Vec<(LoftWalkKey, LoftParts)>> = std::sync::Mutex::new(Vec::new());
    let mut members: Vec<Vec<fieldform_scene::Curve>> = Vec::with_capacity(sections.len());
    for (i, section) in sections.iter().enumerate() {
        let fieldform_scene::Outline::Closed { profile } = section else {
            return Err(CompileError::WallAsLoftSection(i));
        };
        let ids = crate::loop_from(document, *profile, spine)
            .map_err(|e| CompileError::BadLoop(*profile, e))?;
        members.push(ids.iter().filter_map(|id| document.curves.get(id).cloned()).collect());
    }
    let key = LoftWalkKey {
        members,
        spine: spine.and_then(|id| document.curves.get(&id).cloned()),
        turn_degrees,
        tolerance_mm,
    };
    if let Ok(kept) = WALKED.lock()
        && let Some((_, parts)) = kept.iter().find(|(k, _)| *k == key)
    {
        return Ok(parts.clone());
    }
    let mut walked = Vec::with_capacity(sections.len());
    for (i, section) in sections.iter().enumerate() {
        let fieldform_scene::Outline::Closed { profile } = section else {
            return Err(CompileError::WallAsLoftSection(i));
        };
        let points = crate::profile_loop_turning(document, *profile, turn_degrees, spine)
            .map_err(|e| CompileError::BadLoop(*profile, e))?
            .points;
        walked.push(points);
    }
    let (path, derived) = match spine {
        Some(id) => {
            let curve = document.curves.get(&id).ok_or(CompileError::MissingCurve(id))?;
            (crate::Spine::of_curve(curve, tolerance_mm).points().to_vec(), false)
        }
        // **Sampled finely enough that its own facets are not what the solid shows**: a cubic
        // at 128 segments turns under a degree per joint for any loft a person draws.
        None => (
            crate::derived_centerline(&walked, 128)
                .ok_or(CompileError::BadLoft(crate::LoftError::SpineHasNoLength))?,
            true,
        ),
    };
    let seed = walked
        .first()
        .and_then(|first| crate::plane_of(first))
        .map(|(_, normal)| rotation_from_z(normal) * Vector3::x())
        .unwrap_or_else(Vector3::x);
    let parts = LoftParts { sections: walked, spine: path, seed, derived };
    if let Ok(mut kept) = WALKED.lock() {
        kept.insert(0, (key, parts.clone()));
        kept.truncate(8);
    }
    Ok(parts)
}

/// What a loft's walk depends on, and nothing else: the curves of each section's loop, the
/// centerline's curve if named, and the two numbers the walk is made at.
#[derive(Clone, Debug, PartialEq)]
struct LoftWalkKey {
    members: Vec<Vec<fieldform_scene::Curve>>,
    spine: Option<fieldform_scene::Curve>,
    turn_degrees: f64,
    tolerance_mm: f64,
}

/// What a loft's value is divided by, asked of the graph: the number the leaf computes for
/// itself, for the step budget and the shader's divisor before anything has compiled -- the
/// arrangement `sweep_stretch` has.
///
/// **At the default tolerance whatever the compile's own is**, for `sweep_stretch`'s reason: the
/// divisor is written by codegen, which has no tolerance to hand, and a bend's curvature does not
/// depend on how finely its legs are flattened to the precision anything here can see. One for a
/// loft that cannot be built, which the compile refuses by name.
pub fn loft_stretch(
    document: &Document,
    sections: &[fieldform_scene::Outline],
    spine: Option<fieldform_scene::CurveId>,
    smooth: fieldform_scene::LoftSmooth,
) -> f64 {
    loft_built(document, sections, spine, smooth).map(|lofted| lofted.stretch()).unwrap_or(1.0)
}

/// A loft built the one way every reader builds it: at the default tolerance, by angle.
///
/// **The single build behind the field, the shader's numbers and the shader's polylines**, so
/// that the three cannot disagree about where a section sits. `None` for a loft that cannot be
/// built, which the compile refuses by name.
pub fn loft_built(
    document: &Document,
    sections: &[fieldform_scene::Outline],
    spine: Option<fieldform_scene::CurveId>,
    smooth: fieldform_scene::LoftSmooth,
) -> Option<std::sync::Arc<crate::Lofted>> {
    loft_parts(
        document,
        sections,
        spine,
        crate::profile::PROFILE_TURN_DEGREES,
        crate::DEFAULT_TOLERANCE_MM,
    )
    .and_then(|parts| build_loft(&parts, smooth))
    .ok()
}

/// The loft itself, built from its parts -- **once, and shared by every reader of the same
/// parts.**
///
/// Building a loft samples every pair of neighbouring sections on a grid for its bound, 17 ms
/// for four sections (`examples/loft_bound_cost.rs`). A rebuild asks for the loft from the
/// step budget (`loft_stretch`), from codegen (its stations and divisor), from the viewport's
/// binding supplier and from the extent, and a section being dragged rebuilds every frame --
/// so without this a drag paid the build three or four times a frame against a draft frame's
/// 12.7 ms, found in review. The sweep's equivalents are plain walks and never needed it.
///
/// A short list rather than a map, compared whole rather than hashed: the parts *are* the
/// key, and comparing a few thousand coordinates is nothing beside the build. The last eight
/// are kept, most recent first, which covers a document's few lofts and the two smoothnesses.
pub(crate) fn build_loft(
    parts: &LoftParts,
    smooth: fieldform_scene::LoftSmooth,
) -> Result<std::sync::Arc<crate::Lofted>, CompileError> {
    static BUILT: std::sync::Mutex<
        Vec<(LoftParts, fieldform_scene::LoftSmooth, std::sync::Arc<crate::Lofted>)>,
    > = std::sync::Mutex::new(Vec::new());
    if let Ok(kept) = BUILT.lock()
        && let Some((_, _, built)) = kept.iter().find(|(p, s, _)| p == parts && *s == smooth)
    {
        return Ok(built.clone());
    }
    let built = std::sync::Arc::new(
        crate::Lofted::new(
            &parts.sections,
            crate::Spine::new(parts.spine.clone()),
            parts.seed,
            match smooth {
                fieldform_scene::LoftSmooth::G0 => crate::Smooth::Linear,
                fieldform_scene::LoftSmooth::G1 => crate::Smooth::Hermite,
            },
        )
        .map_err(CompileError::BadLoft)?,
    );
    if let Ok(mut kept) = BUILT.lock() {
        kept.insert(0, (parts.clone(), smooth, built.clone()));
        kept.truncate(8);
    }
    Ok(built)
}

/// What a sweep's value is divided by to stay a lower bound: `√(1 + K²) / (1 - k_max * ρ_max)`,
/// the bend's own stretch times what the section law adds along the path.
///
/// The same number [`crate::Swept::under_law`] computes for itself, asked of the graph instead
/// of the compiled leaf, because the tracer buys its steps and the shader writes its divisor
/// before either has compiled anything. `1.0` for a sweep that cannot be built -- a missing
/// curve, an open profile, a fold, a law that closes its section -- since the compile refuses
/// those by name and there is nothing to buy steps for.
pub fn sweep_stretch(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    law: &fieldform_scene::SweepLaw,
) -> f64 {
    let Some(curve) = document.curves.get(&spine) else { return 1.0 };
    let Some(section) = sweep_section(document, outline, spine) else { return 1.0 };
    let Ok(law) = sweep_law(document, law) else { return 1.0 };
    // **At the default tolerance whatever the compile's own is**, because the shader's divisor
    // is written by codegen, which has no tolerance to hand. A corner's curvature does not
    // depend on how finely its legs are flattened, and a smooth bend's converges as the square
    // of the chord angle -- at these tolerances the two figures differ in the fifth place,
    // which is below what `gpu_matches_cpu` can see.
    let path = crate::Spine::of_curve(curve, crate::DEFAULT_TOLERANCE_MM);
    let run = path.length() * (law.window.1 - law.window.0);
    let product = crate::max_curvature(path.points()) * law.reach_under(section.reach, run);
    if !(product < 1.0) {
        return 1.0;
    }
    law.stretch_along(section.reach, run) / (1.0 - product)
}

/// Where a sweep's section reads as it is: the section's reach and the spine's tube, asked of
/// the graph -- what [`crate::Swept`] works out for itself, for the shader's slots before
/// anything has compiled (`sweep-spec.md` §4).
///
/// `None` wherever the whole reading is still divided by [`sweep_stretch`]: a section held at
/// an angle, a law that moves the section along the way, a tube no wider than the section
/// reaches, or a sweep that cannot be built. Decided on the law's *numbers*, as the leaf
/// decides it, so a law driven back to rest reads as the plain sweep on both sides; the
/// shader's law path takes the tube for that reason. At the default tolerance whatever the
/// compile's own, for [`sweep_stretch`]'s reason.
///
/// **The reach is the section's own**, not the solid's, and the tube is the curve's. How far
/// the solid reaches past the section is the frame's lean, and where a segment's own region
/// closes narrows the tube, and both belong to the polyline the shader walks rather than to
/// this one -- a draft walks at two degrees a joint, and the lean is half that; the glyph's
/// tight path flattened to 0.01 mm closes a short segment over at 386, and walked at one degree
/// at 482. So the shader is handed both with the frame ([`sweep_frame`]), lengthens the reach
/// and narrows the tube there, and decides the tube again against them.
pub fn sweep_tube(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    section: &fieldform_scene::Section,
    law: &fieldform_scene::SweepLaw,
) -> Option<(f64, f64)> {
    in_tube(document, outline, spine, section, law).map(|(reach, tube, _)| (reach, tube))
}

/// Whether a sweep reads its walls as they are: [`sweep_tube`]'s tube, round a spine whose frame
/// meets itself. Round a closed spine out of its plane it does not -- carried round the loop the
/// frame comes back turned -- and the reading in the tube is the divided one, held to the tube
/// (`sweep-spec.md` §4.1), so what is asked of it near the surface is as short as ever.
/// How far a held section leans off square to its path, as the cosine the sweep shortens every
/// reading by (`Swept::distance`'s `lean`): 1 for a section square to the path, and otherwise the
/// least `|t·m|` over the path's tangents `t` against the section's plane normal `m`. A flush
/// section carries its start angle the whole way, so its first tangent says it all; a parallel
/// one meets each tangent as it comes, so its least is the worst along the path.
///
/// **Why the shading wants it** (`value_scale_with`). The shortening is right for the walls that
/// lean and a plain lower bound on the faces that do not: the owner's wing, a lens held flush at
/// sixty degrees to a straight spine, read half the true distance above its top face, and the
/// occlusion took the half for material close by -- black wings, and the flank beside the root
/// in terraces where the union's reading switched body sample by sample. Dividing the shading's
/// scale by this reads the top face true and the leaning walls as more open than they are, the
/// mild side of the trade the scale makes everywhere.
pub fn sweep_lean(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    section: &fieldform_scene::Section,
) -> f64 {
    if *section == fieldform_scene::Section::Square {
        return 1.0;
    }
    let Ok(laid) = sweep_outline_section(document, outline, spine, crate::profile::PROFILE_TURN_DEGREES) else {
        return 1.0;
    };
    let Some(curve) = document.curves.get(&spine) else {
        return 1.0;
    };
    let path = curve.tessellate(0.1);
    let mut tangents = path.windows(2).filter_map(|w| (w[1] - w[0]).try_normalize(1e-9));
    let cos = |t: Vector3<f64>| t.dot(&laid.normal).abs();
    let least = match section {
        fieldform_scene::Section::Skewed => tangents.next().map(cos),
        _ => tangents.map(cos).reduce(f64::min),
    };
    least.unwrap_or(1.0).max(1e-3)
}

pub fn sweep_reads_true(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    section: &fieldform_scene::Section,
    law: &fieldform_scene::SweepLaw,
) -> bool {
    in_tube(document, outline, spine, section, law).is_some_and(|(_, _, meets)| meets)
}

/// [`sweep_tube`]'s pair and whether the frame meets itself, from one flattening.
fn in_tube(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    section: &fieldform_scene::Section,
    law: &fieldform_scene::SweepLaw,
) -> Option<(f64, f64, bool)> {
    if *section != fieldform_scene::Section::Square {
        return None;
    }
    let curve = document.curves.get(&spine)?;
    let section_read = sweep_section(document, outline, spine)?;
    let reach = section_read.reach;
    if !sweep_law(document, law).ok()?.is_plain() {
        return None;
    }
    let path = crate::Spine::of_curve(curve, crate::DEFAULT_TOLERANCE_MM);
    // A fold is refused by the compile, and has no tube to speak of.
    if !(crate::max_curvature(path.points()) * reach < 1.0) {
        return None;
    }
    // Whether the frame meets itself is the curve's -- round a closed loop out of its plane the
    // frame comes back turned however the loop is cut -- where the lean and where a segment's
    // region closes are the walk's, and the shader decides those against its own.
    let frame = crate::SpineFrame::along(&path, section_read.seed)?;
    let meets = crate::swept::frame_tilt(path.points(), &frame)?.twist.is_some();
    let tube = crate::swept::tube_radius(path.points());
    (tube > reach).then_some((reach, tube, meets))
}

/// The frame the shader is handed for a sweep: the rotation-minimising normal at each sample
/// of `spine_points`, which must be the very polyline uploaded as the spine, so the two
/// bindings cannot disagree about which sample is which. The arc length is not carried; the
/// shader accumulates it in the walk it already makes.
///
/// **And one entry after the samples: the frame's lean, twist and closing**,
/// `(1 − cos α, twist, closes)`, with `α` the most this polyline's frame leans off a segment it
/// is read on, `twist` the range of its turn about them, or −1 where it does not meet itself
/// round a closed spine, and `closes` how far out the first segment's own region closes over
/// it, 1e30 where none does ([`crate::swept::frame_tilt`]). All three belong to the walk and not
/// to the curve -- the same bend walked at two degrees a joint leans its frame twice as far as
/// at one -- and this is the one place that sees the walk, so the reading in the tube allows for
/// the polyline it is actually marched against (`sweep-spec.md` §4.1).
///
/// `None` when the section cannot be read or the spine has no direction -- both refused by the
/// compile with a message, so the caller only has to say that nothing could be supplied.
pub fn sweep_frame(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
    spine_points: &[Point3<f64>],
) -> Option<Vec<Point3<f64>>> {
    let section = sweep_section(document, outline, spine)?;
    let path = crate::Spine::new(spine_points.to_vec());
    let frame = crate::SpineFrame::along(&path, section.seed)?;
    let tilt = crate::swept::frame_tilt(spine_points, &frame)?;
    let lean = Point3::new(1.0 - tilt.cos, tilt.twist.unwrap_or(-1.0), tilt.closes.min(1e30));
    Some(frame.normals().iter().map(|n| Point3::from(*n)).chain(std::iter::once(lean)).collect())
}

/// The profile's own two axes in the world, for a shader that has to hold a section the way the
/// kernel does.
///
/// **What is shipped is the profile, not a frame.** A flush section is the profile's pair
/// *written in the frame where the path starts*, and the two sides do not sample the path
/// identically — the kernel flattens the curve at the compile's tolerance, the shader is handed
/// whatever the viewport tessellated. Resolving the coefficients here and sending them across
/// therefore measures one side's frame against the other's, which put the traced solid 2.1 mm
/// from the meshed one. So each side decomposes for itself against its own frame, and what
/// crosses is the thing they do agree about: the profile's own plane.
pub fn sweep_axes(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let laid =
        sweep_outline_section(document, outline, spine, crate::profile::PROFILE_TURN_DEGREES)
            .ok()?;
    let rotation = rotation_from_z(laid.normal);
    Some((rotation * Vector3::x(), rotation * Vector3::y()))
}

/// How far a sweep's solid is moved so that it starts at its profile rather than at its spine.
///
/// **A sweep is the profile carried along the path's *shape*, not to the path's place.** The
/// section is flattened about its origin and then ridden along the spine, so without this a
/// path that starts off the section's plane would have the section jump onto the curve — first
/// reported by the owner as *"the profile/sweep geometry seems to jump to the center of the
/// sweeping curve"*, when the origin was the centroid and any path not drawn through it moved
/// the solid.
///
/// The move is the section's origin less the spine's first point. Since 2026-10-02 the origin
/// is the spine's start dropped onto the section's plane (`sweep_outline_section`), so this is
/// **only ever along the plane's normal**: the distance the path starts off the plane, and
/// nothing for a path drawn in it, which is how one is usually drawn.
///
/// One reading, used by the compile, the extent and the shader's own copy, so the three cannot
/// disagree about where the solid is.
pub fn sweep_offset(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
) -> Option<Vector3<f64>> {
    let laid =
        sweep_outline_section(document, outline, spine, crate::profile::PROFILE_TURN_DEGREES)
            .ok()?;
    Some(laid.origin - spine_start(document, spine)?)
}

/// What a sweep's section is, read from the graph: how far the profile reaches from the origin
/// it rides the spine by, and the world direction of its own `+u` axis, which seeds the frame.
/// One reading, so the extent, the stretch and the shader's frame start from the same numbers
/// the compile does.
pub(crate) struct SweepSection {
    pub(crate) reach: f64,
    seed: Vector3<f64>,
}

pub(crate) fn sweep_section(
    document: &Document,
    outline: &fieldform_scene::Outline,
    spine: fieldform_scene::CurveId,
) -> Option<SweepSection> {
    let laid =
        sweep_outline_section(document, outline, spine, crate::profile::PROFILE_TURN_DEGREES)
            .ok()?;
    let reach = match outline {
        fieldform_scene::Outline::Closed { .. } => {
            laid.points.iter().map(|p| (p - laid.origin).norm()).fold(0.0f64, f64::max)
        }
        // A wall's curve is not its reach: the band goes a thickness past it. Asked of the
        // wall itself, so the bound is the one `Swept` computes from the footprint it holds.
        fieldform_scene::Outline::Wall { curve, thickness, side, ends } => {
            let thickness = document.resolve(thickness)?;
            let wall =
                wall_in_plane(&laid, *curve, thickness, *side, *ends, f64::INFINITY).ok()?;
            crate::Footprint::Wall(wall).reach_from_origin()
        }
    };
    Some(SweepSection { reach, seed: rotation_from_z(laid.normal) * Vector3::x() })
}
