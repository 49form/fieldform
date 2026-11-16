// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The plane a curve is drawn on, and the geometry of landing a pointer on it.

use crate::*;

/// One frozen point, with a manipulator motion applied about `pivot`.
///
/// The cage and curve half of what [`Placement::with`] does for a body: those targets have no
/// placement to compose with, so the motion is applied to the positions themselves.
pub(crate) fn moved_point(start: Point3<f64>, motion: gizmo::Motion, pivot: Point3<f64>) -> Point3<f64> {
    match motion {
        gizmo::Motion::Translate(d) => start + d,
        gizmo::Motion::Rotate { axis, degrees } => {
            match nalgebra::Unit::try_new(axis, 1e-9) {
                Some(unit) => {
                    let turn = Rotation3::from_axis_angle(&unit, degrees.to_radians());
                    pivot + turn * (start - pivot)
                }
                None => start,
            }
        }
        // A named axis scales along that axis alone, leaving the two perpendicular components
        // untouched; the uniform handle scales the whole offset. Unlike a body, a cage
        // selection may do either — moving vertices cannot break the distance field, because
        // there is no field being wrapped.
        gizmo::Motion::Scale { axis: Some(a), factor } => {
            let offset = start - pivot;
            let along = offset.dot(&a);
            pivot + offset + a * (along * (factor - 1.0))
        }
        gizmo::Motion::Scale { axis: None, factor } => pivot + (start - pivot) * factor,
    }
}

/// Intersection of a ray with a plane (point + normal), or `None` if the ray is parallel to
/// the plane.
pub(crate) fn ray_plane_intersect(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    plane_point: Point3<f32>,
    plane_normal: Vector3<f32>,
) -> Option<Point3<f32>> {
    let denom = dir.dot(&plane_normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_point - origin).dot(&plane_normal) / denom;
    Some(origin + dir * t)
}

/// How near edge-on a drawing plane may be seen and still take a click.
///
/// The cosine of the angle between the ray and the plane's normal, so this is a ray within
/// about five degrees of the plane itself. Nearer than that, a plane half a metre away puts the
/// point six metres along the ray — `ray_plane_intersect` would happily do it — and the spec's
/// acceptance is that such a click is declined and told why, not placed far away.
pub(crate) const EDGE_ON: f32 = 0.087;

/// Where a click in the viewport lands in space.
///
/// Three answers, in an order that is the whole design (`sketcher-spec.md` §7):
///
/// 1. **A drawing plane, if one is set.** Points land on it even where the model is behind
///    them — that is what setting one means, and it is what keeps a profile drawn over a body
///    flat rather than pulled onto the body's surface. Seen edge-on it declines (`None`).
/// 2. **The surface under the cursor** otherwise: the field, the cage or a curve, by what the
///    mode draws, which is the rule the whole picking path follows.
/// 3. **The plane through the pivot facing the camera**, when the ray hits nothing. Predictable,
///    and in a view square to the model it is the view plane; the point is dragged from there.
///
/// Free of the application so it can be tested without a GPU: `surface` is whatever the caller
/// can hit-test, asked only when it is needed.
pub(crate) fn landing_point(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    plane: Option<(Point3<f32>, Vector3<f32>)>,
    surface: impl FnOnce() -> Option<Point3<f32>>,
    pivot: Point3<f32>,
    forward: Vector3<f32>,
) -> Option<Point3<f32>> {
    if let Some((point, normal)) = plane {
        if dir.dot(&normal).abs() < EDGE_ON {
            return None;
        }
        return ray_plane_intersect(origin, dir, point, normal);
    }
    if let Some(hit) = surface() {
        return Some(hit);
    }
    ray_plane_intersect(origin, dir, pivot, forward)
}

/// Where new curves are built, or the world if there is none.
///
/// **Application state, not document state.** It is where the next thing you draw goes — the same
/// category as which body is selected or where the camera is — and nothing saved refers to it.
/// `sketcher-spec.md` §4 is explicit that this is what keeps the whole feature off the file
/// format: the plane a curve *remembers* is a different thing, and is §6's stamp.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum DrawingPlane {
    /// Free placement, which is what every curve has had until now.
    None,
    /// One of the world planes, named by the axis it faces along.
    World { normal: Axis, offset_mm: f64 },
    /// A plane block in the document. Its origin and facing are read through `Frame::From`, so
    /// this is the same reading every aimed block already gets.
    Node(NodeId),
}

impl DrawingPlane {
    /// How this reads in the console and on the row.
    pub(crate) fn label(self, app: &App) -> String {
        match self {
            DrawingPlane::None => "the world".to_string(),
            DrawingPlane::World { normal, offset_mm } if offset_mm.abs() < 1e-9 => {
                format!("the {} plane", plane_word(normal))
            }
            DrawingPlane::World { normal, offset_mm } => {
                format!("the {} plane at {offset_mm:.3} mm", plane_word(normal))
            }
            DrawingPlane::Node(id) => app.body_label(id),
        }
    }
}

/// The [`Frame`] a drawing plane names, or `None` for the world.
///
/// Free rather than a method so it can be tested without an application around it — the same
/// reason `operand_choices_in` is one. An `App` needs a GPU device; this needs an enum.
pub(crate) fn drawing_frame_of(plane: DrawingPlane) -> Option<fieldform_scene::Frame> {
    match plane {
        DrawingPlane::None => None,
        DrawingPlane::World { normal, offset_mm } => {
            // The origin sits *along the normal*, so `at=25` on the XY plane means 25 mm up
            // rather than 25 mm sideways. That is the only reading of "offset a plane" anybody
            // means, and it is the one that makes `plane on=xy at=25` do what it looks like.
            let along = match normal {
                Axis::X => [offset_mm, 0.0, 0.0],
                Axis::Y => [0.0, offset_mm, 0.0],
                Axis::Z => [0.0, 0.0, offset_mm],
            };
            Some(fieldform_scene::Frame::at_along(
                normal,
                along.map(ParamRef::Literal),
            ))
        }
        DrawingPlane::Node(id) => Some(fieldform_scene::Frame::From { node: id }),
    }
}

/// The distance from a point to a line segment, in screen pixels.
///
/// A segment rather than an infinite line: the nearest point on a *line* through two ends can
/// lie well beyond both of them, which would make a curve pickable along the whole of its own
/// extension. Clamping the parameter to `[0, 1]` is the difference.
pub(crate) fn distance_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let length2 = ab.length_sq();
    // A segment of no length is a point, and the projection below would divide by zero.
    if length2 < 1e-12 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / length2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// How near the cursor an end has to be, in pixels, for a snap to take it.
///
/// **Measured on screen and generously.** A world-space reach means the same gesture snaps from
/// a metre away when zoomed out and demands surgery when zoomed in, which is not what "near it"
/// means to the person doing it — the target is what they can see. Bigger than the grab radius
/// for a handle, because a snap is a deliberate act with the modifier held: there is no other
/// thing the click could have meant.
pub(crate) const SNAP_PIXELS: f32 = 22.0;

/// How big the target drawn over a snappable end is, in pixels.
///
/// Matched to the reach above rather than chosen separately: the box is a promise about where
/// the snap will take, and a box smaller than the reach would understate it while a bigger one
/// would lie.
pub(crate) const SNAP_BOX_PIXELS: f32 = SNAP_PIXELS;

/// Whether setting a drawing plane should also turn the camera to look down it.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Swing {
    /// The usual answer: choosing a plane from a menu or the console means "put me there".
    Yes,
    /// For the ViewCube, which has already turned the camera — and turned it to the face's own
    /// sign, which the plane's unsigned axis cannot express.
    No,
}

/// The world drawing plane a ViewCube region names, for the six that name one.
///
/// A face has exactly one non-zero component and that is its normal. An edge has two and a
/// corner three: they name a direction to look from, not a plane to draw on, so they get `None`
/// and keep orienting the camera and nothing else.
pub(crate) fn plane_of_face(region: CubeRegion) -> Option<DrawingPlane> {
    let nonzero: Vec<usize> = (0..3).filter(|&i| region.sign[i] != 0).collect();
    let [axis] = nonzero[..] else { return None };
    let normal = match axis {
        0 => Axis::X,
        1 => Axis::Y,
        _ => Axis::Z,
    };
    Some(DrawingPlane::World { normal, offset_mm: 0.0 })
}

/// Which way is up when looking squarely along `normal`.
///
/// Generalises `CubeRegion::up`, which answers the same question for the twenty-six cube
/// regions: looking down the world Z there is no sensible "up" along Z, so Y serves; from
/// anywhere else Z is up, which is what makes a front view look like a front view.
///
/// `an_arbitrary_plane_is_stood_up_the_way_a_cube_face_is` holds the two together, because two
/// rules for one question is how a plane picked from the menu ends up rolled differently from
/// the same plane picked off the ViewCube.
/// How square to its plane a view has to be for the grid to show: within three degrees, which is
/// nearer than a hand orbits by accident and further than a snap ever lands off.
pub(crate) const GRID_SQUARE_COS: f32 = 0.998_63;

pub(crate) fn up_for_normal(normal: Vector3<f32>) -> Vector3<f32> {
    if normal.x.abs() < 1e-6 && normal.y.abs() < 1e-6 {
        Vector3::y()
    } else {
        Vector3::z()
    }
}

/// The plane facing along an axis, said the way a person names it.
///
/// The XY plane is the one whose *normal* is Z, which is the reading everybody means and nobody
/// says out loud — so it is written down once here rather than inverted at each call site.
pub(crate) fn plane_word(normal: Axis) -> &'static str {
    match normal {
        Axis::X => "YZ",
        Axis::Y => "XZ",
        Axis::Z => "XY",
    }
}

/// The axis a named plane faces along. The inverse of [`plane_word`], and deliberately beside it.
pub(crate) fn plane_normal(word: &str) -> Option<Axis> {
    match word.to_lowercase().as_str() {
        "yz" | "zy" => Some(Axis::X),
        "xz" | "zx" => Some(Axis::Y),
        "xy" | "yx" => Some(Axis::Z),
        _ => None,
    }
}
