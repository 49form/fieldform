// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The drawing plane's grid: how fine it is, the snap to it, and where its numbers go.

use super::*;

/// The narrowest grid cell worth drawing or snapping to, in pixels.
///
/// Narrower, and the lines read as a tint and the intersections are targets nobody can aim
/// at. The pitch is the smallest decade of millimetres whose cell is at least this wide, so as
/// the view zooms the grid steps rather than crowds.
pub(crate) const GRID_MIN_CELL_PX: f64 = 32.0;

/// The finest grid there is, in millimetres.
///
/// Asked for by name: "when the grid is visible we should see, depending on how zoomed in we
/// are, either every 10 mm or 100 mm or 1000 mm". Ten is where that list starts, so zooming in
/// past it makes the cells bigger rather than the pitch finer — which is the right answer for a
/// grid whose job is to say what size things are.
pub(crate) const GRID_FINEST_MM: f64 = 10.0;

/// The grid pitch for a view, in millimetres, from how many pixels a millimetre covers.
///
/// **Decades, not a 1-2-5 sequence.** It was 1, 2, 5, 10, 20, 50 …, which is what a rule is
/// divided into and what most CAD grids offer — and asked to be changed: ten, a hundred, a
/// thousand, and nothing between. Those are the numbers a part is *described* in, so a grid that
/// only ever shows one of them can be read as a measurement rather than counted.
///
/// Chosen from the zoom rather than typed, so the grid stays readable at every scale from a
/// bracket to an airframe — a fixed 10 mm pitch on a framed metre is a hundred-line tint — and
/// so a snap always goes to something that is drawn. A grid you can snap to and cannot see is
/// the snap-box lesson again.
pub(crate) fn grid_pitch(px_per_mm: f64) -> f64 {
    if !(px_per_mm.is_finite() && px_per_mm > 0.0) {
        return GRID_FINEST_MM;
    }
    // The smallest power of ten whose cell is wide enough to aim at. `ceil` rather than `floor`,
    // since the decade below the requirement is by definition too narrow.
    let least = GRID_MIN_CELL_PX / px_per_mm;
    let decade = 10f64.powi(least.log10().ceil() as i32);
    decade.max(GRID_FINEST_MM)
}

/// The nearest grid intersection to `point`, on the plane placed by `translation` and `rotation`.
///
/// Snapped in the plane's own coordinates and only in the two that lie in it. The third is left
/// exactly as it was, so a point on the plane stays on it to the bit, and a point off it — one
/// snapped to a curve's end elsewhere — is not pulled onto a plane it was never on.
pub(crate) fn snap_to_grid(
    point: Point3<f64>,
    translation: Vector3<f64>,
    rotation: Rotation3<f64>,
    pitch: f64,
) -> Point3<f64> {
    let local = rotation.inverse() * (point.coords - translation);
    let snapped = Vector3::new(
        (local.x / pitch).round() * pitch,
        (local.y / pitch).round() * pitch,
        local.z,
    );
    Point3::from(rotation * snapped + translation)
}

/// The grid drawn on the drawing plane this frame: where it is, how fine, and how much of it.
///
/// Anchored to the plane's own origin — the intersections sit at whole multiples of the pitch in
/// its coordinates, which is what makes them worth snapping to — while the window drawn follows
/// the pivot, so the part of the plane being worked on is the part that has lines.
pub(crate) struct PlaneGrid {
    pub(crate) translation: Vector3<f64>,
    pub(crate) rotation: Rotation3<f64>,
    pub(crate) pitch: f64,
    /// How many pixels a millimetre covers, which is what decides whether there is room to
    /// write a number between two lines.
    pub(crate) px_per_mm: f64,
    /// The line indices at the window's centre, along the plane's X and Y.
    pub(crate) centre: (i64, i64),
    /// Lines either side of the centre, along each axis.
    pub(crate) reach: i64,
}

/// The grid's lines. Faint, like the plane preview: scaffolding under the model, not something
/// the model is behind. Every fifth line a little stronger so distances can be counted off it,
/// and the plane's own axes stronger again so its origin can be found.
///
/// Premultiplied, because that is the form that can be a constant: each is a blue-grey at the
/// alpha named last, with the colour already scaled by it.
pub(crate) const GRID_MINOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(20, 22, 25, 34);
pub(crate) const GRID_MAJOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(42, 47, 52, 72);
pub(crate) const GRID_AXIS: egui::Color32 = egui::Color32::from_rgba_premultiplied(83, 91, 100, 125);

/// The ink and the size of a grid's numbers. Weak, because they are a margin rather than
/// content: strong enough to read against the model, not strong enough to compete with it.
pub(crate) const GRID_LABEL: egui::Color32 = egui::Color32::from_rgba_premultiplied(112, 120, 130, 200);

/// How far inside the viewport a grid number sits, so it is not half off the edge.
pub(crate) const GRID_LABEL_INSET: f32 = 16.0;

/// Where a grid line's number goes: just inside the viewport, at whichever end of the line is
/// nearer the bottom-left.
///
/// **So the numbers read like the margin of a drawing** rather than a trail across the model.
/// The alternative — putting each number where its line crosses the plane's own axis — writes
/// nothing at all the moment the origin is off screen, which is most of the time once anybody
/// has zoomed in on a detail. Which is exactly when somebody wants to know what size it is.
///
/// The segment is clipped to the viewport first (Liang-Barsky), so the anchor is on the part of
/// the line that is actually visible. `None` when the line misses the viewport entirely.
pub(crate) fn grid_label_at(rect: egui::Rect, a: egui::Pos2, b: egui::Pos2) -> Option<egui::Pos2> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [
        (-d.x, a.x - rect.left()),
        (d.x, rect.right() - a.x),
        (-d.y, a.y - rect.top()),
        (d.y, rect.bottom() - a.y),
    ] {
        if p == 0.0 {
            // Parallel to this edge: outside it means outside altogether.
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            t0 = t0.max(r);
        } else {
            if r < t0 {
                return None;
            }
            t1 = t1.min(r);
        }
    }
    let at = |t: f32| a + d * t;
    let corner = egui::pos2(rect.left(), rect.bottom());
    let (p0, p1) = (at(t0), at(t1));
    let mut chosen = if p0.distance(corner) <= p1.distance(corner) { p0 } else { p1 };
    chosen.x = chosen.x.clamp(rect.left() + GRID_LABEL_INSET, rect.right() - GRID_LABEL_INSET);
    chosen.y = chosen.y.clamp(rect.top() + GRID_LABEL_INSET, rect.bottom() - GRID_LABEL_INSET);
    Some(chosen)
}
