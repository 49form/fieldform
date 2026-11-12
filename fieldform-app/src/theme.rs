// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The application's colour tokens, starting with the block families'.
//!
//! **The only place a family colour is written.** The block list, its chips and the graph view
//! all ask here, so a token changes in one line and the views cannot disagree about it. A token
//! changes only through the palette checker in `tools/design/`: edit its table, run it, and copy
//! the hex it passed into [`DARK`] and [`LIGHT`] — the test below holds this module to the table
//! the checker printed, so changing one without the other fails.
//!
//! **Why these hues.** All seven keep clear of red, orange, amber and yellow (OKLCH 130°–345°),
//! which are the application's warnings and errors; they are muted, so a 1 pt line names a hue
//! without shouting; and the six chromatic ones stay at least 10 ΔE₀₀ apart on the dark panel and
//! 9.5 on the light one under simulated protanopia, deuteranopia and tritanopia (Machado,
//! Oliveira and Fernandes, 2009). The seventh, Reference, is a neutral grey that a red–green
//! deficiency would fold onto the teal of Curve, so it is told apart by a dashed frame as well —
//! which is why the dash is not optional.

use eframe::egui;
use egui::{Color32, Stroke};
use fieldform_scene::Family;

use crate::graph_view::model::Status;

/// One theme's seven family colours.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FamilyTokens {
    pub(crate) combine: Color32,
    pub(crate) create: Color32,
    pub(crate) modify: Color32,
    pub(crate) curve: Color32,
    pub(crate) cage: Color32,
    pub(crate) file: Color32,
    pub(crate) reference: Color32,
}

/// The dark theme's tokens, against egui's dark panel (`#1b1b1b`).
///
/// Combine is the brightest on purpose: it is the outermost frame of almost every expanded tree,
/// the seam a reader looks for first. Create is the most numerous row and a step dimmer, so a
/// list of leaves does not shout. Reference is the dimmest, as construction geometry should be.
static DARK: FamilyTokens = FamilyTokens {
    combine: Color32::from_rgb(0x9d, 0xc9, 0xf7),
    create: Color32::from_rgb(0x9d, 0x92, 0xde),
    modify: Color32::from_rgb(0x85, 0xb2, 0x86),
    curve: Color32::from_rgb(0x4f, 0x9c, 0x8e),
    cage: Color32::from_rgb(0xd7, 0x9d, 0xbd),
    file: Color32::from_rgb(0x9c, 0x76, 0xa4),
    reference: Color32::from_rgb(0x7a, 0x7a, 0x7a),
};

/// The light theme's tokens, against egui's light panel (`#f8f8f8`): the same hues, darker.
static LIGHT: FamilyTokens = FamilyTokens {
    combine: Color32::from_rgb(0x43, 0x7e, 0xb9),
    create: Color32::from_rgb(0x5e, 0x51, 0x97),
    modify: Color32::from_rgb(0x56, 0x81, 0x57),
    curve: Color32::from_rgb(0x17, 0x6c, 0x5f),
    cage: Color32::from_rgb(0x9d, 0x67, 0x85),
    file: Color32::from_rgb(0x60, 0x3e, 0x68),
    reference: Color32::from_rgb(0x8c, 0x8c, 0x8c),
};

/// How much of a token a tint is, over the panel: a chip's fill and nothing else.
const TINT: f32 = 0.14;

/// How much of a token a hidden body's row keeps: its family still, dimmed with the rest of it.
pub(crate) const DIMMED: f32 = 0.5;

/// A curvature comb's teeth and envelope. Not a family colour and not held to the checker, but
/// written here because this is where colours are written: a cool blue-grey, clear of red, orange,
/// amber and yellow so a comb never reads as a warning, and clear of the curves' own orange so it
/// is never mistaken for the line it measures.
pub(crate) const ANALYSIS_COMB: egui::Color32 = egui::Color32::from_rgb(128, 168, 208);

/// A frame's corner radius, in points: the row's and the bar's outer corners.
pub(crate) const FRAME_RADIUS: f32 = 3.0;

/// The left bar's width, in points.
pub(crate) const BAR_WIDTH: f32 = 3.0;

/// The frame's padding inside its stroke, in points. The bar sits in the left padding, so the
/// left margin is the bar and this together.
pub(crate) const FRAME_PADDING: f32 = 4.0;

/// Reference's dash: on, then off, in points.
pub(crate) const DASH: (f32, f32) = (4.0, 3.0);

/// The tokens for the theme in use — `ui.visuals().dark_mode`.
pub(crate) fn family_tokens(dark: bool) -> &'static FamilyTokens {
    if dark { &DARK } else { &LIGHT }
}

pub(crate) fn family_colour(family: Family, dark: bool) -> Color32 {
    let t = family_tokens(dark);
    match family {
        Family::Combine => t.combine,
        Family::Create => t.create,
        Family::Modify => t.modify,
        Family::Curve => t.curve,
        Family::Cage => t.cage,
        Family::File => t.file,
        Family::Reference => t.reference,
    }
}

/// A token at [`TINT`]: alpha rather than a second hex per token, so one table stays one table.
pub(crate) fn family_tint(family: Family, dark: bool) -> Color32 {
    family_colour(family, dark).gamma_multiply(TINT)
}

/// How a framed thing is drawn: its outline, whether that outline is dashed, and its bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FramePaint {
    pub(crate) stroke: Stroke,
    /// Reference's construction-line dash, which carries the family where its grey cannot.
    pub(crate) dashed: bool,
    /// The bar's colour, and whether it is hollow (Reference's). `None` for a block whose kind is
    /// not known, which is the one row with no bar.
    pub(crate) bar: Option<(Color32, bool)>,
}

/// **The one rule for a frame's stroke, in both views**: the family's, unless the block has a
/// problem, in which case the warning or the error stroke the application already uses — 1 pt,
/// solid — while the bar keeps the family, so a row never loses its kind while it is wrong.
///
/// A problem is rarer and more urgent than a family and must be unmistakable, which is what the
/// family hues keeping clear of red and amber buys: a status stroke is never confusable with one.
/// The status glyph or hover stays in all three states, so a problem is never colour alone.
///
/// `family` is `None` for a block this build cannot read: the error stroke and no bar, whatever
/// the status says, since a kind that is not known cannot honestly be filed.
pub(crate) fn frame_paint(family: Option<Family>, status: &Status, visuals: &egui::Visuals) -> FramePaint {
    let dark = visuals.dark_mode;
    let Some(family) = family else {
        return FramePaint { stroke: Stroke::new(1.0_f32, visuals.error_fg_color), dashed: false, bar: None };
    };
    let colour = family_colour(family, dark);
    let reference = family == Family::Reference;
    let bar = Some((colour, reference));
    match status {
        Status::Ok => FramePaint { stroke: Stroke::new(1.0_f32, colour), dashed: reference, bar },
        Status::Warning(_) => FramePaint { stroke: Stroke::new(1.0_f32, visuals.warn_fg_color), dashed: false, bar },
        Status::Error(_) => FramePaint { stroke: Stroke::new(1.0_f32, visuals.error_fg_color), dashed: false, bar },
    }
}

impl FramePaint {
    /// The same paint at `alpha` — a hidden body's row, whose frame and bar stay its family's.
    pub(crate) fn dimmed(self, alpha: f32) -> Self {
        Self {
            stroke: Stroke::new(self.stroke.width, self.stroke.color.gamma_multiply(alpha)),
            dashed: self.dashed,
            bar: self.bar.map(|(c, hollow)| (c.gamma_multiply(alpha), hollow)),
        }
    }

    /// Paints the outline and the bar on `rect`, the frame's own outer rectangle.
    ///
    /// **The dash is four dashed sides with plain corners**: egui's painter has a dashed line and
    /// no dashed rounded rectangle, and at 3 pt a rounded dashed corner is a smudge anyway. The
    /// solid outline keeps its corner radius.
    pub(crate) fn paint(&self, painter: &egui::Painter, rect: egui::Rect) {
        let inside = rect.shrink(self.stroke.width * 0.5);
        if self.dashed {
            let corners = [inside.left_top(), inside.right_top(), inside.right_bottom(), inside.left_bottom()];
            for i in 0..4 {
                painter.extend(egui::Shape::dashed_line(
                    &[corners[i], corners[(i + 1) % 4]],
                    self.stroke,
                    DASH.0,
                    DASH.1,
                ));
            }
        } else {
            painter.rect_stroke(rect, FRAME_RADIUS, self.stroke, egui::StrokeKind::Inside);
        }
        if let Some((colour, hollow)) = self.bar {
            let bar = bar_rect(rect);
            let rounding = egui::CornerRadius { nw: FRAME_RADIUS as u8, sw: FRAME_RADIUS as u8, ne: 0, se: 0 };
            if hollow {
                painter.rect_stroke(bar, rounding, Stroke::new(1.0_f32, colour), egui::StrokeKind::Inside);
            } else {
                painter.rect_filled(bar, rounding, colour);
            }
        }
    }
}

/// Paints a chip — a reference field that holds a variable — as a pill in its family: a 1 pt
/// stroke in the token, a fill of its tint, dashed for Reference as its frame is.
///
/// **Text is never drawn on a family colour**, only on its tint, so no token has to meet a text
/// contrast it was not chosen for: the field's own text colour stays the theme's.
///
/// `behind` is a slot reserved in the painter before the field was added, so the fill lies under
/// the field's text rather than over it.
pub(crate) fn paint_chip(
    painter: &egui::Painter,
    behind: egui::layers::ShapeIdx,
    rect: egui::Rect,
    family: Family,
    dark: bool,
) {
    let stroke = Stroke::new(1.0_f32, family_colour(family, dark));
    let radius = (rect.height() * 0.5).min(255.0);
    let tint = family_tint(family, dark);
    if family == Family::Reference {
        painter.set(behind, egui::Shape::rect_filled(rect, radius, tint));
        let mut path = pill_points(rect.shrink(0.5));
        path.push(path[0]);
        painter.extend(egui::Shape::dashed_line(&path, stroke, DASH.0, DASH.1));
    } else {
        painter.set(
            behind,
            egui::epaint::RectShape::new(rect, radius, tint, stroke, egui::StrokeKind::Inside),
        );
    }
}

/// The outline of a pill on `rect`, as a polyline: the right end's half circle from its top, the
/// left end's from its bottom. Twelve segments an end, which at a field's height is under a
/// quarter of a point from the circle.
fn pill_points(rect: egui::Rect) -> Vec<egui::Pos2> {
    use std::f32::consts::{FRAC_PI_2, PI};
    const SEGMENTS: usize = 12;
    let r = rect.height() * 0.5;
    let cy = rect.center().y;
    let left = rect.left() + r;
    let right = (rect.right() - r).max(left);
    let mut out = Vec::with_capacity(2 * SEGMENTS + 2);
    for (centre, start) in [(right, -FRAC_PI_2), (left, FRAC_PI_2)] {
        for i in 0..=SEGMENTS {
            let a = start + PI * i as f32 / SEGMENTS as f32;
            out.push(egui::pos2(centre + r * a.cos(), cy + r * a.sin()));
        }
    }
    out
}

/// Where the bar of a frame whose outer rectangle is `rect` lies: down its left edge, inside it,
/// the full height.
pub(crate) fn bar_rect(rect: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_size(rect.min, egui::vec2(BAR_WIDTH, rect.height()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(c: Color32) -> String {
        format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
    }

    /// **The tokens are the table the checker passed**, written here as data: changing a token
    /// means changing this test, which is the moment to run the checker.
    #[test]
    fn the_tokens_are_the_table_the_palette_checker_passed() {
        let table = [
            (Family::Combine, "#9dc9f7", "#437eb9"),
            (Family::Create, "#9d92de", "#5e5197"),
            (Family::Modify, "#85b286", "#568157"),
            (Family::Curve, "#4f9c8e", "#176c5f"),
            (Family::Cage, "#d79dbd", "#9d6785"),
            (Family::File, "#9c76a4", "#603e68"),
            (Family::Reference, "#7a7a7a", "#8c8c8c"),
        ];
        for (family, dark, light) in table {
            assert_eq!(hex(family_colour(family, true)), dark, "{family:?}, dark");
            assert_eq!(hex(family_colour(family, false)), light, "{family:?}, light");
        }
        assert_eq!(table.len(), Family::ALL.len());
    }

    /// **And the checker's own table is the same**, so the three — checker, this module, the test
    /// above — are one list: the checker is read as text, since it is Python.
    #[test]
    fn the_checker_holds_the_same_tokens() {
        let checker = include_str!("../../tools/design/family_palette_check.py");
        for family in Family::ALL {
            for dark in [true, false] {
                let line = format!("\"{}\": \"{}\"", family.label().to_lowercase(), hex(family_colour(family, dark)));
                assert!(checker.contains(&line), "the checker has no {line}");
            }
        }
    }

    /// **The panels the checker measured contrast against are the panels drawn**: egui's own, which
    /// the application does not retune. If it ever does, this fails until the checker is run
    /// against the new panel and its table copied back.
    #[test]
    fn the_checker_measured_against_the_panels_the_list_is_drawn_on() {
        let checker = include_str!("../../tools/design/family_palette_check.py");
        for (theme, visuals) in [("dark", egui::Visuals::dark()), ("light", egui::Visuals::light())] {
            let panel = hex(visuals.panel_fill);
            assert!(checker.contains(&format!("\"{theme}\": \"{panel}\"")), "the checker's {theme} panel is not {panel}");
        }
    }

    /// **A status replaces the stroke and never the bar**; `Ok` is the family throughout; a kind
    /// that is not known has the error stroke and no bar.
    #[test]
    fn a_status_takes_the_stroke_and_the_bar_keeps_the_family() {
        for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
            let dark = visuals.dark_mode;
            for family in Family::ALL {
                let colour = family_colour(family, dark);
                let ok = frame_paint(Some(family), &Status::Ok, &visuals);
                assert_eq!(ok.stroke, Stroke::new(1.0_f32, colour), "{family:?}");
                assert_eq!(ok.bar.map(|b| b.0), Some(colour));
                assert_eq!(ok.dashed, family == Family::Reference, "only Reference is dashed");
                let warning = frame_paint(Some(family), &Status::Warning("why".into()), &visuals);
                assert_eq!(warning.stroke, Stroke::new(1.0_f32, visuals.warn_fg_color));
                assert_eq!(warning.bar.map(|b| b.0), Some(colour), "the bar keeps the family");
                assert!(!warning.dashed, "a status stroke is solid");
                let error = frame_paint(Some(family), &Status::Error("why".into()), &visuals);
                assert_eq!(error.stroke, Stroke::new(1.0_f32, visuals.error_fg_color));
                assert_eq!(error.bar.map(|b| b.0), Some(colour));
            }
            let unknown = frame_paint(None, &Status::Ok, &visuals);
            assert_eq!(unknown.stroke, Stroke::new(1.0_f32, visuals.error_fg_color));
            assert_eq!(unknown.bar, None, "a kind not known has no bar");
        }
    }

    /// The reason the hue band stops short of red and amber: no family may be mistaken for the
    /// status that replaces its stroke. Held as a hue distance in the colour wheel egui's own
    /// conversion gives, a coarse stand-in for the checker's OKLCH band that is enough to catch a
    /// token pasted from the wrong row.
    #[test]
    fn no_family_is_the_warning_or_the_error_colour() {
        for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
            for family in Family::ALL {
                let c = family_colour(family, visuals.dark_mode);
                assert_ne!(c, visuals.warn_fg_color);
                assert_ne!(c, visuals.error_fg_color);
                let hsva = egui::ecolor::Hsva::from(c);
                // Red through yellow is the first sixth of the wheel and the last sliver of it.
                let hue = hsva.h * 360.0;
                assert!(hsva.s < 0.05 || (70.0..=345.0).contains(&hue), "{family:?} at {hue:.0}°");
            }
        }
    }
}
