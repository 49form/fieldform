// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The small widgets a block's rows are built from.

use crate::*;

/// The most dots a pattern's skip row draws. A pattern may hold five hundred copies, and a row of
/// that many buttons is more than anyone can aim at; what is past this is still kept and still
/// skipped, and the console (`skip=`) reaches all of them.
pub(crate) const SKIP_DOTS_MAX: u32 = 120;

/// One dot per copy, filled while the copy is there and open once it is skipped; a click toggles it.
///
/// **The copy standing on the body has no dot to click.** Leaving it out would delete the body from
/// a pattern that is meant to include it, and the layout refuses a skip list that names it — so the
/// dot is drawn, because it is a copy, and cannot be pressed, with the reason on hover.
pub(crate) fn skip_dots(
    ui: &mut egui::Ui,
    total: u32,
    body: Option<u32>,
    skip: &[u32],
    per_line: Option<u32>,
) -> Vec<u32> {
    let mut out: Vec<u32> = skip.to_vec();
    let shown = total.min(SKIP_DOTS_MAX);
    // One line of dots for a row or a ring; for a grid, one line a row of copies, so the dots sit
    // where the copies do.
    let lines: Vec<std::ops::Range<u32>> = match per_line {
        Some(n) if n > 0 => (0..shown.div_ceil(n)).map(|r| r * n..((r + 1) * n).min(shown)).collect(),
        _ => vec![0..shown],
    };
    ui.vertical(|ui| {
        for (line, range) in lines.into_iter().enumerate() {
            ui.horizontal_wrapped(|ui| {
                if line == 0 {
                    row_label(ui, "Skip:");
                } else {
                    row_label(ui, "");
                }
                for i in range {
                    let left_out = out.contains(&i);
                    let on_body = body == Some(i);
                    let glyph = if left_out { "\u{25cb}" } else { "\u{25cf}" };
                    let dot = egui::Button::new(egui::RichText::new(glyph).small())
                        .frame(false)
                        .min_size(egui::vec2(12.0, 14.0));
                    let hover = if on_body {
                        format!("Copy {i} stands where the body does, so it cannot be skipped.")
                    } else if left_out {
                        format!("Copy {i} is left out — click to put it back.")
                    } else {
                        format!("Copy {i} — click to leave it out.")
                    };
                    if ui.add_enabled(!on_body, dot).on_hover_text(hover).clicked() {
                        if left_out {
                            out.retain(|&n| n != i);
                        } else {
                            out.push(i);
                        }
                    }
                }
            });
        }
        if total > SKIP_DOTS_MAX {
            ui.label(
                egui::RichText::new(format!("… {total} copies; `skip=` reaches them all"))
                    .small()
                    .weak(),
            );
        }
    });
    out.sort_unstable();
    out.dedup();
    out
}

/// How wide the V mark is, so a row without one still lines up with a row that has one.
pub(crate) const MARK_W: f32 = 14.0;

/// A row's label, right-aligned in the column every row in a block shares.
///
/// Allocated even for an empty label: a row that skips its label would start its numbers at the
/// panel edge while every labelled row starts them further in, and the two would not line up —
/// which is the whole thing the width exists for.
pub(crate) fn grid_gaps(ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing.x = 4.0;
}

/// The label column every row shares, right-aligned against what follows it.
///
/// Allocated at its full width and painted, rather than laid out inside a child ui: a label
/// laid out right-to-left reports only the text's own width to whatever placed it, which a
/// horizontal row never noticed (its cursor lands on the text's right edge either way) and a
/// grid did -- the column came out as wide as "Start", the text sat at the far right of the
/// space it was given, and the next column started on top of it. That is how a curve's
/// "Start" and "End" were hidden under their X fields, and under a point's name when the end
/// sat on one.
pub(crate) fn row_label(ui: &mut egui::Ui, text: &str) {
    let colour = ui.visuals().text_color();
    row_label_in(ui, text, colour);
}

/// [`row_label`] in a colour of the caller's: the error colour, for a row whose slot is empty.
pub(crate) fn row_label_in(ui: &mut egui::Ui, text: &str, colour: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(AXIS_LABEL_W, ui.spacing().interact_size.y),
        egui::Sense::hover(),
    );
    if !text.is_empty() {
        ui.painter().text(
            rect.right_center(),
            egui::Align2::RIGHT_CENTER,
            text,
            egui::TextStyle::Body.resolve(ui.style()),
            colour,
        );
    }
}

/// The green a variable wears when it is one.
///
/// **Two shades, not two marks.** The same V sits on a field that *could* be a variable and on
/// one that *is*, because they are one idea at two stages — so the difference is brightness
/// rather than a different symbol, and the eye reads "available" and "on" without reading a
/// legend. A dim V is an offer; a bright one is a fact.
pub(crate) const VARIABLE_ON: egui::Color32 = egui::Color32::from_rgb(90, 190, 110);

/// The same green, dimmed: this value could be named and has not been.
pub(crate) const VARIABLE_OFF: egui::Color32 = egui::Color32::from_rgb(96, 128, 104);

/// One appearance channel: a number dragged within its range. Returns whether it changed.
pub(crate) fn channel(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    hover: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let response = ui.add(
            egui::DragValue::new(value)
                .speed(0.005)
                .range(range)
                .max_decimals(3),
        );
        changed = response.changed();
        response.on_hover_text(hover);
    });
    changed
}

/// A small red cross, for the buttons that take something out.
///
/// **A red cross, and `U+00D7` rather than `U+2715`.** The multiplication X was not in the
/// font, so the button rendered as an empty box -- reported as a box inside a box. The times
/// sign is, and is already used elsewhere in this panel for the same job. Red because the
/// button removes something, which is the one class of button worth colouring.
pub(crate) fn remove_button(ui: &mut egui::Ui, tip: &str) -> bool {
    ui.add(
        egui::Button::new(
            egui::RichText::new("\u{d7}").color(egui::Color32::from_rgb(220, 80, 80)).strong(),
        )
        .small(),
    )
    .on_hover_text(tip)
    .clicked()
}

/// A toggle that looks like a button whether or not it is on.
///
/// **`Button::selectable` draws no frame at all when it is off.** It sets
/// `frame_when_inactive(selected)`, so an unselected toggle has no outline for the style's
/// `inactive.bg_stroke` to be drawn with — which is why setting that stroke did not give the
/// mode switches their border, and why a toggle still read as a label until the pointer crossed
/// it. Asked for twice: a button should look like a button before it is clicked.
///
/// One function rather than `.frame_when_inactive(true)` at twenty call sites, because the
/// twenty-first will be written without it.
pub(crate) fn mode_button(selected: bool, text: impl Into<egui::WidgetText>) -> egui::Button<'static> {
    egui::Button::selectable(selected, text.into()).frame_when_inactive(true)
}
