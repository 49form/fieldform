// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The menu bar, and the help windows it opens.

use crate::*;

impl App {
    /// The menu bar, generated from the command registry (`interaction-spec.md` §10.1).
    ///
    /// It is not a fourth modality beside direct manipulation, the graph and expressions —
    /// those all edit the *document*, and this is the discoverable index of everything that can
    /// be done. See the interface-modality design note.
    ///
    /// **Nothing here builds an item.** Every row is a [`commands::Command`], in the table's
    /// order, with a line wherever its group changes; every row shows its key and its console
    /// line dimmed at the right, so the slow path teaches both fast ones (P5); and every click
    /// goes through [`App::invoke_command`], the one entry point every surface shares. This file
    /// used to hold sixty hand-built buttons beside a console table and a keymap that listed
    /// the same set again.
    pub(crate) fn menu_bar(&mut self, ui: &mut egui::Ui) {
        // Every top-level menu goes through `bars`, so that once one is open the pointer
        // moving onto another opens that one — see `menu_hover`.
        let mut bars = menu_hover::BarMenus::default();
        let mut chosen: Option<&'static commands::Command> = None;
        egui::MenuBar::new().ui(ui, |ui| {
            for (title, items) in commands::menu_bar() {
                bars.menu(ui, title, |ui| {
                    // Asked only while a menu is open, since counting what a two-body tool can
                    // take is not free and the bar is drawn every frame.
                    let ctx = self.command_context();
                    if let Some(c) = menu_items(ui, &items, &ctx) {
                        chosen = Some(c);
                    }
                });
            }
        });
        bars.follow_pointer(ui.ctx());
        if let Some(c) = chosen {
            self.invoke_command(c);
        }
    }

    /// The About window, drawn over the viewport when asked for.
    pub(crate) fn help_windows(&mut self, ctx: &egui::Context) {
        // The keys are the keybinding panel's now (`app::keys_panel`), which lists every command's
        // keys and the mouse's and rebinds them; what is left here is About.
        egui::Window::new("About fieldform")
            .open(&mut self.show_about)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("fieldform {}", env!("CARGO_PKG_VERSION")));
                ui.label("A field-based CAD kernel: geometry is a function, not a mesh.");
                ui.separator();
                if cfg!(debug_assertions) {
                    ui.label("dev build — numeric crates optimized, this one not");
                } else {
                    ui.label("release build");
                }

                // **Fidget's credit is gone, because the claim it made stopped being true.**
                // It read "its expression evaluator, JIT and interval arithmetic are what every
                // traced pixel and every pruned region go through", and none of that is so: the
                // viewport is our own generated WGSL, the pruning is the per-body box guard and
                // the skip grid, and on 2026-09-29 the last dependency went with `bridge.rs`,
                // which nothing had ever called. Neither binary links a fidget crate now --
                // `cargo tree -p fieldform-app -e normal` is the check.
                //
                // Nothing is owed by removing it. The comment here always said so: MPL-2.0 asks
                // only that the notice stay with the files it covers, and `vendor/fidget/`
                // carries its own. The credit was a courtesy, and a courtesy that misdescribes
                // what it credits is worse than none. If a fidget crate is ever linked into a
                // shipped binary again, put the line back -- and make it say what is true.
                // The third-party notice. LGPL asks that a work using the library says so and
                // names the licence; this is that, and it belongs somewhere a user can find it
                // rather than in a file beside the executable.
                //
                // **It is the notice, not the whole obligation.** Distributing this also means
                // offering the modified source of the LGPL parts, and — because OCCT is linked
                // statically here — either providing what is needed to relink against a
                // different build of it, or shipping it as a shared library instead. Neither is
                // owed until something is actually distributed, and `OPENCASCADE_VENDOR.md`
                // carries the detail; saying so here would be telling the user about a duty
                // that is the project's rather than theirs.
                ui.separator();
                ui.label(
                    egui::RichText::new("STEP import uses Open CASCADE Technology")
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(
                        "OCCT is © Open CASCADE SAS and is used under the GNU LGPL 2.1 with \
                         the Open CASCADE exception. fieldform is not endorsed by or \
                         affiliated with Open CASCADE SAS.",
                    )
                    .small(),
                );
                ui.label(
                    egui::RichText::new("https://dev.opencascade.org")
                        .small()
                        .weak(),
                );
            });
    }
}

/// Draws one menu's items and answers the command clicked, if any.
///
/// Free of the `App` so it can draw while the `App` is borrowed for the context it reads, and
/// so what it does with a click is only to say which: running it is `invoke_command`'s.
///
/// A setting with a fixed value (Display quality › Fast) draws as a choice among its siblings; a
/// setting with none (Section) draws as a tick box; everything else is a button. Each row shows
/// its key and console line at the right, and says why when it is greyed out.
fn menu_items(
    ui: &mut egui::Ui,
    items: &[commands::MenuItem],
    ctx: &commands::Ctx,
) -> Option<&'static commands::Command> {
    let mut chosen = None;
    for item in items {
        match item {
            commands::MenuItem::Separator => {
                ui.separator();
            }
            commands::MenuItem::Submenu(title, inner) => {
                ui.menu_button(*title, |ui| {
                    if let Some(c) = menu_items(ui, inner, ctx) {
                        chosen = Some(c);
                    }
                });
            }
            commands::MenuItem::Command(c) => {
                let available = (c.context)(ctx);
                let label = commands::label(c, ctx);
                let right = egui::RichText::new(commands::menu_shortcut_text(c)).weak();
                let response = match c.checked.map(|on| on(ctx)) {
                    Some(on) if c.args.is_empty() => {
                        let mut ticked = on;
                        let checkbox = egui::Checkbox::new(
                            &mut ticked,
                            (label, egui::Atom::grow(), right),
                        );
                        ui.add_enabled(available.is_ok(), checkbox)
                    }
                    Some(on) => ui.add_enabled(
                        available.is_ok(),
                        egui::Button::selectable(on, label).shortcut_text(right),
                    ),
                    None => ui.add_enabled(
                        available.is_ok(),
                        egui::Button::new(label).shortcut_text(right).sense(egui::Sense::click_and_drag()),
                    ),
                };
                // A command row dragged out of the menu onto the shelf stays there (§10.2).
                response.dnd_set_drag_payload(app::shelf::ShelfItem(c.id));
                let response = match available {
                    Ok(()) if !c.hint.is_empty() => response.on_hover_text(c.hint),
                    Ok(()) => response,
                    Err(why) => response.on_disabled_hover_text(why),
                };
                if response.clicked() {
                    chosen = Some(*c);
                    ui.close();
                }
            }
        }
    }
    chosen
}
