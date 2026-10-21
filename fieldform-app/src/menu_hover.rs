// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A menu bar that follows the pointer once one of its menus is open.
//!
//! Click File, move to Edit, and Edit should open by itself. egui 0.34 does that for
//! *sub*menus only (`is_hovered && !is_any_open` in `containers/menu.rs`); a top-level bar
//! button opens on a click and on nothing else, so every menu cost its own click — the one
//! interaction in the application that disagreed with every other program on the machine.
//! There is no flag for it: `MenuBar` carries a style and a config and neither reaches the
//! bar's own buttons.
//!
//! The pieces that are public are enough. egui keeps *one* open top-level popup per viewport
//! and `Popup::open_id` replaces it, so switching is "open the hovered sibling" and closing the
//! old one comes free. What the bar has to supply is the two conditions under which that is
//! wrong, which is why this is a type and not a line in `menu_bar`:
//!
//! - **Nothing is open.** The pointer crossing the bar on its way to the viewport must open
//!   nothing, so a hover only switches *from* a bar menu that is open.
//! - **The hovered button is the open one.** Its own click has to close it, and that is
//!   egui's `Popup::menu` toggling on the click; reopening it here would undo that.
//!
//! Which menu is open is asked of egui by each button's popup id rather than remembered, so
//! this holds nothing between frames: a combo box or a context menu open elsewhere is not a
//! bar menu and does not make the bar follow the pointer.

use eframe::egui;

/// One top-level menu as the bar drew it this frame.
struct Drawn {
    /// The popup id egui gave the button, which is how egui answers "is this one open".
    popup: egui::Id,
    /// Whether the pointer was on the button.
    hovered: bool,
}

/// The top-level menus of one bar, noted as they are drawn.
///
/// Build one before the bar, draw each top-level menu through [`BarMenus::menu`], and call
/// [`BarMenus::follow_pointer`] after the bar — the switch needs every button's hover, and
/// the one to switch *to* may be drawn after the one that is open.
#[derive(Default)]
pub struct BarMenus {
    drawn: Vec<Drawn>,
}

impl BarMenus {
    /// Draw one top-level menu, exactly as `ui.menu_button` does, and note it.
    pub fn menu<R>(
        &mut self,
        ui: &mut egui::Ui,
        title: &str,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::Response {
        let shown = ui.menu_button(title, add_contents);
        self.drawn.push(Drawn {
            popup: egui::Popup::default_response_id(&shown.response),
            hovered: shown.response.hovered(),
        });
        shown.response
    }

    /// If a bar menu is open and the pointer is on a different one, open that one.
    ///
    /// The new menu draws on the next frame, which is why a repaint is asked for: egui only
    /// repaints on input, and a pointer that has stopped moving would otherwise leave the old
    /// menu on screen until it moved again.
    pub fn follow_pointer(&self, ctx: &egui::Context) {
        let Some(open) = self.drawn.iter().find(|m| egui::Popup::is_id_open(ctx, m.popup)) else {
            return;
        };
        let Some(onto) = self.drawn.iter().find(|m| m.hovered && m.popup != open.popup) else {
            return;
        };
        egui::Popup::open_id(ctx, onto.popup);
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TITLES: [&str; 3] = ["File", "Edit", "View"];

    /// A bar of three menus in a headless egui context, driven by the same pointer events a
    /// window would deliver. What the property needs is egui's own hit-testing, popup memory
    /// and click detection, none of which a hand-built `Response` would exercise.
    struct Bar {
        ctx: egui::Context,
        time: f64,
        rects: Vec<egui::Rect>,
        open: Vec<bool>,
    }

    impl Bar {
        fn new() -> Self {
            let mut bar = Self {
                ctx: egui::Context::default(),
                time: 0.0,
                rects: Vec::new(),
                open: Vec::new(),
            };
            // Two frames settle the layout, so every later pointer position is on a button
            // egui has already seen: its hit test reads the previous frame's rectangles.
            bar.frame(vec![]);
            bar.frame(vec![]);
            bar
        }

        fn frame(&mut self, events: Vec<egui::Event>) {
            self.time += 0.05;
            let mut raw = egui::RawInput::default();
            raw.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            ));
            raw.time = Some(self.time);
            raw.events = events;

            let mut rects = Vec::new();
            let mut bars = BarMenus::default();
            let _ = self.ctx.run_ui(raw, |ui| {
                rects.clear();
                bars.drawn.clear();
                egui::MenuBar::new().ui(ui, |ui| {
                    for title in TITLES {
                        let button = bars.menu(ui, title, |ui| {
                            let _ = ui.button("an item");
                        });
                        rects.push(button.rect);
                    }
                });
                bars.follow_pointer(ui.ctx());
            });
            self.rects = rects;
            self.open = bars
                .drawn
                .iter()
                .map(|m| egui::Popup::is_id_open(&self.ctx, m.popup))
                .collect();
        }

        fn centre(&self, menu: usize) -> egui::Pos2 {
            self.rects[menu].center()
        }

        fn hover(&mut self, menu: usize) {
            let at = self.centre(menu);
            self.frame(vec![egui::Event::PointerMoved(at)]);
            // And a few more with the pointer still, so a switch that was only *requested*
            // has had its frames to land in.
            for _ in 0..3 {
                self.frame(vec![]);
            }
        }

        fn click(&mut self, menu: usize) {
            let at = self.centre(menu);
            let button = |pressed| egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            self.frame(vec![egui::Event::PointerMoved(at)]);
            self.frame(vec![button(true)]);
            self.frame(vec![button(false)]);
            for _ in 0..3 {
                self.frame(vec![]);
            }
        }

        fn open_menus(&self) -> Vec<&'static str> {
            TITLES
                .iter()
                .zip(&self.open)
                .filter(|(_, open)| **open)
                .map(|(title, _)| *title)
                .collect()
        }
    }

    #[test]
    fn a_menu_opens_by_hover_once_a_sibling_is_open() {
        let mut bar = Bar::new();
        bar.click(0);
        assert_eq!(bar.open_menus(), vec!["File"], "a click opens a menu, as it always did");

        bar.hover(1);
        assert_eq!(bar.open_menus(), vec!["Edit"], "and moving to Edit swaps it in");

        bar.hover(2);
        assert_eq!(bar.open_menus(), vec!["View"]);

        bar.hover(0);
        assert_eq!(bar.open_menus(), vec!["File"], "and back again");
    }

    #[test]
    fn passing_over_the_bar_with_nothing_open_opens_nothing() {
        let mut bar = Bar::new();
        for menu in [0, 1, 2, 1, 0] {
            bar.hover(menu);
            assert_eq!(bar.open_menus(), Vec::<&str>::new(), "hovering menu {menu}");
        }
    }

    #[test]
    fn clicking_the_open_menus_own_button_closes_it_and_it_stays_closed() {
        let mut bar = Bar::new();
        bar.click(1);
        assert_eq!(bar.open_menus(), vec!["Edit"]);

        bar.click(1);
        assert_eq!(bar.open_menus(), Vec::<&str>::new(), "the second click closes it");

        // The pointer has not moved off the button. A bar that reopened on hover would have
        // brought the menu straight back here, which is the trap the type exists to avoid.
        for _ in 0..5 {
            bar.frame(vec![]);
        }
        assert_eq!(bar.open_menus(), Vec::<&str>::new());
    }

    /// **A judgment call, written down so it is one.** There is no path for a pointer onto
    /// another menu that does not cross it first, so by the time a click lands the hover has
    /// already opened it — and a click on an open menu's own title closes it, which is the
    /// toggle the first trap asks for. Clicking the title you have just arrived at therefore
    /// closes the menu you were brought to. That is what a toggle does in other programs too,
    /// but it is the one place this feature could be felt as a flicker; the alternative is a
    /// bit of state remembering "opened by hover" so that one click is swallowed.
    #[test]
    fn a_click_on_the_menu_the_hover_just_opened_closes_it_like_any_open_menu() {
        let mut bar = Bar::new();
        bar.click(0);
        bar.hover(2);
        assert_eq!(bar.open_menus(), vec!["View"]);
        bar.click(2);
        assert_eq!(bar.open_menus(), Vec::<&str>::new());
    }

    #[test]
    fn leaving_the_bar_leaves_the_open_menu_open() {
        let mut bar = Bar::new();
        bar.click(0);
        // Down into the viewport, nowhere near a button: hover has nothing to switch to,
        // and a click is what closes a menu, not the pointer going away.
        bar.frame(vec![egui::Event::PointerMoved(egui::pos2(400.0, 400.0))]);
        for _ in 0..3 {
            bar.frame(vec![]);
        }
        assert_eq!(bar.open_menus(), vec!["File"]);
    }
}
