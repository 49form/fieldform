// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The interaction tests: a real `App` on a real device, without a window.
//!
//! **Why a real one.** Everything `interaction-spec.md` asks to be tested -- the echo's round
//! trip, a cancel that leaves nothing behind, one undo step per session -- is a property of what
//! `dispatch` and `run_command` do to the document, and those live on the `App`. The tests before
//! this held the free functions around it, which is right for a rule and wrong for a behaviour: a
//! copy of the behaviour in a test is a second implementation, and it agrees with itself.
//!
//! eframe builds a `CreationContext` for exactly this (`_new_kittest`) and egui-wgpu builds a
//! device with no surface, so the `App` here is the one a window gets. A machine with no adapter
//! skips these and says so, rather than failing a run for want of a GPU.

use super::*;
use eframe::App as _;

/// The application as a window would make it, or `None` where this machine has no adapter.
///
/// A test build keeps its hands off the owner's files: no console log (the name is shared and
/// opening it truncates it), no autosave written or offered, and no display quality kept for the
/// next session -- see the `cfg!(test)` guards beside each.
pub(crate) fn headless_app() -> Option<App> {
    let config = wgpu_configuration();
    let instance = pollster::block_on(config.wgpu_setup.new_instance());
    let state = match pollster::block_on(eframe::egui_wgpu::RenderState::create(
        &config,
        &instance,
        None,
        eframe::egui_wgpu::RendererOptions::default(),
    )) {
        Ok(state) => state,
        Err(why) => {
            eprintln!("skipped: no adapter for a headless App ({why})");
            return None;
        }
    };
    let mut cc = eframe::CreationContext::_new_kittest(egui::Context::default());
    cc.wgpu_render_state = Some(state);
    Some(App::new(&cc))
}

/// A key pressed and nothing held with it, as one event of a frame's input.
pub(crate) fn key(key: egui::Key) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }
}

/// One frame of the whole interface, with these events in its input, as the window would run it.
pub(crate) fn frame(app: &mut App, events: Vec<egui::Event>) {
    let ctx = app.egui_ctx.clone();
    // The modifiers held are the ones a key in this frame was pressed with, as a window reports
    // them: egui reads the held state beside each event, and a binding is looked up by it.
    let modifiers = events.iter().fold(egui::Modifiers::NONE, |held, e| match e {
        egui::Event::Key { modifiers, .. } => held | *modifiers,
        _ => held,
    });
    let mut input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))),
        events,
        modifiers,
        ..Default::default()
    };
    app.raw_input_hook(&ctx, &mut input);
    let mut eframe_frame = eframe::Frame::_new_kittest();
    let _ = ctx.run_ui(input, |ui| app.ui(ui, &mut eframe_frame));
}

/// The harness itself: an `App` builds, runs frames, and a typed line and a dispatched operation
/// both reach the document.
#[test]
fn a_headless_app_runs_frames_and_reaches_the_document() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    assert_eq!(app.part_graph().nodes().count(), 0, "it opens on an empty part");
    app.dispatch("create.sphere", &Args::new());
    assert_eq!(app.part_graph().nodes().count(), 1);
    app.run_command("cube size=20 'box'");
    frame(&mut app, Vec::new());
    assert_eq!(app.part_graph().nodes().count(), 2);
    let named = names::in_use(&app.doc);
    assert!(named.iter().any(|(_, n)| n == "box"), "{named:?}");
}

/// **The View menu's settings run through the registry and reach what the old rows flipped by
/// hand** -- and a line typed reaches the same state, which is the point of giving each its
/// operation (P3: the GUI could set a display mode and the console could only cycle one).
#[test]
fn a_menu_setting_and_its_console_line_reach_the_same_state() {
    let Some(mut app) = headless_app() else { return };
    let by = |id: &str| commands::by_id(id).unwrap_or_else(|| panic!("{id} is in the registry"));

    app.invoke_command(by("view.mode.field"));
    assert!(app.view_mode == ViewMode::Field);
    app.run_command("display mode=overlay");
    assert!(app.view_mode == ViewMode::Overlay);

    app.invoke_command(by("view.quality.fast"));
    assert_eq!(app.display_quality, crate::display_quality::DisplayQuality::Fast);
    app.run_command("display quality=balanced");
    assert_eq!(app.display_quality, crate::display_quality::DisplayQuality::Balanced);

    let shown = app.show_control_points;
    app.invoke_command(by("view.control_points"));
    assert_eq!(app.show_control_points, !shown);
    app.run_command("points");
    assert_eq!(app.show_control_points, shown);

    let cube = app.show_view_cube;
    app.run_command("viewcube");
    assert_eq!(app.show_view_cube, !cube);

    app.invoke_command(by("help.about"));
    assert!(app.show_about);
    app.run_command("shortcuts");
    assert!(app.show_shortcuts);

    // A command that is not on offer says why and does nothing: Paste here with nothing copied.
    let curves = app.doc.curves.len();
    app.invoke_command(by("edit.paste_here"));
    assert_eq!(app.doc.curves.len(), curves, "refused while nothing is copied");
    assert!(app.sections().is_empty(), "no section until one is asked for");
    app.invoke_command(by("view.section"));
    assert_eq!(app.sections().len(), 1, "the first toggle makes one");
    // Flip section is gone from the menus and the console (the owner, 2026-11-08): the row's
    // own Flip is the one way to turn a section round.
    assert!(commands::by_id("view.section_flip").is_none());
    assert!(commands::console_words().iter().all(|(word, _)| word != "flipsection" && word != "flip"));
}

/// **A section is a view that stays on, and where it was, through the direct layer's edits**
/// (the owner, 2026-10-24). Made and on it has the handles; a cage face picked in the viewport
/// takes them and leaves the cut on and unmoved; dragging the face moves the cage and not the
/// plane; the row gives the handles back without moving it either. A second section cuts beside the
/// first, each row's switch is its own, and Delete with a row selected removes that section and
/// nothing of the part.
#[test]
fn the_section_stays_on_and_unmoved_while_a_cage_face_is_picked_and_moved() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    // Polygon mode, so a click picks on the cage: picking follows what the mode draws.
    app.dispatch("view.display", &Args::new().with("mode", Value::Str("polygon".into())));
    frame(&mut app, Vec::new());
    assert!(app.view_mode.cage_is_editable());

    app.invoke_command(by("view.section"));
    frame(&mut app, Vec::new());
    assert!(app.sections()[0].on && app.section_selected == Some(app.sections()[0].id), "turning it on reaches for it");
    let placed = app.sections()[0];

    // A point on a face on screen, as the camera draws it: one the pick agrees is that face and
    // that no handle of the section's manipulator is over -- the handles sit at the scene's
    // centre, which is where the cage's near faces project, and a press on a handle is the
    // handle's, as it should be. So the centre of each face and points toward its corners are
    // tried until one is clear.
    let rect = app.viewport_rect.expect("a frame sets the viewport");
    let aspect = rect.width() / rect.height().max(1.0);
    let placement = app.control_placement();
    let gizmo = app.gizmo_frame().expect("the section has the handles");
    let eye = app.camera.eye.cast::<f64>();
    let (face, at) = {
        let project = app.gizmo_project(aspect, rect);
        app.cage()
        .face_ids()
        .find_map(|f| {
            let corners: Vec<Point3<f64>> = app
                .cage()
                .face_vertices(f)
                .iter()
                .map(|&v| placement * app.cage().vertex_position(v))
                .collect();
            let centre = Point3::from(corners.iter().map(|c| c.coords).sum::<Vector3<f64>>() / 4.0);
            std::iter::once(centre)
                .chain(corners.iter().map(|c| centre + (c - centre) * 0.6))
                .find_map(|p| {
                    let at = app.camera.project_to_screen(aspect, rect, p.cast::<f32>())?;
                    let under = crate::pick::pick(app.control(), &placement, &app.cage_mesh, &app.camera, aspect, rect, at);
                    let clear = gizmo::hit_test(&gizmo, at, &project, eye).is_none();
                    (under == Selection::Face(f) && clear).then_some((f, at))
                })
        })
        .expect("some point of some face faces the camera clear of the handles")
    };
    left_click(&mut app, at);
    assert!(app.selection.contains(Selection::Face(face)), "the click picked the face");
    assert!(app.sections()[0].on, "picking leaves the section on");
    assert!(app.section_selected.is_none(), "and takes the handles from it");

    // Dragging the face: a press on what is picked, pulled across the view, let go.
    let vertices = app.cage().face_vertices(face);
    let before: Vec<Point3<f64>> = vertices.iter().map(|&v| app.cage().vertex_position(v)).collect();
    frame(&mut app, vec![press(at, egui::PointerButton::Primary, true)]);
    let mut pos = at;
    for _ in 0..6 {
        pos += egui::vec2(15.0, 0.0);
        frame(&mut app, vec![egui::Event::PointerMoved(pos)]);
    }
    frame(&mut app, vec![press(pos, egui::PointerButton::Primary, false)]);
    frame(&mut app, Vec::new());
    let after: Vec<Point3<f64>> = vertices.iter().map(|&v| app.cage().vertex_position(v)).collect();
    assert!(
        before.iter().zip(&after).any(|(b, a)| (a - b).norm() > 1e-6),
        "the drag moved the face: {before:?} -> {after:?}"
    );
    assert!(app.sections()[0].on, "the cut is still on after the edit");
    assert_eq!(app.sections()[0].origin, placed.origin, "and the plane did not move with the face");
    assert_eq!(app.sections()[0].rotation, placed.rotation);
    assert_eq!(app.sections()[0].flipped, placed.flipped);

    // The row gives the handles back and moves nothing; the face is no longer picked.
    app.select_section(0);
    assert!(app.sections()[0].on && app.section_selected == Some(app.sections()[0].id));
    assert!(app.selection.is_empty());
    assert_eq!(app.sections()[0].origin, placed.origin);
    // And the switch is the only thing that turns it off: the row stays, with nothing to move.
    app.invoke_command(by("view.section"));
    assert!(!app.sections()[0].on && app.selected_section().is_none());
    assert!(app.section_half_spaces().is_empty(), "nothing is cut while none is on");
    app.invoke_command(by("view.section"));
    assert!(app.sections()[0].on, "the toggle switches the current one back on, it makes no second");
    assert_eq!(app.sections().len(), 1);

    // A second section: on beside the first, numbered after it, and the manipulator's; both cut.
    app.invoke_command(by("view.section_new"));
    assert_eq!(app.sections().len(), 2);
    assert!(app.sections()[0].on && app.sections()[1].on, "several may be on at once");
    assert_eq!(app.section_half_spaces().len(), 2);
    assert_eq!((app.sections()[0].number, app.sections()[1].number), (1, 2));
    assert_eq!(app.section_selected, Some(app.sections()[1].id));
    // The first's row takes the handles and switches nothing off; its own switch switches it off.
    app.select_section(0);
    assert!(app.sections()[0].on && app.sections()[1].on && app.section_selected == Some(app.sections()[0].id));
    assert!(app.set_section_on(0, false));
    assert!(!app.sections()[0].on && app.sections()[1].on);
    assert_eq!(app.section_half_spaces().len(), 1);
    // Delete, with a section's row selected, removes that section and nothing of the part; the
    // one behind it keeps its number, and the claim goes with the row.
    let bodies = app.part_graph().roots().len();
    app.invoke_command(by("edit.delete_body"));
    assert_eq!(app.sections().len(), 1);
    assert_eq!(app.sections()[0].number, 2);
    assert!(app.section_selected.is_none());
    assert_eq!(app.part_graph().roots().len(), bodies, "the part is untouched");
    app.remove_section(0);
    assert!(app.sections().is_empty() && app.section_half_spaces().is_empty());
}

// ----------------------------------------------------------------------- the smoke scripts

/// **The smoke scripts pass on a headless `App`**, every `expect` in them, run line by line as
/// `--run … --keep-going` runs them (`benchmarks/smoke/`, the project notes). They are the console's own
/// end-to-end record, and this is what makes them part of `cargo test` rather than a run someone
/// remembers to do. A file a script saves or opens is redirected to a folder of this test's own,
/// so a run never writes into a checkout, and a `snap` -- a picture to look at, not a check -- is
/// left out.
#[test]
fn every_smoke_script_passes_every_expect() {
    let Some(mut app) = headless_app() else { return };
    let scripts = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/smoke");
    let out = std::env::temp_dir().join(format!("fieldform-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&out).expect("a folder for what the scripts save");
    let redirect = |line: &str| -> Option<String> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("snap ") {
            return None;
        }
        // The key `file=` as a word of its own, never the end of `profile=`.
        let Some(at) = trimmed.find(" file=").map(|i| i + 1) else { return Some(trimmed.to_string()) };
        let rest = &trimmed[at + 5..];
        let (path, tail) = match rest.strip_prefix('"') {
            Some(quoted) => match quoted.find('"') {
                Some(end) => (&quoted[..end], &quoted[end + 1..]),
                None => (quoted, ""),
            },
            None => match rest.find(char::is_whitespace) {
                Some(end) => (&rest[..end], &rest[end..]),
                None => (rest, ""),
            },
        };
        let name = std::path::Path::new(path).file_name().expect("a file name").to_string_lossy();
        Some(format!("{}file=\"{}\"{tail}", &trimmed[..at], out.join(&*name).display()))
    };
    let mut failures = Vec::new();
    for script in ["phase-a.ffc", "curves.ffc", "gesture-undo.ffc", "ownership.ffc"] {
        let text = std::fs::read_to_string(scripts.join(script)).expect("the script");
        app.mark_saved();
        app.dispatch("file.new", &Args::new());
        let first = app.expect_results.len();
        for line in text.lines().filter_map(redirect) {
            app.gesture = None;
            let said = app.console_log.len();
            app.run_command(&line);
            app.drain_ownership_changes(false);
            frame(&mut app, Vec::new());
            if std::env::var("SMOKE_TRACE").is_ok() {
                for l in app.console_log.iter().skip(said.min(app.console_log.len())) {
                    eprintln!("{} {}", if l.ok { "  " } else { "!!" }, l.text);
                }
            }
        }
        for r in &app.expect_results[first..] {
            if r.verdict != expect::Verdict::Pass {
                failures.push(format!("{script}: {}", r.said));
            }
        }
        assert!(app.expect_results.len() > first, "{script} measured nothing");
    }
    let _ = std::fs::remove_dir_all(&out);
    assert!(failures.is_empty(), "smoke expects that did not pass:\n{}", failures.join("\n"));
}

// ------------------------------------------------------------------------------------- T1

/// A document written by name rather than by random id, so two documents built apart can be
/// compared: every named block as its node with every id it holds replaced by what that id is
/// called, every curve by its points, every parameter by its source, and which blocks are
/// variables (§6.3: same blocks, values, ids, connections and expression sources).
fn named_form(app: &App) -> std::collections::BTreeMap<String, String> {
    let doc = &app.doc;
    let graph = app.part_graph();
    let mut names: Vec<(String, String)> = Vec::new();
    for (id, _) in graph.nodes() {
        if let Some(n) = doc.node_meta.get(&id).and_then(|m| m.name.clone()) {
            names.push((format!("\"{id}\""), format!("\"block:{n}\"")));
        }
    }
    for c in doc.curves.values() {
        names.push((format!("\"{}\"", c.id), format!("\"curve:{}\"", c.name)));
    }
    for p in doc.parameters.values() {
        names.push((format!("\"{}\"", p.id), format!("\"param:{}\"", p.name)));
    }
    let rename = |mut text: String| {
        for (id, name) in &names {
            text = text.replace(id, name);
        }
        text
    };
    let mut out = std::collections::BTreeMap::new();
    let mut unnamed = Vec::new();
    for (id, node) in graph.nodes() {
        let text = rename(serde_json::to_string(node).expect("a node serializes"));
        match doc.node_meta.get(&id).and_then(|m| m.name.clone()) {
            Some(n) => {
                let variable = doc.ownership_nodes.get(&id) == Some(&fieldform_scene::Ownership::Variable);
                out.insert(format!("block {n}"), format!("{text}{}", if variable { " (variable)" } else { "" }));
            }
            None => unnamed.push(text),
        }
    }
    unnamed.sort();
    for (i, text) in unnamed.into_iter().enumerate() {
        out.insert(format!("unnamed {i}"), text);
    }
    for c in doc.curves.values() {
        out.insert(format!("curve {}", c.name), format!("{} {:?} {:?}", c.degree(), c.control_points(), c.weights()));
    }
    for p in doc.parameters.values() {
        let said = match p.value.expression() {
            Some(e) => format!("= {}", e.source()),
            None => format!("{:?}", p.value.as_literal()),
        };
        out.insert(format!("param {}", p.name), said);
    }
    // A profile's name, which is all a profile row holds beyond the curves already compared.
    for name in doc.profile_names.values() {
        out.insert(format!("profile {name}"), String::new());
    }
    out
}

fn by(id: &str) -> &'static commands::Command {
    commands::by_id(id).unwrap_or_else(|| panic!("{id} is in the registry"))
}

fn curve_points(app: &App, name: &str) -> Vec<nalgebra::Point3<f64>> {
    let id = app.curve_named(name).unwrap_or_else(|| panic!("no curve is called {name}"));
    app.doc.curve(id).expect("the curve").control_points().to_vec()
}

/// **A 2D sketch follows its plane, and a point taken off the plane makes it a 3D one.** A line
/// drawn while sketching on a plane on a surface is laid in the plane's coordinates; move the body
/// under the plane and the line goes with it. Edit a point across the plane and the sketch keeps
/// its new shape through the next move; take one off the plane and it stops following, and says so.
#[test]
fn a_sketch_follows_its_plane_until_a_point_leaves_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cube size=40,20,10 'block'");
    app.run_command("plane on=block at=0,0,5 'lid'");
    frame(&mut app, Vec::new());
    let lid = app.node_named("lid").expect("made");
    select(&mut app, "lid");
    app.run_command("sketch");
    // Typed in world, flat on the lid: a sketch there. And typed across the plane with `on=`.
    app.run_command("line p0=2,3,5 p1=12,3,5 'edge'");
    app.run_command("line on=lid p0=0,0 p1=0,8 'rib'");
    frame(&mut app, Vec::new());
    let edge = app.curve_named("edge").expect("made");
    let rib = app.curve_named("rib").expect("made");
    assert!(app.doc.sketches.get(&edge).is_some_and(|s| s.live && s.plane == lid), "drawn while sketching, so a sketch");
    assert!(app.doc.sketches.get(&rib).is_some_and(|s| s.live && s.plane == lid));
    // To the lid's own precision: it stands on a point bisected to a thousandth of the block.
    let near = |a: nalgebra::Point3<f64>, b: nalgebra::Point3<f64>| (a - b).norm() < 1e-3;
    let before = curve_points(&app, "rib");
    assert!(near(before[1], nalgebra::Point3::new(0.0, 8.0, 5.0)), "across the lid: {:?}", before);
    app.run_command("sketch");

    // The body moves, the lid with it, and the sketches with the lid.
    app.run_command("set block.centre=30,0,0");
    frame(&mut app, Vec::new());
    let moved = curve_points(&app, "edge");
    assert!(near(moved[0], nalgebra::Point3::new(32.0, 3.0, 5.0)), "{moved:?}");
    assert!(near(curve_points(&app, "rib")[1], nalgebra::Point3::new(30.0, 8.0, 5.0)));

    // Edited across the plane: the sketch keeps its new shape through the next move.
    app.run_command("set edge.p1=40,3,5");
    frame(&mut app, Vec::new());
    assert!(app.doc.sketches[&edge].live, "still flat, still a sketch");
    app.run_command("set block.centre=0,0,0");
    frame(&mut app, Vec::new());
    let back = curve_points(&app, "edge");
    assert!(near(back[1], nalgebra::Point3::new(10.0, 3.0, 5.0)), "the edit came along: {back:?}");

    // A point taken off the plane: a 3D sketch now, and the console says so.
    let said = app.console_log.len();
    app.run_command("set rib.p1=0,8,20");
    frame(&mut app, Vec::new());
    assert!(!app.doc.sketches[&rib].live, "no longer following");
    assert!(app.console_log[said..].iter().any(|l| l.text.contains("3D sketch")), "and it says so");
    app.run_command("set block.centre=30,0,0");
    frame(&mut app, Vec::new());
    assert!(near(curve_points(&app, "rib")[0], nalgebra::Point3::new(0.0, 0.0, 5.0)), "left behind");
    assert!(near(curve_points(&app, "edge")[0], nalgebra::Point3::new(32.0, 3.0, 5.0)), "the 2D one still follows");

    // Laid back onto the plane on purpose: flat on it, and following again.
    app.lie_on_plane(rib, lid);
    frame(&mut app, Vec::new());
    assert!(app.doc.sketches[&rib].live);
    // Projected straight onto the lid's plane, where it stands: x stays, z comes down to the lid.
    let laid = curve_points(&app, "rib");
    assert!(near(laid[0], nalgebra::Point3::new(0.0, 0.0, 5.0)) && near(laid[1], nalgebra::Point3::new(0.0, 8.0, 5.0)), "{laid:?}");
    app.run_command("set block.centre=40,0,0");
    frame(&mut app, Vec::new());
    assert!(near(curve_points(&app, "rib")[1], nalgebra::Point3::new(10.0, 8.0, 5.0)), "and it follows again");
    app.run_command("undo");
    frame(&mut app, Vec::new());
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert!(!app.doc.sketches[&rib].live && (curve_points(&app, "rib")[1].z - 20.0).abs() < 1e-6, "one step back");
}

/// **While sketching, the handles are the plane's, and a curve's degree is one line away.** On a
/// plane standing on a side wall the manipulator's Z is the wall's normal and only the in-plane
/// moves are offered; `degree degree=3` raises the selected line and keeps its ends.
#[test]
fn sketch_handles_are_the_planes_and_degree_is_one_line() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cube size=40,20,10 'block'");
    app.run_command("plane on=block at=-20,3,2 'side'");
    frame(&mut app, Vec::new());
    select(&mut app, "side");
    app.run_command("sketch");
    app.run_command("line on=side p0=0,0 p1=5,0 'l'");
    frame(&mut app, Vec::new());
    let gizmo = app.gizmo_frame().expect("handles on the selected line");
    let z = gizmo.basis * nalgebra::Vector3::z();
    assert!((z + nalgebra::Vector3::x()).norm() < 1e-6, "Z is the wall's normal: {z}");
    let handles = gizmo.handles.all();
    assert!(!handles.contains(&gizmo::Handle::Axis(2)), "no move out of the plane: {handles:?}");
    assert!(handles.contains(&gizmo::Handle::Plane(2)) && handles.contains(&gizmo::Handle::Arc(2)), "{handles:?}");

    let l = app.curve_named("l").expect("made");
    app.run_command("degree degree=3");
    frame(&mut app, Vec::new());
    assert_eq!(app.doc.curve(l).map(|c| c.degree()), Some(3));
    assert!(app.doc.sketches[&l].live, "raising the degree keeps it on its plane");
    app.run_command("undo");
    assert_eq!(app.doc.curve(l).map(|c| c.degree()), Some(1));
    app.run_command("sketch off=true");
    assert!(app.gizmo_frame().is_some_and(|g| g.basis == nalgebra::Rotation3::identity()), "world-aligned again");
}

/// **A sketch's echo says its plane and its points across it, and replays to the same curve.**
#[test]
fn a_sketch_echoes_as_a_curve_on_its_plane() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cube size=40,20,10 'block'");
    app.run_command("plane on=block at=0,0,5 'lid'");
    frame(&mut app, Vec::new());
    let before = echo::Snapshot::of(&app.doc, app.part_graph());
    app.run_command("line on=lid p0=1,2 p1=9,2 'rib'");
    frame(&mut app, Vec::new());
    let lines = echo::lines(&before, &app.doc, app.part_graph(), &[]);
    let line = lines.iter().find(|l| l.starts_with("curve on=lid")).unwrap_or_else(|| panic!("{lines:?}"));
    assert!(line.contains("p=1,2;9,2"), "{line}");
    let copy = line.replace("'rib'", "'copy'");
    app.run_command(&copy);
    frame(&mut app, Vec::new());
    assert_eq!(curve_points(&app, "copy"), curve_points(&app, "rib"));
    assert!(app.doc.sketches.contains_key(&app.curve_named("copy").unwrap()));
}

/// **Sketch on a plane is one act, and stopping gives the view back**: the plane becomes the
/// drawing plane and the projection goes orthographic together; a point built now lies on the
/// plane; Esc with nothing pending, or the command again, clears the plane and restores the
/// projection the sketch began from.
#[test]
fn sketch_on_a_plane_turns_the_view_and_stopping_gives_it_back() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cube size=40,20,10 'block'");
    app.run_command("plane on=block at=0,0,5 'lid'");
    frame(&mut app, Vec::new());
    let lid = app.node_named("lid").expect("made");
    let before = app.camera.projection;
    assert!(!app.sketching());

    select(&mut app, "lid");
    app.invoke_command(by("draw.sketch"));
    assert!(app.drawing_plane == crate::DrawingPlane::Node(lid), "the plane is what is drawn on");
    assert_eq!(app.camera.projection, crate::display_quality::Projection::Orthographic);
    // To the point's own solve tolerance: the lid stands on a surface point, which is found by
    // bisection to a thousandth of the body.
    let on = app.into_drawing_plane(nalgebra::Point3::new(1.0, 2.0, 0.0));
    assert!((on.z - 5.0).abs() < 1e-3, "a point built now lies on the lid: {on}");

    assert!(app.escape_stops_sketching(), "Esc with nothing pending stops it");
    assert!(app.drawing_plane == crate::DrawingPlane::None);
    assert_eq!(app.camera.projection, before, "and the projection is the one it began from");
    assert!(!app.escape_stops_sketching(), "a second Esc is not this feature's");

    // The command again turns square to the plane again rather than stopping (the plane has
    // usually moved); `sketch off`, or `sketch` with nothing selected, stops.
    app.run_command("sketch");
    assert!(app.sketching());
    app.run_command("sketch");
    assert!(app.sketching(), "again is a re-orientation, not a stop");
    app.run_command("sketch off=true");
    assert!(!app.sketching() && app.camera.projection == before);
    app.run_command("sketch target=lid");
    app.select_body(None);
    assert!(app.sketching());
    app.run_command("sketch");
    assert!(!app.sketching() && app.camera.projection == before);
}

fn select(app: &mut App, name: &str) {
    let id = app.node_named(name).unwrap_or_else(|| panic!("nothing is called {name}"));
    app.select_body(Some(id));
}

fn edit(app: &mut App, name: &str, change: impl FnOnce(&mut Node)) {
    let id = app.node_named(name).unwrap_or_else(|| panic!("nothing is called {name}"));
    let mut node = app.part_graph().node(id).cloned().expect("the block");
    change(&mut node);
    // A separate edit, as a person makes one: the gesture window would otherwise fold two edits
    // a test makes a millisecond apart into one undo entry.
    app.gesture = None;
    app.replace_node(id, node);
    frame(app, Vec::new());
}

/// Replays `lines` in a fresh `App` through the console, refusing on the first warning.
fn replay(lines: &[String]) -> App {
    let mut app = headless_app().expect("an adapter, as the session had one");
    for line in lines {
        let warned = app.console_warnings;
        app.run_command(line);
        frame(&mut app, Vec::new());
        if app.console_warnings > warned {
            let said: Vec<&str> = app.console_log.iter().rev().take(4).map(|l| l.text.as_str()).collect();
            panic!("replaying `{line}` was refused: {said:?}\nthe whole echo:\n{}", lines.join("\n"));
        }
    }
    app
}

fn assert_same_graph(gui: &App, replayed: &App, lines: &[String]) {
    let (a, b) = (named_form(gui), named_form(replayed));
    if a != b {
        let mut diff = String::new();
        for k in a.keys().chain(b.keys()).collect::<std::collections::BTreeSet<_>>() {
            if a.get(k) != b.get(k) {
                diff.push_str(&format!("\n{k}:\n  gui:    {:?}\n  replay: {:?}", a.get(k), b.get(k)));
            }
        }
        panic!("the replay is not the graph the GUI built:{diff}\n\nthe echo:\n{}", lines.join("\n"));
    }
}

/// **T1: the echo of a GUI session, replayed in a fresh document, builds the same graph**
/// (`interaction-spec.md` §6.3, §11). Every block the menu makes, edits as the panel and the
/// handles make them, a parameter whose expression is typed on its row and a field reading it,
/// curves and every block built on them, every modifier, a re-edit that turns a boolean into a
/// blend, an undo and a delete -- written by the echo, run line by line through the console of an
/// `App` that saw none of it, and compared block by block by name.
#[test]
fn t1_the_echo_of_a_gui_session_replays_to_the_same_graph() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    // What an empty part says on its first frame is not the session's.
    let said_before = app.console_log.len();
    // A command from the menu, and -- where it leaves its block pending -- Enter to keep it, as a
    // person would before going on.
    let invoke = |app: &mut App, id: &str| {
        app.invoke_command(by(id));
        frame(app, Vec::new());
        if app.session.is_some() {
            frame(app, vec![key(egui::Key::Enter)]);
        }
    };

    // Every primitive from the menu.
    for id in [
        "create.sphere", "create.cube", "create.cylinder", "create.cone", "create.torus",
        "create.capsule", "create.skewbox", "create.point", "create.plane",
    ] {
        invoke(&mut app, id);
    }
    // Edits as the panel and a handle make them: a block's numbers rewritten in place.
    edit(&mut app, "Sphere_0", |n| {
        if let Node::Sphere { radius, centre } = n {
            *radius = ParamRef::Literal(37.5);
            *centre = fieldform_scene::Place::at(10.0, 0.1, -3.0);
        }
    });
    edit(&mut app, "Cube_0", |n| {
        if let Node::Cuboid { size, .. } = n {
            *size = [ParamRef::Literal(10.0), ParamRef::Literal(20.25), ParamRef::Literal(1.0 / 3.0)];
        }
    });
    // A Number from the menu, its expression typed on its row, and a field reading it.
    invoke(&mut app, "create.number");
    let number = *app.doc.parameters.keys().next().expect("the Number made a parameter");
    app.commit_expression(number, "12.5 * 2").expect("an expression");
    // The row keeps the keyboard until the pointer goes elsewhere, and the echo waits for typing to
    // end; a click on the viewport is what ends it.
    if let Some(focused) = app.egui_ctx.memory(|m| m.focused()) {
        app.egui_ctx.memory_mut(|m| m.surrender_focus(focused));
    }
    frame(&mut app, Vec::new());
    edit(&mut app, "Torus_0", |n| {
        if let Node::Torus { major_radius, .. } = n {
            *major_radius = ParamRef::Named(number);
        }
    });

    // Curves, and the blocks built on them.
    app.dispatch("create.line", &Args::new().with("p0", Value::Str("0,0,0".into())).with("p1", Value::Str("0,0,80".into())));
    frame(&mut app, Vec::new());
    // A closed profile as one is drawn: four lines, end to end, which close a loop.
    let line = |app: &mut App, a: &str, b: &str| {
        app.dispatch("create.line", &Args::new().with("p0", Value::Str(a.into())).with("p1", Value::Str(b.into())));
        frame(app, Vec::new());
    };
    line(&mut app, "60,0,0", "90,0,0");
    line(&mut app, "90,0,0", "90,30,0");
    line(&mut app, "90,30,0", "60,30,0");
    line(&mut app, "60,30,0", "60,0,0");
    invoke(&mut app, "create.extrude");
    invoke(&mut app, "create.revolve");
    invoke(&mut app, "create.sweep");
    // A second loop above the first, and a loft through both.
    line(&mut app, "60,0,40", "90,0,40");
    line(&mut app, "90,0,40", "90,30,40");
    line(&mut app, "90,30,40", "60,30,40");
    line(&mut app, "60,30,40", "60,0,40");
    invoke(&mut app, "create.loft");
    // A curve of higher degree, and a pipe along it.
    let arc = "0,60,0;40,100,10;80,60,20";
    app.dispatch("create.curve", &Args::new().with("p", Value::Str(arc.into())).with("degree", Value::Int(2)));
    frame(&mut app, Vec::new());
    invoke(&mut app, "create.pipe");

    // Every modifier, on a body selected first.
    for (body, id) in [
        ("Sphere_0", "modify.offset"),
        ("Cube_0", "modify.shell"),
        ("Cylinder_0", "modify.twist"),
        ("Cone_0", "modify.mirror"),
        ("Capsule_0", "modify.pattern"),
        ("SkewedBox_0", "modify.attract"),
        ("Torus_0", "modify.infill"),
    ] {
        select(&mut app, body);
        invoke(&mut app, id);
    }
    invoke(&mut app, "create.boolean");
    invoke(&mut app, "modify.morph");
    invoke(&mut app, "create.tpms");
    invoke(&mut app, "create.ramp");
    // Placements as the manipulator leaves them: a move, then a turn that does not fold into it.
    let place = |app: &mut App, op: &str, target: &str, by: &str| {
        app.gesture = None;
        app.dispatch(op, &Args::new().with("target", Value::Str(target.into())).with("by", Value::Str(by.into())));
        frame(app, Vec::new());
    };
    place(&mut app, "modify.move", "Shell_0", "5,0,2.5");
    place(&mut app, "modify.turn", "Placement_0", "0,0,30");
    // A body split in two from the menu, and a block made a variable by its V.
    select(&mut app, "Twist_0");
    invoke(&mut app, "modify.split");
    app.dispatch("block.make_variable", &Args::new().with("name", Value::Str("Cube_0".into())));
    frame(&mut app, Vec::new());

    // Renames as each kind of row's own field writes one, the block's with an edit in the same
    // breath: the echo has to say the rename before the `set` that calls the block by its new name.
    let sphere = app.node_named("Sphere_0").expect("the sphere");
    assert!(app.set_body_name(sphere, "ball"));
    edit(&mut app, "ball", |n| {
        if let Node::Sphere { radius, .. } = n {
            *radius = ParamRef::Literal(30.0);
        }
    });
    let arc = app.doc.curves.values().find(|c| c.degree() == 2).map(|c| c.id).expect("the arc");
    app.rename_curve(arc, "rail");
    app.doc.rename_parameter(number, "wall").expect("a free name");
    app.mark_changed();
    app.refresh_profiles();
    let base = app.profiles.loops[0][0];
    app.rename_profile(base, "base");
    frame(&mut app, Vec::new());

    // Re-edits of committed blocks: a depth, a boolean given a fillet (it becomes a blend), a
    // pattern's count.
    edit(&mut app, "Extrude_0", |n| {
        if let Node::Extrude { height, rims, .. } = n {
            *height = ParamRef::Literal(42.0);
            *rims = fieldform_scene::Rims { radius: ParamRef::Literal(1.5), continuity: fieldform_scene::Continuity::G2, at: fieldform_scene::RimEnds::End, second: None };
        }
    });
    edit(&mut app, "Loft_0", |n| {
        if let Node::Loft { rims, .. } = n {
            *rims = fieldform_scene::Rims { radius: ParamRef::Literal(1.0), continuity: fieldform_scene::Continuity::Chamfer, at: fieldform_scene::RimEnds::Both, second: None };
        }
    });
    edit(&mut app, "Infill_0", |n| {
        if let Node::Infill { wall, join, .. } = n {
            *wall = ParamRef::Literal(1.25);
            *join = fieldform_scene::Fillet { radius: ParamRef::Literal(0.5), continuity: fieldform_scene::Continuity::G2, second: None };
        }
    });
    edit(&mut app, "Boolean_0", |n| {
        if let Node::Boolean { op, inputs } = n.clone()
            && let [a, b] = inputs[..]
        {
            *n = Node::Blend {
                op,
                inputs: vec![a, b],
                fillet: fieldform_scene::Fillet { radius: ParamRef::Literal(2.0), continuity: fieldform_scene::Continuity::G2, second: None },
            };
        }
    });
    edit(&mut app, "Pattern_0", |n| {
        if let Node::Pattern { count, .. } = n {
            *count = ParamRef::Literal(5.0);
        }
    });
    // An edit taken back, and a block deleted, from the Edit menu.
    edit(&mut app, "Torus_0", |n| {
        if let Node::Torus { minor_radius, .. } = n {
            *minor_radius = ParamRef::Literal(4.0);
        }
    });
    invoke(&mut app, "edit.undo");
    select(&mut app, "Point_0");
    invoke(&mut app, "edit.delete_body");

    // The session itself refused nothing it was asked: a refusal here is a step the echo then has
    // nothing to say about, and the test would be checking less than it reads as checking.
    let refused: Vec<&str> = app.console_log[said_before..].iter().filter(|l| !l.ok).map(|l| l.text.as_str()).collect();
    assert!(refused.is_empty(), "the GUI session was refused: {refused:#?}");
    let lines = app.echo_log.clone();
    eprintln!("T1: the session echoed {} lines", lines.len());
    assert!(lines.len() >= 30, "the session echoed {} lines:\n{}", lines.len(), lines.join("\n"));
    let replayed = replay(&lines);
    assert_same_graph(&app, &replayed, &lines);
}

// ------------------------------------------------------------------------------ T3, T4, T5

/// The document as the file would hold it: the archive's bytes, which is what "byte-identical"
/// is about (§11, T3).
fn document_bytes(app: &App) -> Vec<u8> {
    let archive = fieldform_io::to_archive(&app.doc).expect("the document writes");
    let mut bytes = std::io::Cursor::new(Vec::new());
    archive.write_to(&mut bytes).expect("the archive writes");
    bytes.into_inner()
}

/// Edits the block a session is waiting on, as its panel does: in place, the selection unmoved.
fn edit_pending(app: &mut App, change: impl FnOnce(&mut Node)) {
    let id = app.session.as_ref().expect("a session is pending").block;
    let mut node = app.part_graph().node(id).cloned().expect("the pending block");
    change(&mut node);
    app.gesture = None;
    app.replace_node(id, node);
    frame(app, Vec::new());
}

/// **T3: a cancelled session leaves the document, its names and the history byte for byte as
/// they were** (`interaction-spec.md` §3.5, §11). A history with something to redo, a session
/// begun from the menu and edited twice, then Esc: the file the document would write, the undo and
/// redo lists, the name the next block of that kind would be given and the echo are all unchanged.
#[test]
fn t3_a_cancelled_session_leaves_the_document_its_names_and_the_history_as_they_were() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere 'a'");
    app.run_command("cube 'b'");
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert_eq!(document_bytes(&app), document_bytes(&app), "the archive is the same bytes each time it is written");
    let before = (document_bytes(&app), app.stack.history(), names::instance_name(&app.doc, "Cylinder"), app.echo_log.len());
    assert!(!before.1.1.is_empty(), "there is something to redo, which a cancel must not lose");

    app.invoke_command(by("create.cylinder"));
    frame(&mut app, Vec::new());
    assert!(app.session.is_some(), "the cylinder is pending");
    for r in [7.0, 9.5] {
        edit_pending(&mut app, |n| {
            if let Node::Cylinder { radius, .. } = n {
                *radius = ParamRef::Literal(r);
            }
        });
    }
    frame(&mut app, vec![key(egui::Key::Escape)]);

    assert!(app.session.is_none(), "Esc takes it back");
    let after = (document_bytes(&app), app.stack.history(), names::instance_name(&app.doc, "Cylinder"), app.echo_log.len());
    assert!(after.0 == before.0, "the document is not byte-identical after a cancel");
    assert_eq!(after.1, before.1, "the history, redo included");
    assert_eq!(after.2, before.2, "no name was taken");
    assert_eq!(after.3, before.3, "nothing was echoed");
}

/// **T4: a session with N edits is one undo step, and undo removes the block entirely**
/// (§3.4, §11). Three edits to a pending sphere, Enter, and the history holds one entry more,
/// named for the command, which an undo takes back whole.
#[test]
fn t4_a_session_is_one_undo_step_and_undo_removes_its_block() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    let (depth, nodes) = (app.stack.history().0.len(), app.part_graph().nodes().count());
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    for r in [10.0, 20.0, 30.0] {
        edit_pending(&mut app, |n| {
            if let Node::Sphere { radius, .. } = n {
                *radius = ParamRef::Literal(r);
            }
        });
    }
    frame(&mut app, vec![key(egui::Key::Enter)]);
    assert!(app.session.is_none(), "Enter keeps it");
    let (done, _) = app.stack.history();
    assert_eq!(done.len(), depth + 1, "one entry for the making and three edits: {done:?}");
    assert_eq!(done.last().map(String::as_str), Some("Sphere"));
    assert_eq!(app.part_graph().nodes().count(), nodes + 1);
    assert!(
        app.echo_log.last().is_some_and(|l| l.starts_with("sphere radius=30 ")),
        "the echo carries the values it was kept with: {:?}",
        app.echo_log
    );
    app.invoke_command(by("edit.undo"));
    frame(&mut app, Vec::new());
    assert_eq!(app.part_graph().nodes().count(), nodes, "undo removes the block entirely");
}

/// **T5: running another command while one is pending keeps the first, then starts the second**
/// (§3.4, §11) -- and the next session of a command starts from what the last one was kept with
/// (§3.2).
#[test]
fn t5_another_command_keeps_the_pending_one_first() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    let depth = app.stack.history().0.len();
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    edit_pending(&mut app, |n| {
        if let Node::Sphere { radius, .. } = n {
            *radius = ParamRef::Literal(12.0);
        }
    });
    let sphere = app.session.as_ref().expect("pending").block;
    app.invoke_command(by("create.cube"));
    frame(&mut app, Vec::new());
    let pending = app.session.as_ref().expect("the cube is pending");
    assert!(matches!(app.part_graph().node(pending.block), Some(Node::Cuboid { .. })));
    let (done, _) = app.stack.history();
    assert_eq!(done.len(), depth + 2, "the sphere kept as one entry, and the cube's making: {done:?}");
    assert_eq!(done[depth], "Sphere");
    assert_eq!(app.body_name(sphere), Some("Sphere_0"), "kept, so named");
    assert!(app.echo_log.iter().any(|l| l.starts_with("sphere radius=12 ")), "{:?}", app.echo_log);
    assert!(!app.echo_log.iter().any(|l| l.starts_with("cube ")), "the cube is not kept yet");

    frame(&mut app, vec![key(egui::Key::Escape)]);
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    let again = app.session.as_ref().expect("a second sphere").block;
    assert!(
        matches!(app.part_graph().node(again), Some(Node::Sphere { radius: ParamRef::Literal(r), .. }) if *r == 12.0),
        "a sphere starts from the radius the last was kept with"
    );
}

// ------------------------------------------------------------------------------------- T6

/// One character typed, as egui hears it: the key, where it has one, and the text.
fn typed(c: char) -> Vec<egui::Event> {
    let mut events = Vec::new();
    if let Some(key) = egui::Key::from_name(&c.to_ascii_uppercase().to_string()) {
        events.push(key_event(key));
    }
    events.push(egui::Event::Text(c.to_string()));
    events
}

fn key_event(key: egui::Key) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }
}

fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        frame(app, typed(c));
    }
}

/// **T6: typing `2`, `0`, Enter, Enter into a fresh session sets its first number to 20 and keeps
/// it; an expression typed into a row echoes as its source text** (`interaction-spec.md` §4.3,
/// §6.3, §11). No click anywhere: the extrude's panel takes the digits because a session is
/// pending and nothing else holds the keyboard. Then `radius+2` into a sphere's radius -- the `r`
/// starts typing because a parameter's name starts with it -- becomes a variable named after the
/// row, and the echo writes the expression, not the twelve it comes to. Then Tab, which confirms a
/// row and moves to the next.
#[test]
fn t6_typed_entry_sets_the_first_number_and_an_expression_echoes_as_typed() {
    let Some(mut app) = headless_app() else { return };
    // The floating panel is what this holds, and it is off by default since the owner's trial.
    app.floating_panel = true;
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=30,0,0",
        "line p0=30,0,0 p1=30,20,0",
        "line p0=30,20,0 p1=0,20,0",
        "line p0=0,20,0 p1=0,0,0",
    ] {
        app.run_command(line);
        frame(&mut app, Vec::new());
    }
    app.invoke_command(by("create.extrude"));
    frame(&mut app, Vec::new());
    let extrude = app.session.as_ref().expect("the extrude is pending").block;
    assert_eq!(app.panel_rows().first().map(|r| r.key), Some("height"), "the first number is the depth");
    type_text(&mut app, "20");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    assert!(app.session.is_some(), "the first Enter confirms the row, not the session");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    assert!(app.session.is_none(), "the second Enter keeps it");
    assert!(
        matches!(app.part_graph().node(extrude), Some(Node::Extrude { height: ParamRef::Literal(h), .. }) if *h == 20.0),
        "{:?}",
        app.part_graph().node(extrude)
    );
    let extruded = app.echo_log.last().cloned().unwrap_or_default();
    assert!(extruded.starts_with("extrude ") && extruded.contains(" height=20 "), "{extruded}");

    // An expression, which a letter starting a parameter's name begins.
    app.run_command("param name=radius value=10");
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    let sphere = app.session.as_ref().expect("the sphere is pending").block;
    type_text(&mut app, "radius+2");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    assert!(app.session.is_none());
    let variable = app.doc.parameter_named("Radius_1").expect("a variable named after the row");
    assert!(matches!(app.part_graph().node(sphere), Some(Node::Sphere { radius: ParamRef::Named(id), .. }) if *id == variable));
    assert_eq!(app.doc.parameter_value(variable), Some(12.0));
    let n = app.echo_log.len();
    let said = &app.echo_log[n.saturating_sub(2)..];
    assert_eq!(said[0], "param name=Radius_1 expr=radius+2", "the source text, not the value: {said:?}");
    assert!(said[1].starts_with("sphere radius=Radius_1 "), "{said:?}");

    // Tab confirms the row it leaves and moves to the next.
    app.invoke_command(by("create.cube"));
    frame(&mut app, Vec::new());
    let cube = app.session.as_ref().expect("the cube is pending").block;
    type_text(&mut app, "5");
    frame(&mut app, vec![key_event(egui::Key::Tab)]);
    assert_eq!(app.session.as_ref().map(|s| s.panel.focus), Some(1), "Tab moves to the centre");
    type_text(&mut app, "1,2,3");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    let Some(Node::Cuboid { size, centre, .. }) = app.part_graph().node(cube).cloned() else { panic!("the cube") };
    assert_eq!(size, [ParamRef::Literal(5.0); 3]);
    assert_eq!(centre, fieldform_scene::Place::at(1.0, 2.0, 3.0));
}

/// **A middle-button drag on a number scrubs it, live, as one edit when the button lifts**
/// (`interaction-spec.md` §4.4): pressed on the sphere's radius and dragged forty pixels right, the
/// radius grows, and the session still holds one entry.
#[test]
fn a_middle_drag_on_a_panel_number_scrubs_it() {
    let Some(mut app) = headless_app() else { return };
    // The floating panel is what this holds, and it is off by default since the owner's trial.
    app.floating_panel = true;
    frame(&mut app, Vec::new());
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    frame(&mut app, Vec::new());
    let sphere = app.session.as_ref().expect("pending").block;
    let radius = |app: &App| match app.part_graph().node(sphere) {
        Some(Node::Sphere { radius: ParamRef::Literal(r), .. }) => *r,
        other => panic!("{other:?}"),
    };
    let before = radius(&app);
    let at = app.session.as_ref().expect("pending").panel.rects.first().copied().expect("the radius row was drawn").center();
    let button = |pos: egui::Pos2, pressed: bool| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Middle,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    frame(&mut app, vec![egui::Event::PointerMoved(at)]);
    frame(&mut app, vec![button(at, true)]);
    for step in 1..=4 {
        frame(&mut app, vec![egui::Event::PointerMoved(at + egui::vec2(10.0 * step as f32, 0.0))]);
    }
    frame(&mut app, vec![button(at + egui::vec2(40.0, 0.0), false)]);
    assert!(radius(&app) > before, "the drag grew the radius: {before} -> {}", radius(&app));
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    assert_eq!(app.stack.history().0.last().map(String::as_str), Some("Sphere"), "one entry for the session");
}

/// **A value that cannot be taken is said in the panel, and the session stays pending** (§3.6):
/// no dialog, and nothing half made.
#[test]
fn a_refused_value_is_said_in_the_panel_and_the_session_waits() {
    let Some(mut app) = headless_app() else { return };
    // The floating panel is what this holds, and it is off by default since the owner's trial.
    app.floating_panel = true;
    frame(&mut app, Vec::new());
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    type_text(&mut app, "-nothing");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    let session = app.session.as_ref().expect("still pending");
    assert!(session.panel.error.is_some(), "the refusal is said in the panel");
    assert!(app.doc.parameter_named("Radius").is_none(), "no variable is left behind");
}

// ------------------------------------------------------------------------------------- T7

/// **T7: a console line typed into the palette is the same line typed into the console**
/// (`interaction-spec.md` §5.5, §11). Space opens the palette with the keyboard in it, the line is
/// typed key by key, and Enter runs it as the console would; a second `App` runs the same line from
/// the console, and the two documents are the same, block for block.
#[test]
fn t7_a_console_line_from_the_palette_is_the_line_from_the_console() {
    let line = "sphere radius=10 'ball'";
    let Some(mut palette) = headless_app() else { return };
    frame(&mut palette, Vec::new());
    frame(&mut palette, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
    assert!(palette.palette.is_some(), "Space opens the palette");
    type_text(&mut palette, line);
    frame(&mut palette, vec![key_event(egui::Key::Enter)]);
    frame(&mut palette, Vec::new());
    assert!(palette.palette.is_none(), "it closes once the line has run");
    assert_eq!(palette.console_history.last().map(String::as_str), Some(line), "the line is in the console's history as typed");

    let mut console = headless_app().expect("an adapter, as the first had one");
    frame(&mut console, Vec::new());
    console.run_command(line);
    frame(&mut console, Vec::new());
    assert_eq!(named_form(&palette), named_form(&console));
    assert!(names::in_use(&palette.doc).iter().any(|(_, n)| n == "ball"));
}

/// **A search runs the command it finds, and Esc closes the palette having run nothing** (§5.4):
/// `cyl` and Enter make a cylinder pending as its menu row would; Esc in a fresh palette changes
/// nothing.
#[test]
fn the_palette_runs_what_it_finds_and_esc_runs_nothing() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    frame(&mut app, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
    type_text(&mut app, "cyl");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    frame(&mut app, Vec::new());
    let pending = app.session.as_ref().expect("the cylinder is pending, as from its menu row");
    assert!(matches!(app.part_graph().node(pending.block), Some(Node::Cylinder { .. })));
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    assert!(app.session.is_none());
    assert_eq!(app.recent.first().copied(), Some("create.cylinder"), "remembered as run lately");

    let nodes = app.part_graph().nodes().count();
    frame(&mut app, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
    type_text(&mut app, "sph");
    frame(&mut app, vec![key_event(egui::Key::Escape)]);
    frame(&mut app, Vec::new());
    assert!(app.palette.is_none(), "Esc closes it");
    assert_eq!(app.part_graph().nodes().count(), nodes, "and nothing ran");
}

/// **A click outside the palette puts it away and runs nothing**, as a click outside any popup
/// does; a click on it keeps it open. It used to stay until Esc.
#[test]
fn a_click_outside_the_palette_closes_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    frame(&mut app, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
    type_text(&mut app, "sph");
    assert!(app.palette.is_some());
    let nodes = app.part_graph().nodes().count();
    // Its own field, near the top of the screen's middle, keeps it.
    let field = egui::pos2(700.0, 900.0 * 0.18 + 12.0);
    left_click(&mut app, field);
    assert!(app.palette.is_some(), "a click on the palette keeps it");
    left_click(&mut app, egui::pos2(1300.0, 860.0));
    assert!(app.palette.is_none(), "a click outside closes it");
    assert_eq!(app.part_graph().nodes().count(), nodes, "and runs nothing");
}

// ------------------------------------------------------------------------------- the keys

fn chord_event(key: egui::Key, ctrl: bool, shift: bool, alt: bool) -> egui::Event {
    let modifiers = egui::Modifiers { alt, ctrl, shift, mac_cmd: false, command: ctrl };
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }
}

/// **A command is rebound by pressing the keys in the Keys panel, and a chord in use is refused
/// by name** (`interaction-spec.md` §9.2, T10). Duplicate moves from Ctrl+D to Ctrl+Alt+D and
/// answers to the new chord only; asking it to take Ctrl+S, which is Save's, is refused and changes
/// nothing; Esc gives up waiting.
#[test]
fn a_command_is_rebound_by_pressing_keys_and_a_taken_chord_is_refused() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere 'ball'");
    app.select_body(app.node_named("ball"));
    frame(&mut app, Vec::new());
    let nodes = |app: &App| app.part_graph().nodes().count();

    app.invoke_command(by("help.shortcuts"));
    app.begin_capture("edit.duplicate");
    frame(&mut app, vec![chord_event(egui::Key::D, true, false, true)]);
    assert!(app.capturing.is_none(), "the chord pressed was taken");
    assert_eq!(app.bindings_note.as_deref(), Some("Duplicate is on Ctrl+Alt+D"));

    let before = nodes(&app);
    frame(&mut app, vec![chord_event(egui::Key::D, true, false, false)]);
    assert_eq!(nodes(&app), before, "Ctrl+D no longer duplicates");
    app.select_body(app.node_named("ball"));
    frame(&mut app, vec![chord_event(egui::Key::D, true, false, true)]);
    assert_eq!(nodes(&app), before + 1, "Ctrl+Alt+D does");

    app.begin_capture("edit.duplicate");
    frame(&mut app, vec![chord_event(egui::Key::S, true, false, false)]);
    assert_eq!(app.bindings_note.as_deref(), Some("Ctrl+S is Save's — take it off there first"));
    assert!(app.keymap.lookup(&Shortcut::new("S", Modifiers::ctrl())).is_some_and(|b| b.op == "file.save"));

    app.begin_capture("edit.duplicate");
    frame(&mut app, vec![key_event(egui::Key::Escape)]);
    assert!(app.capturing.is_none() && app.bindings_note.is_none(), "Esc gives up");
}

/// **F frames and Alt+H shows every hidden body** (§9.1's table).
#[test]
fn f_frames_and_alt_h_shows_all() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere 'a'");
    app.run_command("cube 'b' centre=200,0,0");
    for name in ["a", "b"] {
        app.select_body(app.node_named(name));
        frame(&mut app, vec![key_event(egui::Key::H)]);
    }
    let hidden = |app: &App| app.part_graph().roots().into_iter().filter(|id| !app.is_visible(*id)).count();
    assert_eq!(hidden(&app), 2, "H hid both");
    frame(&mut app, vec![chord_event(egui::Key::H, false, false, true)]);
    assert_eq!(hidden(&app), 0, "Alt+H shows them all");
    let before = (app.camera.eye, app.camera.target);
    app.select_body(app.node_named("b"));
    frame(&mut app, vec![key_event(egui::Key::F)]);
    for _ in 0..30 {
        frame(&mut app, Vec::new());
    }
    assert!((app.camera.eye, app.camera.target) != before, "F frames the selection");
}

// -------------------------------------------------------------------- the marking menu, repeat

fn press(pos: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
    egui::Event::PointerButton { pos, button, pressed, modifiers: egui::Modifiers::NONE }
}

/// A point inside the viewport, found by hovering until the viewport says the pointer is over it.
fn in_viewport(app: &mut App) -> egui::Pos2 {
    for (x, y) in [(1000.0, 520.0), (900.0, 450.0), (1100.0, 600.0), (800.0, 400.0)] {
        let at = egui::pos2(x, y);
        frame(app, vec![egui::Event::PointerMoved(at)]);
        if app.viewport_hover.is_some() {
            return at;
        }
    }
    panic!("no point of the window is over the viewport");
}

/// A right-click there: press and release, as the window sends them.
fn right_click(app: &mut App, at: egui::Pos2) {
    frame(app, vec![press(at, egui::PointerButton::Secondary, true)]);
    frame(app, vec![press(at, egui::PointerButton::Secondary, false)]);
    frame(app, Vec::new());
}

fn left_click(app: &mut App, at: egui::Pos2) {
    frame(app, vec![egui::Event::PointerMoved(at)]);
    frame(app, vec![press(at, egui::PointerButton::Primary, true)]);
    frame(app, vec![press(at, egui::PointerButton::Primary, false)]);
    frame(app, Vec::new());
}

/// **A right-click with a body selected opens the body's marking menu, and a slot is picked by a
/// click or by a left-drag from the centre** (`interaction-spec.md` §7, T8's other half on a real
/// `App`). East on a body is Shell: a click there makes a pending shell. Then, from a fresh menu, a
/// left press at the centre dragged North-East and released makes an offset. With nothing
/// selected, the menu offers what can be made, and its South, More…, opens the palette narrowed to
/// what is on offer.
#[test]
fn a_right_click_opens_the_marking_menu_and_a_slot_runs() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere 'ball'");
    app.select_body(app.node_named("ball"));
    let at = in_viewport(&mut app);
    right_click(&mut app, at);
    let menu = app.marking.as_ref().expect("the right-click opened the marking menu");
    assert_eq!(menu.subject, marking::Subject::Body);
    let (dx, dy) = marking::slot_offset(2, 84.0);
    left_click(&mut app, at + egui::vec2(dx, dy));
    let pending = app.session.as_ref().expect("East on a body is Shell, now pending");
    assert!(matches!(app.part_graph().node(pending.block), Some(Node::Shell { .. })));
    frame(&mut app, vec![key_event(egui::Key::Enter)]);

    app.select_body(app.node_named("ball"));
    right_click(&mut app, at);
    assert!(app.marking.is_some());
    let (dx, dy) = marking::slot_offset(1, 60.0);
    frame(&mut app, vec![egui::Event::PointerMoved(at)]);
    frame(&mut app, vec![press(at, egui::PointerButton::Primary, true)]);
    for step in 1..=6 {
        let t = step as f32 / 6.0;
        frame(&mut app, vec![egui::Event::PointerMoved(at + egui::vec2(dx * t, dy * t))]);
    }
    frame(&mut app, vec![press(at + egui::vec2(dx, dy), egui::PointerButton::Primary, false)]);
    frame(&mut app, Vec::new());
    let pending = app.session.as_ref().expect("a drag North-East picked Offset");
    assert!(matches!(app.part_graph().node(pending.block), Some(Node::Offset { .. })));
    frame(&mut app, vec![key_event(egui::Key::Escape)]);

    app.select_body(None);
    right_click(&mut app, at);
    assert_eq!(app.marking.as_ref().map(|m| m.subject), Some(marking::Subject::Empty));
    let (dx, dy) = marking::slot_offset(marking::MORE, 84.0);
    left_click(&mut app, at + egui::vec2(dx, dy));
    let palette = app.palette.as_ref().expect("More… opens the palette");
    let only = palette.only.as_ref().expect("narrowed to what is on offer");
    assert!(only.contains(&"create.sphere") && !only.contains(&"edit.paste_here"), "Paste here is not on offer");

    frame(&mut app, vec![key_event(egui::Key::Escape)]);
    right_click(&mut app, at);
    frame(&mut app, vec![key_event(egui::Key::Escape)]);
    assert!(app.marking.is_none(), "Esc closes it, having run nothing");
}

/// **Shift+R runs the last command again, from the values it was last kept with, on the current
/// selection** (§8); and the marking menu's centre, which is empty, runs nothing: a click where a
/// stray right-click landed only closes the menu.
#[test]
fn shift_r_repeats_the_last_command_with_its_kept_values() {
    let Some(mut app) = headless_app() else { return };
    // The floating panel is what this holds, and it is off by default since the owner's trial.
    app.floating_panel = true;
    frame(&mut app, Vec::new());
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    type_text(&mut app, "12");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    let spheres = |app: &App| {
        app.part_graph().nodes().filter(|(_, n)| matches!(n, Node::Sphere { radius: ParamRef::Literal(r), .. } if *r == 12.0)).count()
    };
    assert_eq!(spheres(&app), 1);
    frame(&mut app, vec![chord_event(egui::Key::R, false, true, false)]);
    assert!(app.session.is_some(), "a repeated session command is pending, Esc-able as any is");
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    assert_eq!(spheres(&app), 2, "the second starts from the radius the first was kept with");

    // The centre of the marking menu closes it and repeats nothing, by a click or a right-click.
    app.select_body(None);
    let at = in_viewport(&mut app);
    let nodes = app.part_graph().nodes().count();
    right_click(&mut app, at);
    assert!(app.marking.is_some(), "the menu opened");
    left_click(&mut app, at);
    assert!(app.marking.is_none() && app.session.is_none(), "a click at the centre closes it");
    right_click(&mut app, at);
    right_click(&mut app, at);
    assert!(app.marking.is_none() && app.session.is_none(), "so does a second right-click");
    assert_eq!(app.part_graph().nodes().count(), nodes, "and nothing ran");
}

// ------------------------------------------------------------------- the shelf, the hint

/// **The shelf is empty until built, runs what it holds through the one door, and gives a command
/// up when asked** (`interaction-spec.md` §10.2): no toolbar at first; a sphere put on it shows as a
/// button, a click on which makes a pending sphere as its menu row would; taken off, it is gone.
#[test]
fn the_shelf_starts_empty_and_runs_what_is_put_on_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    assert!(app.shelf.is_empty() && app.shelf_drawn.is_empty(), "no default toolbar");
    app.shelf_add("create.sphere");
    app.shelf_add("create.sphere");
    assert_eq!(app.shelf, vec!["create.sphere"], "once each");
    frame(&mut app, Vec::new());
    let (_, rect) = app.shelf_drawn.first().copied().expect("the shelf draws its button");
    left_click(&mut app, rect.center());
    let pending = app.session.as_ref().expect("the shelf's sphere is pending");
    assert!(matches!(app.part_graph().node(pending.block), Some(Node::Sphere { .. })));
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    app.shelf_remove("create.sphere");
    frame(&mut app, Vec::new());
    assert!(app.shelf.is_empty() && app.shelf_drawn.is_empty());
}

/// **The first-run hint shows on an empty document until the palette has been opened three times**
/// (§10.3), and never on a document with something in it.
#[test]
fn the_first_run_hint_goes_after_three_palette_openings() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    assert!(app.first_run_hint_due(), "an empty document, the palette never opened");
    for _ in 0..2 {
        frame(&mut app, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
        frame(&mut app, vec![key_event(egui::Key::Escape)]);
    }
    assert!(app.first_run_hint_due(), "twice is not yet");
    app.run_command("sphere");
    assert!(!app.first_run_hint_due(), "not over a document with something in it");
    app.run_command("undo");
    assert!(app.first_run_hint_due());
    frame(&mut app, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
    frame(&mut app, vec![key_event(egui::Key::Escape)]);
    assert!(!app.first_run_hint_due(), "the third opening ends it for good");
    assert_eq!(app.palette_opens, 3);
}

// ------------------------------------------------------------------------------ rename

/// **`rename` names any row as its own field does, and warns what the field refuses** -- a block,
/// a curve and a parameter by what they are called; a name another block holds, a row nobody has
/// and a line with no new name are warned, so a script stops there -- and F2, with nothing named,
/// opens the selected row's name for typing.
#[test]
fn rename_names_any_row_and_warns_what_it_refuses() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in ["sphere 'ball'", "cube 'box'", "param name=wall value=2", "line p0=0,0,0 p1=0,0,10 'rail'"] {
        app.run_command(line);
    }
    for line in ["rename 'ball' to=orb", "rename 'rail' to=spine", "rename 'wall' to=thickness"] {
        let warned = app.console_warnings;
        app.run_command(line);
        assert_eq!(app.console_warnings, warned, "`{line}` was refused");
    }
    assert!(app.node_named("orb").is_some() && app.node_named("ball").is_none());
    assert!(app.curve_named("spine").is_some());
    assert!(app.doc.parameter_named("thickness").is_some());
    for line in ["rename 'orb' to=box", "rename 'nothing' to=x", "rename 'orb'"] {
        let warned = app.console_warnings;
        app.run_command(line);
        assert!(app.console_warnings > warned, "`{line}` was not refused");
    }
    assert!(app.node_named("orb").is_some(), "a refused rename leaves the name it had");
    select(&mut app, "orb");
    frame(&mut app, vec![key_event(egui::Key::F2)]);
    assert!(
        matches!(&app.renaming, Some((crate::blocks::list::RowKey::Node(_), text)) if text == "orb"),
        "F2 opens the selected row's name"
    );
}

/// **A rename in the GUI is echoed from what the row answered to before**: an unnamed block by the
/// label it showed, a cleared name as `to=""`, and the lines replay to the same names.
#[test]
fn a_gui_rename_echoes_from_the_label_or_name_the_row_had() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere");
    frame(&mut app, Vec::new());
    let sphere = app.node_named("Sphere").expect("an unnamed sphere, by its label");
    let written = app.echo_log.len();
    assert!(app.set_body_name(sphere, "ball"));
    frame(&mut app, Vec::new());
    assert!(app.set_body_name(sphere, ""));
    frame(&mut app, Vec::new());
    let lines = app.echo_log[written..].to_vec();
    assert_eq!(lines, vec!["rename 'Sphere' to=ball".to_string(), "rename 'ball' to=\"\"".to_string()]);
    let mut all = vec!["sphere".to_string()];
    all.extend(lines.iter().cloned());
    let replayed = replay(&all);
    assert_same_graph(&app, &replayed, &all);
}

/// **The tool panel opens clear of the block it is making** (`interaction-spec.md` §4.1): a sphere
/// invoked with the pointer over the middle of the view, where the sphere appears, gets its panel
/// beside the sphere rather than on it.
#[test]
fn the_tool_panel_opens_clear_of_the_block_it_is_making() {
    let Some(mut app) = headless_app() else { return };
    // The floating panel is what this holds, and it is off by default since the owner's trial.
    app.floating_panel = true;
    frame(&mut app, Vec::new());
    let middle = app.viewport_rect.expect("the viewport was drawn").center();
    frame(&mut app, vec![egui::Event::PointerMoved(middle)]);
    app.invoke_command(by("create.sphere"));
    frame(&mut app, Vec::new());
    frame(&mut app, Vec::new());
    let block = app.session.as_ref().expect("a pending sphere").block;
    let extent = app.screen_extent(block).expect("the sphere is on screen");
    let beside_pointer = egui::Rect::from_min_size(middle + egui::vec2(28.0, 28.0), egui::vec2(300.0, 100.0));
    assert!(extent.intersects(beside_pointer), "the test means something only if the pointer's place would cover the sphere");
    let window = app
        .egui_ctx
        .memory(|m| m.area_rect(egui::Id::new(("tool panel", "create.sphere"))))
        .expect("the panel was drawn");
    assert!(!window.intersects(extent), "the panel {window:?} covers the sphere {extent:?}");
}

/// **Ctrl makes a notebook number field's drag ten times coarser** (`interaction-spec.md` §4.4), as
/// it does the tool panel's; Shift's tenth is egui's own and is not applied twice.
#[test]
fn ctrl_makes_a_number_field_s_drag_coarse() {
    let ctx = egui::Context::default();
    for (ctrl, times) in [(false, 1.0), (true, 10.0)] {
        let input = egui::RawInput { modifiers: egui::Modifiers { ctrl, ..Default::default() }, ..Default::default() };
        let mut rate = 0.0;
        let _ = ctx.run_ui(input, |ui| rate = scrub_rate(ui, 500.0, 0.25));
        assert_eq!(rate, scrub_speed(500.0, 0.25) * times, "ctrl {ctrl}");
    }
}

/// **With the floating panel off -- the default since the owner's trial -- a pending block opens in
/// the list with its first number waiting**: no panel is drawn, the block's first field has the
/// keyboard, typing replaces its value, Tab moves on, and Enter keeps the block with what was
/// typed as one undo step.
#[test]
fn with_no_floating_panel_the_pending_block_waits_in_the_list() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    assert!(!app.floating_panel, "off by default");
    let depth = app.stack.history().0.len();
    app.invoke_command(by("create.sphere"));
    for _ in 0..4 {
        frame(&mut app, Vec::new());
    }
    assert!(app.session.as_ref().is_some_and(|s| !s.awaiting), "the block opened and took the keyboard");
    assert!(app.egui_ctx.memory(|m| m.focused()).is_some(), "its first number has the keyboard");
    assert!(
        app.egui_ctx.memory(|m| m.area_rect(egui::Id::new(("tool panel", "create.sphere")))).is_none(),
        "no floating panel"
    );
    // At: X, Y, Z, then Radius -- the order the block's rows read in.
    for (i, text) in ["5", "6", "7", "25"].into_iter().enumerate() {
        if i > 0 {
            frame(&mut app, vec![key_event(egui::Key::Tab)]);
        }
        type_text(&mut app, text);
    }
    frame(&mut app, vec![key_event(egui::Key::Enter)]);
    frame(&mut app, Vec::new());
    assert!(app.session.is_none(), "Enter in one of its numbers keeps the block");
    let sphere = app
        .part_graph()
        .nodes()
        .find_map(|(_, n)| match n {
            Node::Sphere { centre, radius } => Some((centre.clone(), *radius)),
            _ => None,
        })
        .expect("the sphere");
    assert_eq!(sphere.1, ParamRef::Literal(25.0), "the radius typed: {sphere:?}");
    assert_eq!(sphere.0, fieldform_scene::Place::at(5.0, 6.0, 7.0), "the centre typed: {sphere:?}");
    assert_eq!(app.stack.history().0.len(), depth + 1, "one undo step for the making and every number");
}

// --------------------------------------------------------- the drag gestures of the notebook

/// The operand slots of the block called `holder`.
///
/// Matched on the two kinds these tests build rather than walked generically: `Node`'s own slot
/// walker is `pub(crate)` to `fieldform_scene`, and a test that says which kinds it means is
/// clearer here than one that would need the walker opened up to say it.
fn slots_of(app: &App, holder: &str) -> Vec<fieldform_scene::Operand> {
    let id = app.node_named(holder).expect("the holder is in the graph");
    match app.part_graph().node(id).expect("the holder's node") {
        Node::Boolean { inputs, .. } | Node::Blend { inputs, .. } => inputs.clone(),
        Node::Shell { input, .. } => vec![*input],
        other => panic!("slots_of does not know {other:?}"),
    }
}

/// **A nested block dragged out onto the list becomes a plain block and empties its slot**
/// (`notebook-spec.md` §7.2, and item 112's "Done when": "dragging a nested block out leaves a
/// plain block and an empty slot in the error state").
///
/// `App::drop_on_the_list` is what a release over a gutter or the empty area calls, so this is the
/// gesture's own handler and not a copy of it. Two halves, and the section names both: the slot it
/// came from is empty, and the block is **plain** -- `Ownership::Free`, not `Variable`. R5 is the
/// reason the second half matters: only an explicit *Make variable* declares, and a drop is not
/// one. "To pull a block out *and* keep it wired, use *Make variable*" is the section's own
/// sentence, which is exactly the distinction a drop that declared would erase.
///
/// **What this does not reach**: the payload is built here, so it holds the handler to the right
/// edit without saying that the *row* hands it the right holder. That wiring is
/// `block_row(.., Some(id))` from a holder's own `inside` list, and seeing it would mean pressing
/// the grip's rectangle in a real frame -- the egui drag plumbing, which no test here drives yet.
#[test]
fn dragging_a_nested_block_out_onto_the_list_leaves_a_plain_block_and_an_empty_slot() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=10 centre=20,0,0 'ball'");
    app.run_command("cube size=40,40,40 centre=0,0,0 'box'");
    app.run_command("boolean -u in=ball,box 'joined'");
    frame(&mut app, Vec::new());

    let ball = app.node_named("ball").expect("the ball");
    let joined = app.node_named("joined").expect("the union");
    let key = fieldform_scene::BlockKey::Node(ball);
    assert_eq!(app.doc.ownership(key), fieldform_scene::Ownership::Nested, "nested to begin with");
    assert_eq!(
        slots_of(&app, "joined").iter().filter(|s| s.source() == Some(ball)).count(),
        1,
        "and the union reads it",
    );

    // The release itself, over the empty area under the rows (`onto: None`), carrying the row's
    // own holder as the grip put it there.
    let depth = app.stack.history().0.len();
    app.drop_on_the_list(DraggedBlock { id: ball, from: Some(joined) }, None);
    frame(&mut app, Vec::new());

    // Both halves read before either is asserted, so a run says which of the two the gesture got
    // wrong rather than stopping at the first and leaving the other unknown.
    let slots = slots_of(&app, "joined");
    let emptied = slots.iter().any(|s| s.is_empty());
    let ownership = app.doc.ownership(key);
    assert!(
        emptied && ownership == fieldform_scene::Ownership::Free,
        "§7.2 wants an empty slot and a plain block; the slot is {}and the ball is {ownership:?} \
         (slots {slots:?})",
        if emptied { "empty " } else { "still filled " },
    );
    assert_eq!(slots.len(), 2, "the row is still there to be filled, not closed up");
    assert!(app.part_graph().node(ball).is_some(), "and the block itself is still in the graph");
    assert_eq!(app.stack.history().0.len(), depth + 1, "one undo step for the whole gesture");

    // And it is undoable, since a gesture that cannot be taken back is not one.
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert_eq!(slots_of(&app, "joined")[0].source(), Some(ball), "the undo puts it back in its slot");
    assert_eq!(app.doc.ownership(key), fieldform_scene::Ownership::Nested, "nested again");
}

/// **A block the wiring declared is dragged out of nothing: its row is already its own**
/// (`notebook-spec.md` §7.2 as amended 2026-10-24, §17).
///
/// This asserted the shared instance until the owner drove it: a block two blocks read was nested
/// in both, and dragging it off one row left it nested in the other. A second reader declares it
/// now, so there is no second slot to be in and no row inside a reader to grab — the drag-out has
/// nothing to do, and does nothing, which is the state the grip's own tooltip describes.
#[test]
fn a_block_two_readers_declared_is_already_out_of_every_slot() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=10 centre=20,0,0 'ball'");
    app.run_command("cube size=40,40,40 centre=0,0,0 'box'");
    app.run_command("boolean -u in=ball,box 'joined'");
    app.run_command("cube size=10,10,10 centre=60,0,0 'far'");
    app.run_command("boolean -u in=ball,far 'second'");
    frame(&mut app, Vec::new());

    let ball = app.node_named("ball").expect("the ball");
    let second = app.node_named("second").expect("the second union");
    let key = fieldform_scene::BlockKey::Node(ball);
    assert_eq!(
        app.doc.ownership(key),
        fieldform_scene::Ownership::Variable,
        "two readers declared it",
    );
    assert_eq!(
        slots_of(&app, "joined").iter().filter(|s| s.source() == Some(ball)).count()
            + slots_of(&app, "second").iter().filter(|s| s.source() == Some(ball)).count(),
        2,
        "and both still read it, by name rather than by holding it",
    );

    // A drop on the list carries no holder for it, since its row is at the top level, so §7.2
    // finds no slot to empty and leaves everything as it is.
    app.drop_on_the_list(DraggedBlock { id: ball, from: None }, None);
    frame(&mut app, Vec::new());
    assert_eq!(app.doc.ownership(key), fieldform_scene::Ownership::Variable, "still declared");
    assert_eq!(
        slots_of(&app, "second").iter().filter(|s| s.source() == Some(ball)).count(),
        1,
        "and no slot was emptied",
    );
    let _ = second;
}

/// **A block dragged from one input to another moves: the slot it left empties** (§7.3 read with
/// §7.1 and §7.2 — plain moves, Ctrl shares). One gesture, so **one** undo step.
///
/// The drop arms the move and `replace_node` completes it in the same graph edit, which is what
/// this drives: `completing_a_move` set exactly as the release sets it, then the destination write
/// the panel row makes. **What it does not reach** is the release itself — that the payload
/// carries the row's holder, and that Ctrl is read off the frame the button came up in. Those live
/// in the egui drag plumbing, which no test here drives.
#[test]
fn a_block_dragged_from_one_input_to_another_leaves_the_slot_it_came_from_empty() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=10 centre=20,0,0 'ball'");
    app.run_command("cube size=40,40,40 centre=0,0,0 'box'");
    app.run_command("boolean -u in=ball,box 'joined'");
    app.run_command("shell target=none thickness=2 'hollow'");
    frame(&mut app, Vec::new());

    let ball = app.node_named("ball").expect("the ball");
    let joined = app.node_named("joined").expect("the union");
    let hollow = app.node_named("hollow").expect("the shell");
    let depth = app.stack.history().0.len();

    // The release: the ball, out of the union's slot, onto the shell's empty one, no Ctrl.
    app.completing_a_move = Some((joined, ball));
    let mut node = app.part_graph().node(hollow).cloned().expect("the shell's node");
    match &mut node {
        Node::Shell { input, .. } => *input = ball.into(),
        other => panic!("a shell, not {other:?}"),
    }
    app.replace_node(hollow, node);
    frame(&mut app, Vec::new());

    assert_eq!(slots_of(&app, "hollow")[0].source(), Some(ball), "the slot it was dropped on");
    assert!(
        slots_of(&app, "joined").iter().all(|s| s.source() != Some(ball)),
        "and the slot it came from let go: {:?}",
        slots_of(&app, "joined"),
    );
    assert_eq!(
        app.doc.ownership(fieldform_scene::BlockKey::Node(ball)),
        fieldform_scene::Ownership::Nested,
        "it is nested in its new reader, not left at the top level",
    );
    assert_eq!(app.stack.history().0.len(), depth + 1, "one undo step for the move");
    assert!(app.completing_a_move.is_none(), "and the pending half is spent");

    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert_eq!(
        slots_of(&app, "joined").iter().filter(|s| s.source() == Some(ball)).count(),
        1,
        "one undo puts both slots back",
    );
    assert!(slots_of(&app, "hollow")[0].is_empty(), "including the one it was dropped on");
}

/// **Ctrl+drag shares instead of moving** (§7.3): the block ends up read by both, which since the
/// owner's ruling of 2026-10-24 (§17) makes it a **variable** — one block, a row of its own, and a
/// chip in each reader. The gesture is unchanged; what it produces is.
///
/// Ctrl is the *absence* of an armed move, which is what makes this the control for the test
/// above: the same destination write, with nothing armed, must leave the first slot alone.
#[test]
fn a_block_shared_onto_a_second_input_stays_in_the_first() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=10 centre=20,0,0 'ball'");
    app.run_command("cube size=40,40,40 centre=0,0,0 'box'");
    app.run_command("boolean -u in=ball,box 'joined'");
    app.run_command("shell target=none thickness=2 'hollow'");
    frame(&mut app, Vec::new());

    let ball = app.node_named("ball").expect("the ball");
    let hollow = app.node_named("hollow").expect("the shell");

    // The same release with Ctrl held: nothing is armed, so nothing lets go.
    let mut node = app.part_graph().node(hollow).cloned().expect("the shell's node");
    match &mut node {
        Node::Shell { input, .. } => *input = ball.into(),
        other => panic!("a shell, not {other:?}"),
    }
    app.replace_node(hollow, node);
    frame(&mut app, Vec::new());

    assert_eq!(slots_of(&app, "hollow")[0].source(), Some(ball), "the shell reads it");
    assert_eq!(
        slots_of(&app, "joined").iter().filter(|s| s.source() == Some(ball)).count(),
        1,
        "and so does the union it was already in",
    );
    assert_eq!(
        app.doc.ownership(fieldform_scene::BlockKey::Node(ball)),
        fieldform_scene::Ownership::Variable,
        "so the wiring declared it: one block, a row of its own, referenced by both (§17)",
    );
}

/// **Moved to a sibling slot of the same block, only the slot it left empties.**
///
/// The case that decides the gesture needs an index at all: "every slot that names it"
/// ([`fieldform_scene::Node::stop_reading`], which §7.2's drag-out wants) would empty the slot the
/// drop had just filled, and the move would read as a delete. The destination is the slot that
/// changed, so the one to empty is the one that did not.
#[test]
fn a_block_moved_to_a_sibling_slot_empties_only_the_slot_it_left() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=10 centre=20,0,0 'ball'");
    app.run_command("cube size=40,40,40 centre=0,0,0 'box'");
    app.run_command("boolean -s in=ball,box 'cut'");
    frame(&mut app, Vec::new());

    let ball = app.node_named("ball").expect("the ball");
    let cut = app.node_named("cut").expect("the subtraction");
    assert_eq!(slots_of(&app, "cut")[0].source(), Some(ball), "the ball is the first operand");

    // Dragged from the first slot onto the second, within the one block.
    app.completing_a_move = Some((cut, ball));
    let mut node = app.part_graph().node(cut).cloned().expect("the node");
    match &mut node {
        Node::Boolean { inputs, .. } | Node::Blend { inputs, .. } => inputs[1] = ball.into(),
        other => panic!("a boolean, not {other:?}"),
    }
    app.replace_node(cut, node);
    frame(&mut app, Vec::new());

    let slots = slots_of(&app, "cut");
    assert_eq!(slots[1].source(), Some(ball), "it is in the slot it was dropped on: {slots:?}");
    assert!(slots[0].is_empty(), "and out of the one it left: {slots:?}");
    assert_eq!(slots.len(), 2, "both rows are still there");
}

/// **A block put back into the slot it came from recovers it**, which is the second half of §7.2's
/// gesture: the hole is filled, the block is nested again, and the whole return is one undo step.
///
/// The hole is made the way the only built control makes one -- the operand row's *Clear*, whose
/// own hover text calls it "the same state dragging the block out of the slot leaves", typed here
/// as `target=none` because that is the console's half of the same act. Filling it goes through
/// `block_edit::node_with_keys`, which is what a drop on the field writes through as well as what a
/// typed name does: one reader for the gesture and the line, so this holds the state the drop
/// lands in rather than a second implementation of it.
#[test]
fn a_block_put_back_into_an_empty_slot_fills_it_in_one_undo_step() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=10 centre=20,0,0 'ball'");
    app.run_command("cube size=40,40,40 centre=0,0,0 'box'");
    app.run_command("boolean -u in=ball,box 'joined'");
    app.run_command("shell target=none thickness=2 'hollow'");
    frame(&mut app, Vec::new());
    assert!(slots_of(&app, "hollow")[0].is_empty(), "the shell is waiting to be told what to read");

    let joined = app.node_named("joined").expect("the union");
    let depth = app.stack.history().0.len();
    app.run_command("set hollow.target=joined");
    frame(&mut app, Vec::new());

    assert_eq!(slots_of(&app, "hollow")[0].source(), Some(joined), "the slot reads the union now");
    assert_eq!(
        app.doc.ownership(fieldform_scene::BlockKey::Node(joined)),
        fieldform_scene::Ownership::Nested,
        "and the block it was given is nested in it",
    );
    assert_eq!(app.stack.history().0.len(), depth + 1, "one undo step for the whole return");

    // And back out again, because a recovery that cannot be undone is not one.
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert!(slots_of(&app, "hollow")[0].is_empty(), "the undo leaves the hole as it was");
}

/// **The active cage follows the cage being worked on**: the selected cage block's, a visible one
/// when the active one is hidden, and the one left when the active one is deleted -- so Tab and
/// the cage tools reach the cage on screen with two in the part (the owner, 2026-10-24).
#[test]
fn the_active_cage_follows_the_selected_visible_or_remaining_cage_block() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    let first = app.active_cage;
    app.run_command("cage");
    frame(&mut app, Vec::new());
    let second = app.active_cage;
    assert_ne!(first, second, "a second cage is a new one");
    let node_of = |app: &App, cage: CageId| app.part_graph().nodes().find(|(_, n)| matches!(n, Node::Cage(c) if *c == cage)).map(|(id, _)| id).unwrap();

    // Selecting the first cage's block makes it active.
    let first_node = node_of(&app, first);
    app.select_body(Some(first_node));
    assert_eq!(app.active_cage, first);
    // Hiding it moves the tools to the cage still on screen.
    app.toggle_visibility(first_node);
    frame(&mut app, Vec::new());
    assert_eq!(app.active_cage, second, "the visible cage is the one to work on");
    // Deleting the second leaves the first, hidden or not, as the only cage to work on.
    let second_node = node_of(&app, second);
    app.select_body(Some(second_node));
    assert!(app.delete_body(second_node));
    frame(&mut app, Vec::new());
    let left: Vec<CageId> = app.part_graph().nodes().filter_map(|(_, n)| if let Node::Cage(c) = n { Some(*c) } else { None }).collect();
    assert_eq!(app.active_cage, first, "cages left in the part: {left:?}; first {first:?}, second {second:?}");
    // Shown again, Tab's reading of the part finds its block.
    app.toggle_visibility(first_node);
    frame(&mut app, Vec::new());
    assert!(app.is_visible(first_node));
    assert_eq!(app.active_cage, first);
}

/// **A Cut Through line across the whole view cuts the cube in one closed loop**, resolved to edge
/// points and committed as one `knife` line; a line that spans nothing is refused by name.
#[test]
fn a_cut_through_line_across_the_view_cuts_the_cube_in_one_loop() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    app.invoke_command(by("cage.cut"));
    frame(&mut app, Vec::new());
    assert!(app.cage_tool.as_ref().is_some_and(|t| t.command.op == "cage.cut"), "the knife is armed");
    let rect = app.viewport_rect.expect("a frame sets the viewport");
    let aspect = rect.width() / rect.height().max(1.0);
    let y = rect.center().y;
    let (a, b) = (egui::pos2(rect.min.x + 10.0, y), egui::pos2(rect.max.x - 10.0, y));
    let paths = app.through_paths(aspect, rect, a, b).expect("a line across the view cuts the cube");
    assert_eq!(paths.len(), 1, "one loop round the cube: {paths:?}");
    assert_eq!(paths[0].first(), paths[0].last(), "closed");
    assert!(paths[0].len() >= 5, "at least four edges crossed: {}", paths[0].len());
    // A short line off in a corner spans nothing.
    let corner = egui::pos2(rect.min.x + 5.0, rect.min.y + 5.0);
    assert!(app.through_paths(aspect, rect, corner, corner + egui::vec2(20.0, 0.0)).is_err());

    // Placed as the tool places it, read against the view each frame, and cut on Enter.
    if let Some(tool) = app.cage_tool.as_mut() {
        tool.knife.through = Some((a, Some(b)));
    }
    frame(&mut app, Vec::new());
    assert_eq!(app.cage_tool.as_ref().map(|t| t.knife.through_paths.len()), Some(1));
    if let Some((path, Err((why, faces)))) = app.cage_tool.as_ref().and_then(|t| t.knife.through_paths.first()) {
        panic!("the loop's plan was refused: {why} at {faces:?}; path {path:?}");
    }
    let history = app.console_history.len();
    let faces_before = app.cage().face_count();
    frame(&mut app, vec![key(egui::Key::Enter)]);
    // A typed line is its own record, in the console's history rather than the echo.
    let lines: Vec<&String> = app.console_history[history..].iter().filter(|l| l.starts_with("knife points=")).collect();
    let said = app.console_log.len();
    assert_eq!(lines.len(), 1, "one knife line: {lines:?}; the console has {said} lines");
    assert!(app.cage().face_count() > faces_before, "the loop split every face it crossed");
    assert!(!app.doc.cage_is_red(app.active_cage), "and left the cage closed");
    assert!(app.cage_tool.as_ref().is_some_and(|t| t.knife.through.is_none()), "the line is spent; the knife stays armed");
}


/// **Crease and Uncrease from the menu act on the selected edges and run as the lines that do
/// it**, so the console shows what happened and a script can say it again; the edges stay picked
/// for the next step, and with nothing picked the menu says so and does nothing.
#[test]
fn crease_and_uncrease_from_the_menu_run_as_lines_on_the_selected_edges() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    app.invoke_command(by("cage.crease"));
    frame(&mut app, Vec::new());
    assert!(!app.cage().has_creases(), "nothing picked, nothing creased");

    let (a, b) = app.cage().edges().next().expect("a cube has edges");
    app.selection.toggle(Selection::Edge(a, b));
    app.invoke_command(by("cage.crease"));
    frame(&mut app, Vec::new());
    assert!(app.cage().is_creased(a, b));
    let line = format!("crease edges={}-{}", a.index(), b.index());
    assert_eq!(app.console_history.last(), Some(&line), "run as its line");
    assert!(app.selection.parts().contains(&Selection::Edge(a, b)), "the edge stays picked");

    app.invoke_command(by("cage.uncrease"));
    frame(&mut app, Vec::new());
    assert!(!app.cage().is_creased(a, b));
    assert_eq!(app.console_history.last(), Some(&format!("uncrease edges={}-{}", a.index(), b.index())));
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert!(app.cage().is_creased(a, b), "undo puts the crease back");
}

/// **The knife's panel: Pick type, Offset mode and Offset read and write the active point, Angle
/// increment and Cut Through faces validate their words** (`cage-tools-instructions.md` §1). A
/// typed offset moves the active point along its own edge from the end the tick marks, in mm or
/// as a fraction, and a drag along the same edge does the same by the pointer.
#[test]
fn the_knife_s_panel_reads_and_writes_the_active_point_s_offset() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    app.invoke_command(by("cage.cut"));
    frame(&mut app, Vec::new());
    assert!(app.cage_tool.as_ref().is_some_and(|t| t.command.op == "cage.cut"), "the knife is armed");

    // Nothing placed yet: the panel's defaults, and no offset to read.
    let rows = app.knife_panel_rows();
    assert_eq!(rows.iter().map(|r| r.key).collect::<Vec<_>>(), vec!["pick_type", "offset_mode", "offset", "angle_increment", "cut_through_faces"]);
    assert_eq!(rows[0].text, "both");
    assert_eq!(rows[1].text, "mm");
    assert_eq!(rows[2].text, "\u{2014}", "no active point yet");
    assert_eq!(rows[3].text, "45");
    assert_eq!(rows[4].text, "all");

    // A word row refuses a word that is not one of its choices, and keeps the setting.
    assert!(app.knife_panel_set("pick_type", "sideways").is_err());
    app.knife_panel_set("pick_type", "edges").expect("a pick type");
    assert_eq!(app.knife_panel_rows()[0].text, "edges");

    // An edge point placed directly on the path (as the Cut Through test places its line),
    // active at the end nearer where it would have been clicked.
    let mesh = app.cage().clone();
    let (a, b) = mesh.edges().next().expect("a cube has edges");
    let v0 = mesh.vertex_ids().find(|&v| v != a && v != b).expect("a third vertex");
    let len = (mesh.vertex_position(a) - mesh.vertex_position(b)).norm();
    if let Some(tool) = app.cage_tool.as_mut() {
        tool.knife.path = vec![fieldform_cage::edit::knife::CutPoint::Vertex(v0), fieldform_cage::edit::knife::CutPoint::Edge { edge: (a, b), t: 0.5 }];
        tool.knife.active_ref_end = Some(a);
    }
    let offset = app.knife_panel_rows()[2].text.clone();
    assert!((offset.parse::<f64>().unwrap() - len * 0.5).abs() < 1e-6, "half the edge in mm: {offset}");

    // Fraction mode reads the same point as 0.5, not the millimetre length.
    app.knife_panel_set("offset_mode", "fraction").expect("a mode");
    assert_eq!(app.knife_panel_rows()[2].text, "0.5");

    // A typed offset moves the point along its edge, from the end the tick marks -- here `a`,
    // so a fraction of 0.25 is a quarter of the way from `a` to `b`.
    app.knife_panel_set("offset", "0.25").expect("a number");
    let t = app.cage_tool.as_ref().and_then(|t| t.knife.path.last().copied());
    assert_eq!(t, Some(fieldform_cage::edit::knife::CutPoint::Edge { edge: (a, b), t: 0.25 }));

    // Back to mm: a distance past the edge's own length is clamped to it rather than refused.
    app.knife_panel_set("offset_mode", "mm").expect("a mode");
    app.knife_panel_set("offset", &(len * 10.0).to_string()).expect("a number");
    let t = app.cage_tool.as_ref().and_then(|t| t.knife.path.last().copied());
    assert_eq!(t, Some(fieldform_cage::edit::knife::CutPoint::Edge { edge: (a, b), t: 1.0 }), "clamped to the far end");

    // Angle increment and Cut Through faces: a bad word or number is refused and nothing changes.
    assert!(app.knife_panel_set("angle_increment", "0").is_err(), "an increment of nought is refused");
    assert!(app.knife_panel_set("cut_through_faces", "diagonal").is_err());
    app.knife_panel_set("cut_through_faces", "front-facing").expect("a choice");
    assert_eq!(app.knife_panel_rows()[4].text, "front-facing");

    // A drag along the active point's own edge, on screen, slides it without reaching for the
    // camera: `knife_drag_active` is what the viewport's drag handler calls once the press
    // landed on the point (`knife_active_point_near`).
    let rect = app.viewport_rect.expect("a frame sets the viewport");
    let aspect = rect.width() / rect.height().max(1.0);
    let placement = app.control_placement();
    let project = |v: fieldform_cage::VertexId| app.camera.project_to_screen(aspect, rect, vertex_pos(&mesh, &placement, v));
    if let (Some(sa), Some(sb)) = (project(a), project(b)) {
        // The clamp above left the active point at `t=1.0`, on `b`: a pointer a few pixels off it
        // is still near it.
        let near_b = sb + (sa - sb) * 0.02;
        assert!(app.knife_active_point_near(aspect, rect, near_b), "the active point, at b, is near a pointer beside it");
        let three_quarters = sa + (sb - sa) * 0.75;
        app.knife_drag_active(aspect, rect, three_quarters);
        let t = app.cage_tool.as_ref().and_then(|t| t.knife.path.last().copied());
        let Some(fieldform_cage::edit::knife::CutPoint::Edge { edge: dragged_edge, t: dragged_t }) = t else {
            panic!("the active point is still an edge point: {t:?}");
        };
        assert_eq!(dragged_edge, (a, b));
        assert!((dragged_t - 0.75).abs() < 1e-4, "dragged to three quarters along a-b: {dragged_t}");
    }
}

/// **An extrude's panel offers `vertex_normals` as a direction**: each cap vertex goes along its own
/// normal, so the panel's cap lands where the primitive's own extrude puts it, the distance row
/// reads the same number back, and the kept line says the words and the distance.
#[test]
fn an_extrude_in_vertex_normals_lands_where_the_primitive_puts_it() {
    use fieldform_cage::FaceId;
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    let cage = app.active_cage;
    let before = app.doc.cage_body(cage).unwrap().mesh().clone();
    // Two faces meeting at a cube edge: the vertices on that edge have a diagonal normal.
    let faces = [FaceId::from_index(0), FaceId::from_index(2)];
    for f in faces {
        app.selection.toggle(Selection::Face(f));
    }
    app.invoke_command(by("cage.extrude"));
    frame(&mut app, Vec::new());
    assert!(app.cage_session.is_some());
    app.cage_panel_set("direction", "vertex_normals").expect("a direction");
    app.cage_panel_set("distance", "10").expect("a number");
    frame(&mut app, Vec::new());
    assert_eq!(app.cage_panel_rows()[0].text, "10", "the row reads the distance back along each vertex's own normal");

    let want = fieldform_cage::edit::extrude::extrude(
        &before,
        &faces,
        fieldform_cage::edit::extrude::Mode::Region,
        fieldform_cage::edit::extrude::Direction::VertexNormals,
        10.0,
    )
    .expect("the primitive extrudes it");
    let session = app.cage_session.as_ref().unwrap();
    let mesh = app.cage();
    let straight = session.cap.iter().all(|&v| mesh.vertex_position(v).coords.iter().all(|c| c.is_finite()));
    assert!(straight);
    let mut got: Vec<[i64; 3]> = session.cap.iter().map(|&v| key_of(mesh.vertex_position(v))).collect();
    let mut expected: Vec<[i64; 3]> = want.new_vertices.iter().map(|&v| key_of(want.mesh.vertex_position(v))).collect();
    got.sort();
    expected.sort();
    assert_eq!(got, expected, "the cap stands where the primitive's extrude puts it");

    let echoed = app.echo_log.len();
    frame(&mut app, vec![key(egui::Key::Enter)]);
    assert_eq!(&app.echo_log[echoed..], &["cageextrude faces=0,2 distance=10 direction=vertex_normals".to_string()]);

    fn key_of(p: nalgebra::Point3<f64>) -> [i64; 3] {
        [(p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64, (p.z * 1e6).round() as i64]
    }
}

/// **Insert Edge, Bridge and Weld have a settings window too**: Count and a typed offset (from the
/// end nearer the pointer, millimetres or a fraction) for Insert Edge, Divisions for Bridge,
/// Position for Weld. What they hold reaches the line the click commits (`count=`, `divisions=`,
/// `at=`), and several rings come out evenly spaced as one undo step.
#[test]
fn the_click_tools_settings_reach_the_lines_they_commit() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    let mesh = app.cage().clone();
    let (a, b) = mesh.edges().next().expect("a cube has edges");

    // Insert Edge: the rows, a refused count, and a count that reaches the line.
    app.invoke_command(by("cage.insert_ring"));
    frame(&mut app, Vec::new());
    let keys: Vec<&str> = app.cage_tool_panel_rows().iter().map(|r| r.key).collect();
    assert_eq!(keys, vec!["count", "offset_mode", "offset"]);
    assert!(app.cage_tool_panel_set("count", "0").is_err() && app.cage_tool_panel_set("count", "65").is_err());
    app.cage_tool_panel_set("count", "3").expect("a count");
    assert!(app.cage_tool_row_disabled("offset"), "an offset means nothing to several evenly spaced rings");
    app.hover = Selection::Edge(a, b);
    let line = app.ring_line().expect("an edge is hovered");
    assert!(line.starts_with(&format!("insertring edge={}-{} t=", a.index(), b.index())) && line.ends_with(" count=3"), "{line}");
    let faces = app.cage().face_count();
    app.run_command(&line);
    assert_eq!(app.cage().face_count(), faces + 12, "three rings of four faces each, as one step");
    app.run_command("undo");
    assert_eq!(app.cage().face_count(), faces, "one undo step");

    // A typed offset overrides the pointer: a quarter in from the end nearer the pointer.
    app.cage_tool_panel_set("count", "1").expect("a count");
    app.cage_tool_panel_set("offset_mode", "fraction").expect("a mode");
    assert!(app.cage_tool_panel_set("offset", "far").is_err());
    app.cage_tool_panel_set("offset", "0.25").expect("an offset");
    let rect = app.viewport_rect.expect("a frame sets the viewport");
    let aspect = rect.width() / rect.height().max(1.0);
    let placement = app.control_placement();
    let project = |v: fieldform_cage::VertexId| app.camera.project_to_screen(aspect, rect, vertex_pos(&mesh, &placement, v));
    let (sa, sb) = (project(a).unwrap(), project(b).unwrap());
    app.hover = Selection::Edge(a, b);
    app.cage_tool_hover(aspect, rect, Some(sa + (sb - sa) * 0.3), false, false, false);
    let tool = app.cage_tool.as_ref().unwrap();
    assert_eq!(tool.ref_end, Some(a), "the end nearer the pointer");
    assert!((tool.t - 0.25).abs() < 1e-9, "a quarter from a: {}", tool.t);
    app.hover = Selection::Edge(a, b);
    app.cage_tool_hover(aspect, rect, Some(sa + (sb - sa) * 0.7), false, false, false);
    let tool = app.cage_tool.as_ref().unwrap();
    assert_eq!(tool.ref_end, Some(b));
    assert!((tool.t - 0.75).abs() < 1e-9, "a quarter from b is three quarters from a: {}", tool.t);
    app.cage_tool_panel_set("offset", "off").expect("cleared");
    assert_eq!(app.cage_tool_panel_rows()[2].text, "\u{2014}");

    // A press on the hovered edge is the tool's, a drag along it slides t, and the release
    // commits where it was left; the pointer need not stay on the edge.
    app.hover = Selection::Edge(a, b);
    assert!(app.cage_tool_press(aspect, rect, sa + (sb - sa) * 0.5));
    app.cage_tool_drag(aspect, rect, sa);
    app.hover = Selection::None;
    app.cage_tool_hover(aspect, rect, Some(sa + (sb - sa) * 0.8 + egui::vec2(0.0, 40.0)), false, false, false);
    let t = app.cage_tool.as_ref().unwrap().t;
    assert!(t > 0.7 && t < 0.9, "t follows the pointer along the pressed edge: {t}");
    app.cage_tool_release();
    assert_eq!(app.cage().face_count(), faces + 4, "the release inserted one ring");
    app.run_command("undo");

    // Bridge: Divisions, whole and in range, reaching the line.
    app.invoke_command(by("cage.bridge"));
    frame(&mut app, Vec::new());
    assert_eq!(app.cage_tool_panel_rows().iter().map(|r| r.key).collect::<Vec<_>>(), vec!["divisions"]);
    assert!(app.cage_tool_panel_set("divisions", "0").is_err() && app.cage_tool_panel_set("divisions", "999").is_err());
    app.cage_tool_panel_set("divisions", "2").expect("divisions");
    assert_eq!(app.cage_tool.as_ref().unwrap().divisions, 2);

    // Weld: Position.
    app.invoke_command(by("cage.weld"));
    frame(&mut app, Vec::new());
    assert_eq!(app.cage_tool_panel_rows()[0].text, "target");
    assert!(app.cage_tool_panel_set("position", "beside").is_err());
    app.cage_tool_panel_set("position", "midpoint").expect("a position");
    assert!(app.cage_tool.as_ref().unwrap().weld_midpoint);
    assert_eq!(app.cage_tool_panel_rows().iter().map(|r| r.key).collect::<Vec<_>>(), vec!["position", "boundary"]);
    assert!(app.cage_tool_panel_set("boundary", "everywhere").is_err());
    app.cage_tool_panel_set("boundary", "zip").expect("a boundary mode");
    assert!(app.cage_tool.as_ref().unwrap().weld_zip);
}

/// **The weld's boundary mode: `weld zip=A-B` zips the whole open rim through A onto the one
/// through B**, from one pair, as one undo step -- two rims of a tube closing into a torus.
#[test]
fn a_zip_weld_closes_a_tube_in_one_step() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    // Two rings round the girth of the plain cube make three layers between its ends.
    let start = app.cage().clone();
    let (a, b) = start
        .edges()
        .find(|&(a, b)| (start.vertex_position(a).xy() - start.vertex_position(b).xy()).norm() < 1e-9)
        .expect("a cube has vertical edges");
    app.run_command(&format!("insertring edge={}-{} count=2", a.index(), b.index()));
    frame(&mut app, Vec::new());
    let mesh = app.cage().clone();
    let (lo, hi) = mesh.vertex_ids().fold((f64::MAX, f64::MIN), |(lo, hi), v| {
        let z = mesh.vertex_position(v).z;
        (lo.min(z), hi.max(z))
    });
    let at = |v: fieldform_cage::VertexId, z: f64| (mesh.vertex_position(v).z - z).abs() < 1e-9;
    let ends: Vec<String> = mesh
        .face_ids()
        .filter(|&f| mesh.face_vertices(f).iter().all(|&v| at(v, hi)) || mesh.face_vertices(f).iter().all(|&v| at(v, lo)))
        .map(|f| f.index().to_string())
        .collect();
    assert_eq!(ends.len(), 2, "the cube's own top and bottom, after the rings: {} faces", mesh.face_count());
    assert!(mesh.face_count() == 14, "six faces and two rings of four");
    app.run_command(&format!("deletefaces faces={}", ends.join(",")));
    frame(&mut app, Vec::new());
    let tube = app.cage().clone();
    assert!(tube.is_open(), "a tube");
    let top = tube.vertex_ids().find(|&v| at(v, hi) && tube.vertex_position(v).x > 0.1 && tube.vertex_position(v).y > 0.1).expect("a corner of the top rim");
    let p = tube.vertex_position(top);
    let bottom = tube.vertex_ids().find(|&v| at(v, lo) && (tube.vertex_position(v).xy() - p.xy()).norm() < 1e-9).expect("the vertex below it");
    let faces = tube.face_count();
    app.run_command(&format!("weld zip={}-{}", top.index(), bottom.index()));
    frame(&mut app, Vec::new());
    assert!(!app.cage().is_open(), "both rims zipped: the tube is a closed torus");
    assert_eq!(app.cage().face_count(), faces);
    assert_eq!(app.cage().euler_characteristic(), 0, "a torus");
    app.run_command("undo");
    assert!(app.cage().is_open(), "one undo step puts the tube back");
}

/// **Space with an extrude pending keeps it and extrudes again from the new cap** (the owner,
/// 2026-11-12), whether the cap was placed by a number set or by one typed straight into the panel;
/// each lift is its own line and its own undo step, and with nothing pending Space is the palette.
#[test]
fn space_with_an_extrude_pending_keeps_it_and_extrudes_again() {
    use fieldform_cage::FaceId;
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    let cage = app.active_cage;
    let before = app.doc.cage_body(cage).unwrap().mesh().clone();
    let space = || vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())];

    app.selection.toggle(Selection::Face(FaceId::from_index(1)));
    app.invoke_command(by("cage.extrude"));
    frame(&mut app, Vec::new());
    app.cage_panel_set("distance", "10").expect("a number");
    frame(&mut app, Vec::new());
    let echoed = app.echo_log.len();
    frame(&mut app, space());
    frame(&mut app, Vec::new());
    assert!(app.palette.is_none(), "Space did not open the palette");
    assert!(app.cage_session.is_some(), "the next lift is pending");
    assert_eq!(&app.echo_log[echoed..], &["cageextrude faces=1 distance=10".to_string()], "the first was kept");

    // Typed straight into the panel, then Space: the number is taken before the cap is kept.
    let echoed = app.echo_log.len();
    type_text(&mut app, "5");
    frame(&mut app, space());
    frame(&mut app, Vec::new());
    assert!(app.palette.is_none());
    assert!(app.cage_session.is_some(), "and the one after that");
    assert_eq!(&app.echo_log[echoed..], &["cageextrude faces=1 distance=5".to_string()], "the second, at the number typed");

    // The third never moved: Esc puts it back, and two extrusions of one face are eight walls.
    frame(&mut app, vec![key(egui::Key::Escape)]);
    assert!(app.cage_session.is_none());
    assert_eq!(app.cage().face_count(), before.face_count() + 8);
    assert!(!app.doc.cage_is_red(cage), "a solid");
    app.run_command("undo");
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert_eq!(app.doc.cage_body(cage).unwrap().geometry_fingerprint(), before.geometry_fingerprint(), "one undo step each");

    app.selection.clear();
    frame(&mut app, space());
    assert!(app.palette.is_some(), "with nothing pending Space is the palette");
}

/// **An extrude's panel: a typed distance places the cap, a typed direction re-places it, and the
/// kept line says the numbers rather than positions** (R8). An inset's thickness re-runs the inset.
#[test]
fn a_cage_session_s_panel_places_the_cap_and_the_kept_line_says_the_distance() {
    use fieldform_cage::FaceId;
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    let cage = app.active_cage;
    let before = app.doc.cage_body(cage).unwrap().mesh().clone();
    let face = FaceId::from_index(1);
    let normal = fieldform_cage::validate::face_normal(&before, face).normalize();

    app.selection.toggle(Selection::Face(face));
    app.invoke_command(by("cage.extrude"));
    frame(&mut app, Vec::new());
    assert!(app.cage_session.is_some(), "E on a face opens a session");
    let rows: Vec<&str> = app.cage_panel_rows().iter().map(|r| r.key).collect::<Vec<_>>().iter().map(|s| *s).collect();
    assert_eq!(rows, vec!["distance", "direction", "mode"]);
    assert_eq!(app.cage_panel_rows()[0].text, "0", "lifted at nought");

    app.cage_panel_set("distance", "10").expect("a number");
    frame(&mut app, Vec::new());
    assert_eq!(app.cage_panel_rows()[0].text, "10", "the row reads the cap");
    let session = app.cage_session.as_ref().unwrap();
    let mesh = app.cage();
    for (&v, b) in session.cap.iter().zip(&session.cap_before) {
        assert!((mesh.vertex_position(v) - (b + normal * 10.0)).norm() < 1e-9, "the cap stands 10 off along the normal");
    }
    app.cage_panel_set("direction", "sideways").expect_err("not a direction");
    app.cage_panel_set("direction", "z").expect("an axis");
    frame(&mut app, Vec::new());
    let session = app.cage_session.as_ref().unwrap();
    let mesh = app.cage();
    for (&v, b) in session.cap.iter().zip(&session.cap_before) {
        assert!((mesh.vertex_position(v) - (b + nalgebra::Vector3::z() * 10.0)).norm() < 1e-9, "the same distance, along z");
    }

    let echoed = app.echo_log.len();
    frame(&mut app, vec![key(egui::Key::Enter)]);
    assert!(app.cage_session.is_none(), "Enter keeps");
    let lines = &app.echo_log[echoed..];
    assert_eq!(lines, &["cageextrude faces=1 distance=10 direction=z".to_string()], "the line says the numbers, not positions");
    assert!(!app.doc.cage_is_red(cage), "kept at a height, the cage is a solid");
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert_eq!(app.doc.cage_body(cage).unwrap().geometry_fingerprint(), before.geometry_fingerprint(), "one undo step");

    // An inset: thickness typed re-runs it, and the kept line carries thickness and depth.
    app.selection.clear();
    app.selection.toggle(Selection::Face(face));
    app.invoke_command(by("cage.inset"));
    frame(&mut app, Vec::new());
    assert_eq!(app.cage_panel_rows().iter().map(|r| r.key).collect::<Vec<_>>(), vec!["thickness", "depth", "mode"]);
    app.cage_panel_set("thickness", "2.5").expect("a number");
    app.cage_panel_set("depth", "-1").expect("a number");
    frame(&mut app, Vec::new());
    let echoed = app.echo_log.len();
    frame(&mut app, vec![key(egui::Key::Enter)]);
    let lines = &app.echo_log[echoed..];
    assert_eq!(lines, &["cageinset faces=1 thickness=2.5 depth=-1".to_string()]);
}

/// **The graph view is a view: the menu and the console reach it by the same command, it moves
/// nothing in the document, and the keys that frame the viewport frame the graph instead while it
/// is shown** (`notebook-graph-view-spec.md` G2, G3).
#[test]
fn switching_the_blocks_panel_to_the_graph_changes_no_document_state() {
    let Some(mut app) = headless_app() else { return };
    let by = |id: &str| commands::by_id(id).unwrap_or_else(|| panic!("{id} is in the registry"));
    frame(&mut app, Vec::new());
    app.run_command("sphere r=10");
    frame(&mut app, Vec::new());
    let moves = app.document_moves;
    let json = serde_json::to_string(&app.doc.parts.get(&app.active_part).unwrap().graph).unwrap();

    app.invoke_command(by("view.panel.graph"));
    assert!(app.doc.graph_layout.shown);
    app.run_command("view show=notebook");
    assert!(!app.doc.graph_layout.shown);
    app.run_command("view show=graph map=off");
    assert!(app.doc.graph_layout.shown);
    frame(&mut app, Vec::new());
    app.run_command("frameall");
    app.run_command("relayout");
    frame(&mut app, vec![key(egui::Key::F)]);

    assert_eq!(app.document_moves, moves, "a view is not an edit");
    assert_eq!(
        serde_json::to_string(&app.doc.parts.get(&app.active_part).unwrap().graph).unwrap(),
        json,
        "the generated graph is byte-identical"
    );
}

/// **A named number is a node of the graph, wired to every block whose field reads it**, so the
/// number can be seen and changed in the graph as it can in the notebook (`param name=.. value=..`).
#[test]
fn a_named_number_is_a_node_wired_to_the_blocks_that_read_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("param name=reach value=11");
    app.run_command("sphere radius=reach 'ball'");
    app.run_command("cube size=reach 'box'");
    frame(&mut app, Vec::new());
    let list = app.list_readings();
    let sources = crate::graph_view::sources(
        &app.doc,
        app.part_graph(),
        &list.nesting,
        &list.listed,
        &|id| app.body_name(id).unwrap_or("?").to_string(),
        &|_| None,
        &|_| crate::graph_view::CurveRow::Own,
    );
    let model = crate::graph_view::model::build(&sources);
    let reach = model.nodes.iter().find(|n| n.label == "reach").expect("the number is a node");
    assert!(reach.variable && reach.number.is_some_and(|n| n.value == 11.0 && n.editable));
    let readers: Vec<_> = model.wires.iter().filter(|w| w.from == reach.id).map(|w| w.to).collect();
    assert_eq!(readers.len(), 2, "both the ball and the box read it");

    // A set from the graph is the notebook's own command, and the number moves.
    app.run_command("param name=reach value=12");
    assert_eq!(app.doc.parameter_value(app.doc.parameter_named("reach").unwrap()), Some(12.0));
}

#[test]
fn a_block_with_an_empty_input_is_flagged_in_the_graph_and_a_plain_one_is_not() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere radius=5 'ball'");
    app.run_command("shell target=none thickness=2 'hollow'");
    frame(&mut app, Vec::new());
    let list = app.list_readings();
    let sources = crate::graph_view::sources(
        &app.doc,
        app.part_graph(),
        &list.nesting,
        &list.listed,
        &|id| app.body_name(id).unwrap_or("?").to_string(),
        &|_| None,
        &|_| crate::graph_view::CurveRow::Own,
    );
    let model = crate::graph_view::model::build(&sources);
    let status = |name: &str| model.nodes.iter().find(|n| n.label == name).map(|n| n.status.clone());
    assert_eq!(status("ball"), Some(crate::graph_view::model::Status::Ok));
    assert!(matches!(status("hollow"), Some(crate::graph_view::model::Status::Error(_))), "{:?}", status("hollow"));
}

#[test]
fn a_closed_loop_of_curves_is_one_node_wired_to_the_extrude_that_reads_it() {
    use crate::graph_view::{model::Kind, CurveRow};
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=40,0,0 'a'",
        "line p0=40,0,0 p1=40,20,0 'b'",
        "line p0=40,20,0 p1=0,20,0 'c'",
        "line p0=0,20,0 p1=0,0,0 'd'",
        "extrude profile=a height=10 'slab'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    app.refresh_profiles();
    let list = app.list_readings();
    let nesting = &list.nesting;
    let curve_row = |c| match app.folded_curve_row(c, nesting) {
        Some(crate::blocks::list::Block::Profile(key)) => CurveRow::Profile(app.profiles.members(key).map_or(0, <[_]>::len)),
        None => app.profiles.key_of(c).map_or(CurveRow::Own, CurveRow::Inside),
        Some(_) => CurveRow::Own,
    };
    let sources = crate::graph_view::sources(
        &app.doc,
        app.part_graph(),
        &list.nesting,
        &list.listed,
        &|id| app.body_name(id).unwrap_or("?").to_string(),
        &|_| None,
        &curve_row,
    );
    let model = crate::graph_view::model::build(&sources);
    let loops: Vec<_> = model.nodes.iter().filter(|n| matches!(n.kind, Kind::Profile(_))).collect();
    assert_eq!(loops.len(), 1, "four curves, one loop, one node");
    assert!(!model.nodes.iter().any(|n| matches!(n.kind, Kind::Curve(_))), "no member has a node of its own");
    let slab = model.nodes.iter().find(|n| n.label == "slab").unwrap();
    assert!(slab.rows.iter().any(|r| r.from == loops[0].id), "the extrude's row is fed by the loop");
}

#[test]
fn find_lights_the_nodes_named_for_it_and_changes_no_document_state() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere radius=5 'Ball'");
    app.run_command("cube size=5 'box'");
    app.run_command("view show=graph");
    let before = app.doc.clone();
    app.run_command("find 'ball'");
    let (model, _) = app.graph_model();
    let hits = crate::graph_view::matching(&model, "BALL");
    assert_eq!(hits.len(), 1, "case is ignored and only the ball matches");
    assert!(crate::graph_view::matching(&model, "  ").is_empty(), "no text, no matches");
    app.run_command("find 'ball'");
    assert_eq!(app.doc.graph_layout, before.graph_layout);
    assert_eq!(app.doc.created, before.created, "finding is not an edit");
}

/// **A through hole has handles, at the point it goes in.** It runs on for ever and has no box,
/// which used to leave it with none while a blind hole beside it could be dragged; the handles
/// stand on the block's own place before they ask for a box, so both are moved the same way.
#[test]
fn a_through_hole_is_moved_from_where_it_goes_in() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cube size=40,40,20");
    app.run_command("hole at=5,6,10 dir=0,0,-1 depth=through");
    frame(&mut app, Vec::new());
    let hole = app
        .part_graph()
        .nodes()
        .find(|(_, n)| matches!(n, Node::Hole { depth: fieldform_scene::HoleDepth::Through, .. }))
        .map(|(id, _)| id)
        .expect("a through hole was made");
    assert!(app.body_extent(hole).as_box().is_none(), "unbounded: this is the case without a box");
    app.select_body(Some(hole));
    let gizmo = app.gizmo_frame().expect("a through hole has handles");
    assert!((gizmo.pivot - Point3::new(5.0, 6.0, 10.0)).norm() < 1e-9, "{:?}", gizmo.pivot);
}

/// **Standing a block on a surface keeps its length and faces the way its kind does**: a hole
/// goes in, a cylinder stands out, both from the point picked and as typed points, one undo step
/// each; and a block a placement moves is refused, since the click is not in its coordinates.
#[test]
fn a_block_stood_on_a_surface_keeps_its_length_and_faces_the_way_its_kind_does() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cylinder from=0,0,0 to=0,0,30 radius=4 'post'");
    app.run_command("hole at=5,6,10 dir=0,0,-1 depth=12");
    frame(&mut app, Vec::new());
    let find = |app: &App, hole: bool| {
        app.part_graph()
            .nodes()
            .find(|(_, n)| matches!(n, Node::Hole { .. }) == hole && matches!(n, Node::Hole { .. } | Node::Cylinder { .. }))
            .map(|(id, _)| id)
            .expect("made")
    };
    let ends_of = |app: &App, id| {
        let node = app.part_graph().node(id).cloned().expect("there");
        app.resolved_ends(node.ends()[0])
    };
    let (cylinder, hole) = (find(&app, false), find(&app, true));
    let at = Point3::new(40.0, -3.0, 2.0);
    let into = Vector3::new(-1.0, 0.0, 0.0);

    app.stand_ends_at(cylinder, at, into).expect("a cylinder stands");
    frame(&mut app, Vec::new());
    let (a, b) = ends_of(&app, cylinder);
    assert!((a - at).norm() < 1e-9 && (b - Point3::new(70.0, -3.0, 2.0)).norm() < 1e-9, "out, 30 long: {a} {b}");

    let (h0, h1) = ends_of(&app, hole);
    let was = (h1 - h0).norm();
    app.stand_ends_at(hole, at, into).expect("a hole stands");
    frame(&mut app, Vec::new());
    let (a, b) = ends_of(&app, hole);
    assert!((a - at).norm() < 1e-9, "it goes in where picked");
    assert!(((b - a) - into * was).norm() < 1e-9, "into the body, as long as it was: {a} {b}");
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert!((ends_of(&app, hole).0 - h0).norm() < 1e-9, "one undo step puts it back");

    app.run_command("move target=post by=0,0,5 wrap=true");
    frame(&mut app, Vec::new());
    assert!(app.stand_ends_at(cylinder, at, into).is_err(), "a placed block is refused");
}

/// **A rounded boolean takes a list from the console as a hard one does**: three bodies and a
/// radius make one rounded subtract of three, `set` takes the radius off and puts it back without
/// losing a body, and the echo writes every operand.
#[test]
fn a_rounded_boolean_of_three_is_made_and_kept_from_the_console() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cube size=40 'block'");
    app.run_command("sphere centre=12,0,20 radius=6 'one'");
    app.run_command("sphere centre=-12,0,20 radius=6 'two'");
    app.run_command("boolean op=subtract in=block,one,two radius=2 'cut'");
    frame(&mut app, Vec::new());
    let cut = app.node_named("cut").expect("made");
    let inputs = |app: &App| match app.part_graph().node(cut).cloned() {
        Some(Node::Blend { inputs, .. }) => (true, inputs.len()),
        Some(Node::Boolean { inputs, .. }) => (false, inputs.len()),
        other => panic!("a boolean, not {other:?}"),
    };
    assert_eq!(inputs(&app), (true, 3), "rounded, all three");
    assert_eq!(app.part_graph().roots(), vec![cut]);
    app.run_command("set cut.radius=0");
    frame(&mut app, Vec::new());
    assert_eq!(inputs(&app), (false, 3), "sharp, all three");
    app.run_command("set cut.radius=1.5");
    frame(&mut app, Vec::new());
    assert_eq!(inputs(&app), (true, 3), "rounded again, all three");
    let lines = crate::echo::create_lines(&app.doc, app.part_graph(), cut).expect("echoes");
    assert!(
        lines.iter().any(|l| l.starts_with("boolean") && l.contains("in=block,one,two") && l.contains("radius=1.5")),
        "the echo names every operand: {lines:?}"
    );
}

/// **A draft the section has no room for is said where it is typed, and a typed one is refused.**
/// A slot 4 mm wide has 2 mm for its walls to move in, so over 20 mm with mitred corners 5° fits
/// and 6.5° makes the outline cross itself. Typed, `set` refuses it by name and the draft stays at
/// 5°: `replace_node` gates only an edit that rewires the graph, so this used to say "Set slot"
/// and leave the next line to inherit the complaint. Through the panel's path the number goes in,
/// as a dragged number must, and the compile's reason is kept against the block, which is what
/// the extrude panel reads to say it under the draft.
#[test]
fn a_draft_past_the_section_s_room_is_refused_when_typed_and_said_beside_the_draft_in_the_panel() {
    let Some(mut app) = headless_app() else { return };
    for line in [
        "line p0=-20,-2,0 p1=20,-2,0 's1'",
        "line p0=20,-2,0 p1=20,2,0 's2'",
        "line p0=20,2,0 p1=-20,2,0 's3'",
        "line p0=-20,2,0 p1=-20,-2,0 's4'",
        "extrude profile=s1 height=20 draft=-5 corners=sharp 'slot'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    let slot = app.node_named("slot").expect("the slot");
    let draft = |app: &App| match app.part_graph().node(slot) {
        Some(Node::Extrude { draft, .. }) => app.doc.resolve(draft),
        _ => None,
    };
    assert_eq!(draft(&app), Some(-5.0));
    assert_eq!(app.compile_error(), None, "5\u{b0} fits");

    let said = app.console_log.len();
    app.run_command("set slot.draft=-6.5");
    let refused: Vec<&str> =
        app.console_log[said..].iter().filter(|l| !l.ok).map(|l| l.text.as_str()).collect();
    assert!(
        refused.iter().any(|l| l.contains("cross itself") && l.contains("2.00 mm") && l.contains("5.7\u{b0}")),
        "the console names the room and the draft that fits: {refused:?}"
    );
    assert_eq!(draft(&app), Some(-5.0), "and the typed draft is refused, not written");

    // The panel's path: the number goes in, and the reason is kept against this block.
    let mut node = app.part_graph().node(slot).cloned().expect("the slot");
    if let Node::Extrude { draft, .. } = &mut node {
        *draft = ParamRef::Literal(-6.5);
    }
    app.replace_node(slot, node);
    frame(&mut app, Vec::new());
    assert!(app.compile_error().is_some_and(|why| why.contains("cross itself")), "{:?}", app.compile_error());
    assert_eq!(app.draft_refused, Some(slot), "the panel says it beside this block's draft");
}

/// The lattice block an infill named `name` reads, and the block itself.
fn infill_named(app: &App, name: &str) -> (NodeId, fieldform_scene::ParamRef, fieldform_scene::Fillet, Node) {
    let id = app.node_named(name).unwrap_or_else(|| panic!("nothing is called {name}"));
    let graph = app.part_graph();
    let Some(Node::Infill { wall, fill, join, .. }) = graph.node(id) else {
        panic!("{name} is not an infill")
    };
    let fill = graph.node(fill.source().expect("the slot is filled")).expect("a block").clone();
    (id, *wall, *join, fill)
}

/// **The first click, on the case the numbers were chosen against.** A sixty millimetre cube with
/// nothing typed, off the origin so that where the cells start says something.
#[test]
fn a_first_click_fills_a_sixty_millimetre_cube_as_the_benchmark_does() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60 centre=5,0,0");
    app.run_command("infill target=box 'part'");
    frame(&mut app, Vec::new());
    let (id, wall, join, fill) = infill_named(&app, "part");
    assert_eq!(wall, ParamRef::Literal(2.0));
    assert!(join == Fillet::hard());
    let Node::Lattice { cell, frame: where_, map, phase } = fill else { panic!("{fill:?}") };
    assert_eq!(cell, fieldform_scene::LatticeCell::Gyroid);
    assert_eq!(map, CellMap::Rectangular { cell: ParamRef::Literal(10.0) });
    assert_eq!(phase, LatticePhase::sheet(ParamRef::Literal(1.0)));
    let fieldform_scene::Frame::At { origin, .. } = where_ else { panic!("{where_:?}") };
    assert_eq!(origin, [5.0, 0.0, 0.0].map(ParamRef::Literal), "at the body's centre, as a number");
    assert_eq!(app.part_graph().roots(), vec![id], "the body and the lattice are both consumed");
}

/// `cell=` alone writes that cell's own start, and a phase number typed beats it.
#[test]
fn a_cell_named_starts_from_its_preset_and_a_number_typed_wins() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60");
    app.run_command("infill target=box wall=1.5 cell=octet cell_size=8 join=0.4 'beams'");
    frame(&mut app, Vec::new());
    let (_, wall, join, fill) = infill_named(&app, "beams");
    assert_eq!(wall, ParamRef::Literal(1.5));
    assert_eq!(join.radius, ParamRef::Literal(0.4));
    let Node::Lattice { cell, phase, .. } = fill else { panic!("{fill:?}") };
    assert_eq!(cell, fieldform_scene::LatticeCell::Octet);
    assert_eq!(phase, LatticePhase::Strut { radius: ParamRef::Literal(0.4) }, "5 % of the cell");

    app.run_command("cube 'other' size=60,60,60 centre=100,0,0");
    app.run_command("infill target=other cell=gyroid cell_size=10 thickness=1.25 'typed'");
    frame(&mut app, Vec::new());
    let (_, _, _, fill) = infill_named(&app, "typed");
    let Node::Lattice { phase, .. } = fill else { panic!("{fill:?}") };
    assert_eq!(phase, LatticePhase::sheet(ParamRef::Literal(1.25)));
}

/// `fill=` reads a block that is already there, and makes nothing.
#[test]
fn a_fill_named_reads_the_block_and_makes_no_lattice() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60");
    app.run_command("tpms 'shared' cell=gyroid cell_size=10");
    let shared = app.node_named("shared").unwrap();
    let before = app.part_graph().nodes().count();
    app.run_command("infill target=box wall=2 fill=shared 'part'");
    frame(&mut app, Vec::new());
    assert_eq!(app.part_graph().nodes().count(), before + 1, "the infill and nothing else");
    let id = app.node_named("part").unwrap();
    let Some(Node::Infill { fill, .. }) = app.part_graph().node(id) else { panic!() };
    assert_eq!(fill.source(), Some(shared));
}

/// Refused by name, and nothing left behind, which is the half a refusal can forget.
#[test]
fn a_line_the_block_cannot_mean_makes_nothing() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60");
    app.run_command("tpms 'shared' cell=gyroid cell_size=10");
    let before = app.part_graph().nodes().count();
    for line in [
        "infill target=box wall=-1",
        "infill target=box fill=shared cell=gyroid",
        "infill target=box cell=nothing_of_the_kind",
        "infill target=box phase=strut cell=gyroid",
    ] {
        let warned = app.console_warnings;
        app.run_command(line);
        assert!(app.console_warnings > warned, "{line} was not refused");
        assert_eq!(app.part_graph().nodes().count(), before, "{line} left something behind");
    }
}

/// **A graded lattice cannot be aimed**, and a frame moved to the body's centre counts, so a ramp
/// keeps the world's frame rather than turning the line into a shader that will not generate.
#[test]
fn a_graded_infill_off_the_origin_still_draws() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60 centre=40,0,0");
    app.run_command("plane 'floor' origin=0,0,-30");
    app.run_command("ramp 'grade' input=floor in=0,60 out=0.6,2.0");
    app.run_command("infill target=box wall=2 cell=gyroid cell_size=10 thickness=grade 'part'");
    frame(&mut app, Vec::new());
    assert!(app.node_named("part").is_some(), "the line was taken");
    let graph = app.part_graph().clone();
    fieldform_gpu::generate(&app.doc, &graph).expect("a shader for it");
}

/// One undo step, the lattice with it, and the body back as a body.
#[test]
fn undoing_an_infill_takes_its_lattice_and_gives_the_body_back() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60");
    let body = app.node_named("box").unwrap();
    // No name on the line: a name given there is an edit of its own, with a step of its own.
    app.run_command("infill target=box");
    frame(&mut app, Vec::new());
    assert_eq!(app.part_graph().nodes().count(), 3);
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert_eq!(app.part_graph().nodes().count(), 1, "one step");
    assert_eq!(app.part_graph().roots(), vec![body]);
}

/// The palette's "infill" finds the tool that does it, and only that.
#[test]
fn infill_is_the_tool_called_infill() {
    let tool = commands::by_id("modify.infill").expect("in the registry");
    assert_eq!(tool.display_name, "Infill");
    assert!(tool.aliases.contains(&"lightweight"));
    let tpms = commands::by_id("create.tpms").expect("in the registry");
    assert!(!tpms.aliases.contains(&"infill"), "two tools answering to one word");
}

/// **The panel, drawn**, in every state it has a branch for: it has to survive a wall of nought, a
/// rounded join, a fill that is not a lattice and an empty slot, and the report has to come back.
#[test]
fn the_panel_draws_in_every_state_and_its_report_arrives() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("cube 'box' size=60,60,60");
    app.run_command("infill target=box 'part'");
    let id = app.node_named("part").unwrap();
    // The block's own body, since a row draws it only once it has been opened.
    let panel = |app: &mut App, id: NodeId| {
        let ctx = app.egui_ctx.clone();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| app.node_block_body(ui, id));
    };
    panel(&mut app, id);
    assert!(app.infill_cavities.contains_key(&id), "the panel was not drawn");

    // The report is asked for once the document has sat still, and arrives off the thread.
    let started = std::time::Instant::now();
    let report = loop {
        panel(&mut app, id);
        if let Some(report) = app.infill_cavities.get(&id).and_then(|run| run.report()) {
            break report;
        }
        assert!(started.elapsed().as_secs() < 30, "no report in thirty seconds");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    assert!((report.deepest - 30.0).abs() < 0.01, "read {}", report.deepest);
    assert!(report.has_room(2.0) && report.fill_lands);
    assert!((0.05..0.6).contains(&report.solid_share), "read {}", report.solid_share);

    for line in [
        "set part.wall=0",
        "set part.wall=40",
        "set part.wall=2 part.join=0.5",
    ] {
        app.run_command(line);
        panel(&mut app, id);
    }
    app.run_command("sphere 'ball' radius=5");
    app.run_command("cube 'other' size=60,60,60 centre=100,0,0");
    app.run_command("infill target=other fill=ball 'plain'");
    let plain = app.node_named("plain").unwrap();
    panel(&mut app, plain);
    app.run_command("infill target=none fill=none 'empty'");
    let empty = app.node_named("empty").expect("an empty block is a block");
    panel(&mut app, empty);
    // Against a body that only bounds its distance, which marks the wall as a minimum.
    app.run_command("sphere 'round' radius=30 centre=300,0,0");
    app.run_command("cube 'corner' size=40,40,40 centre=320,0,0");
    app.run_command("morph a=round b=corner t=0.5 'soft'");
    app.run_command("infill target=soft wall=2 cell=gyroid 'bounded'");
    let bounded = app.node_named("bounded").unwrap();
    panel(&mut app, bounded);
}

/// Runs frames until the list has been to the row and let go, and says whether it arrived.
fn reveal_arrives(app: &mut App) -> bool {
    let mut reached = false;
    for _ in 0..200 {
        frame(app, Vec::new());
        reached |= app.reveal.as_ref().is_some_and(|r| r.reached);
    }
    assert!(app.reveal.is_none(), "the list lets go once it is there");
    reached
}

/// **A plane two blocks down is found from the viewport.** Every row starts closed, so the row
/// holding the selection does not exist until the rows above it have opened: that it is drawn at
/// all is what this asserts, and the row is what scrolls and shows the highlight.
#[test]
fn a_block_selected_in_the_viewport_opens_every_row_above_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "sphere 'ball' radius=10",
        "plane origin=0,0,2 'deep'",
        "boolean in=ball,deep -i 'inner'",
        "cube size=40,40,40 'box'",
        "boolean in=box,inner -s 'outer'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    let deep = app.node_named("deep").expect("the plane");
    // As the plane's outline under a click does it.
    app.select_body(Some(deep));
    assert!(reveal_arrives(&mut app), "the plane's row was never drawn");
}

/// **And a curve folded into a profile under the block reading it**, which has two kinds of row
/// above it, the loop's and the extrude's.
#[test]
fn a_curve_selected_in_the_viewport_opens_its_profile_and_its_reader() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=40,0,0 'a'",
        "line p0=40,0,0 p1=40,20,0 'b'",
        "line p0=40,20,0 p1=0,20,0 'c'",
        "line p0=0,20,0 p1=0,0,0 'd'",
        "extrude profile=a height=10 'slab'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    let curve = app.curve_named("c").expect("the curve");
    app.select_curve(curve);
    assert!(reveal_arrives(&mut app), "the curve's row was never drawn");
}

/// **A row clicked in the list stays where it is.** It is under the pointer already, and
/// scrolling it to the middle would move it away from the hand that chose it. The control for the
/// two above: the same selection, from the list.
#[test]
fn a_row_chosen_in_the_list_is_not_scrolled_to() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere 'ball' radius=10");
    app.run_command("cube size=40,40,40 'box'");
    frame(&mut app, Vec::new());
    let ball = app.node_named("ball").expect("the sphere");
    app.select_body(Some(ball));
    app.chose_from_list(RowKey::Node(ball));
    frame(&mut app, Vec::new());
    assert!(app.reveal.is_none());
}

/// **A selection made while the graph is up is found in the notebook when you go there**, which is
/// also what *Show in notebook* needs: each view follows on its own, so changing view is not
/// taken for having already seen it.
#[test]
fn a_selection_made_in_the_graph_opens_the_notebook_down_to_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "sphere 'ball' radius=10",
        "plane origin=0,0,2 'deep'",
        "boolean in=ball,deep -i 'inner'",
        "cube size=40,40,40 'box'",
        "boolean in=box,inner -s 'outer'",
        "view show=graph",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    let deep = app.node_named("deep").expect("the plane");
    app.select_body(Some(deep));
    frame(&mut app, Vec::new());
    assert_eq!(app.graph_followed, Some(deep), "the graph has followed it");
    assert!(app.reveal.is_none(), "the notebook is not drawn, so it has not moved");
    app.run_command("view show=notebook");
    assert!(reveal_arrives(&mut app), "the plane's row was never drawn");
}

/// **K is the knife on the cage with nothing selected, and Split off it** (the owner, 2026-10-28).
/// The knife starts from no selection -- its first point is a click -- so it used to wait for
/// something to be picked before K meant it; a key that does nothing until you have pointed at
/// something is a key nobody finds.
#[test]
fn k_is_the_knife_on_the_cage_whether_or_not_anything_is_selected() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    assert_eq!(app.control, ControlMesh::Cage);
    assert!(app.selection.is_empty());
    assert_eq!(app.shortcut_op("modify.split".into()), "cage.cut", "nothing selected");
    app.selection.toggle(Selection::Face(fieldform_cage::FaceId::from_index(1)));
    assert_eq!(app.shortcut_op("modify.split".into()), "cage.cut", "a face selected");
    app.control = ControlMesh::Surface(fieldform_scene::QuadMeshId::default());
    assert_eq!(app.shortcut_op("modify.split".into()), "modify.split", "a surface keeps Split");
}

/// **Shift lands a dragged selection on the grid and the snap needs the grid to be drawn**: the
/// centroid goes to an intersection in the drawing plane's two axes and is left alone in the third,
/// and a perspective view -- where there is no grid -- moves it exactly where the cursor put it.
#[test]
fn shift_lands_a_dragged_selection_on_the_drawn_grid() {
    use fieldform_cage::FaceId;
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("cage");
    frame(&mut app, Vec::new());
    // Through the plane's own setter with the swing, since the grid is drawn only square to its
    // plane (the owner, 2026-10-31): a plane set by assignment leaves the view where it was.
    app.use_drawing_plane(DrawingPlane::World { normal: Axis::Z, offset_mm: 0.0 }, Swing::Yes);
    // The swing is animated over frames; a tick past its whole duration lands it.
    app.camera.tick_snap(60.0);
    app.show_grid = true;
    app.camera.projection = Projection::Orthographic;
    let (aspect, rect) = (1400.0 / 900.0, egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0)));
    let pitch = app.drawing_grid(aspect, rect).expect("an orthographic view with a plane has a grid").pitch;

    app.selection.clear();
    app.selection.toggle(Selection::Face(FaceId::from_index(1)));
    let centroid = |app: &App| {
        let placement = app.control_placement();
        app.selection.centroid(app.control(), &placement).unwrap().cast::<f64>()
    };
    let from = rect.center();
    let to = from + egui::vec2(37.0, 21.0);

    app.begin_move(aspect, rect, from);
    app.continue_move(aspect, rect, to, false, false);
    let free = centroid(&app);
    app.end_move();
    app.run_command("undo");

    app.begin_move(aspect, rect, from);
    app.continue_move(aspect, rect, to, true, false);
    let snapped = centroid(&app);
    app.end_move();
    let on_grid = |v: f64| ((v / pitch).round() * pitch - v).abs() < 1e-3 * pitch.max(1.0);
    assert!(on_grid(snapped.x) && on_grid(snapped.y), "{snapped:?} sits on a {pitch} mm grid");
    assert!(!(on_grid(free.x) && on_grid(free.y)), "the unsnapped drag was not already there: {free:?}");

    app.run_command("undo");
    app.camera.projection = Projection::Perspective;
    assert!(app.drawing_grid(aspect, rect).is_none(), "no grid is drawn in perspective");
    let centroid_after = |app: &mut App, grid: bool| {
        app.begin_move(aspect, rect, from);
        app.continue_move(aspect, rect, to, grid, false);
        let at = centroid(app);
        app.end_move();
        app.run_command("undo");
        at
    };
    let with_shift = centroid_after(&mut app, true);
    let without = centroid_after(&mut app, false);
    assert!((with_shift - without).norm() < 1e-3, "perspective: Shift moves nothing but the cursor");
}

/// **A panel shows a number as a person reads it and keeps every digit underneath** (the owner,
/// 2026-10-28, after a knife's offset read `18.69785451225658 mm`): four decimals, no trailing
/// zeros, a value too small for four decimals in scientific form rather than as a nought, and
/// words, lists and counts left exactly as they are.
#[test]
fn a_panel_row_shows_a_number_rounded_and_keeps_the_exact_one() {
    use crate::app::tool_panel::{rounded_for_display, Row};
    let row = |text: &str, unit| Row { key: "offset", text: text.to_string(), unit };
    assert_eq!(rounded_for_display(18.69785451225658), "18.6979");
    assert_eq!(rounded_for_display(2.5), "2.5");
    assert_eq!(rounded_for_display(10.0), "10");
    assert_eq!(rounded_for_display(-0.00004), "-4.000e-5", "too small to show is not nought");
    assert_eq!(rounded_for_display(0.0), "0");
    let long = row("18.69785451225658", commands::Unit::Mm);
    assert_eq!(long.shown(), "18.6979");
    assert_eq!(long.text, "18.69785451225658", "the row keeps every digit");
    assert_eq!(long.number(), Some(18.69785451225658), "and a scrub starts from them");
    assert_eq!(row("1,2,3", commands::Unit::Mm).shown(), "1,2,3", "a list is not one number");
    assert_eq!(row("zip", commands::Unit::Word).shown(), "zip");
    assert_eq!(row("\u{2014}", commands::Unit::Mm).shown(), "\u{2014}", "an empty row stays a dash");
}

/// T5: an unknown key is refused with what the verb reads, as data, and a near miss is named.
#[test]
fn an_unknown_key_is_refused_with_what_the_verb_reads() {
    let Some(mut app) = headless_app() else { return };
    let before = app.part_graph().nodes().count();
    let out = app.run_line("cube sise=20");
    assert!(!out.ok);
    let err = out.error.expect("a refusal carries its structure");
    assert_eq!(err.code, "unknown_key");
    assert_eq!(err.key.as_deref(), Some("sise"));
    assert_eq!(err.did_you_mean.as_deref(), Some("size"));
    for key in ["size", "centre"] {
        assert!(err.accepted.iter().any(|k| k == key), "{key} missing from {:?}", err.accepted);
    }
    assert_eq!(app.part_graph().nodes().count(), before, "a refused line makes nothing");
}

#[test]
fn a_line_that_works_says_what_the_console_said() {
    let Some(mut app) = headless_app() else { return };
    let out = app.run_line("sphere radius=10 'ball'");
    assert!(out.ok, "{out:?}");
    assert!(out.error.is_none() && out.warnings.is_empty());
    assert!(out.said.is_some_and(|s| s.contains("sphere")));
    let out = app.run_line("notaverb");
    assert_eq!(out.error.map(|e| e.code), Some("unknown_verb"));
}

#[test]
fn a_line_reports_the_blocks_it_made_changed_and_removed_by_the_graph() {
    let Some(mut app) = headless_app() else { return };
    let made = app.run_line("sphere radius=10 'ball'");
    assert_eq!(made.blocks.created.len(), 1, "{made:?}");
    let ball = made.blocks.created[0].clone();
    assert_eq!((ball.name.as_deref(), ball.kind.as_str(), ball.body), (Some("ball"), "sphere", true));
    assert!(ball.id.starts_with("n:") && made.blocks.changed.is_empty() && made.blocks.removed.is_empty());

    let refused = app.run_line("sphere radus=10");
    assert_eq!(refused.blocks, Default::default(), "a refused line touches nothing");

    // Naming is its own undo step, so the first undo takes the label off and changes the block.
    let unnamed = app.run_line("undo");
    assert_eq!(unnamed.blocks.changed.len(), 1, "{unnamed:?}");
    let undone = app.run_line("undo");
    assert_eq!(undone.blocks.removed.len(), 1, "{undone:?}");
    assert_eq!(undone.blocks.removed[0].id, ball.id);
}

/// What each key's field holds on the block its verb makes with no keys given, rendered as
/// compact JSON with a number slot's `{"literal": 5}` shown as `5`. Feeds `key_defaults::DEFAULTS`.
/// A verb that needs an input and is refused bare simply has no entry: unknown, not guessed.
pub(crate) fn derived_key_defaults() -> Option<std::collections::BTreeMap<(String, String), String>> {
    use serde_json::Value as J;
    fn plain(v: &J) -> J {
        match v {
            J::Object(o) if o.len() == 1 && o.contains_key("literal") => o["literal"].clone(),
            J::Array(a) => J::Array(a.iter().map(plain).collect()),
            other => other.clone(),
        }
    }
    let mut out = std::collections::BTreeMap::new();
    for verb in crate::console_keys::VERBS {
        let crate::console_keys::Keys::Declared { reads, .. } = &verb.keys else { continue };
        if !verb.op.starts_with("create.") {
            continue;
        }
        let Some((word, _)) = crate::console_commands().into_iter().find(|(_, op)| *op == verb.op) else { continue };
        let mut app = headless_app()?;
        let made = app.run_line(&word);
        let Some(block) = made.blocks.created.iter().find(|b| b.id.starts_with("n:")).filter(|_| made.ok) else { continue };
        let raw = u64::from_str_radix(block.id.trim_start_matches("n:"), 16).unwrap();
        let graph = app.part_graph();
        let Some((_, node)) = graph.nodes().find(|(id, _)| id.raw() == raw) else { continue };
        let json = serde_json::to_value(node).unwrap();
        for k in reads.iter() {
            let [field] = k.covers else { continue };
            // A block's id is minted fresh each run, so a default that is one is not a default.
            let is_id = |v: &J| v.as_str().is_some_and(|s| s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit()));
            if let Some(v) = json.get(*field).filter(|v| !is_id(v)) {
                out.insert((verb.op.to_string(), k.word.to_string()), plain(v).to_string());
            }
        }
    }
    Some(out)
}

#[test]
fn the_key_defaults_match_what_the_verbs_make() {
    let Some(derived) = derived_key_defaults() else { return };
    if std::env::var_os("FIELDFORM_PRINT_KEY_DEFAULTS").is_some() {
        for ((op, key), v) in &derived {
            println!("    ({op:?}, {key:?}, {v:?}),");
        }
    }
    let written: std::collections::BTreeMap<(String, String), String> = crate::key_defaults::DEFAULTS
        .iter()
        .map(|(o, k, v)| ((o.to_string(), k.to_string()), v.to_string()))
        .collect();
    assert_eq!(written, derived, "regenerate DEFAULTS (see key_defaults.rs)");
}

#[test]
fn a_line_that_makes_a_parameter_or_a_curve_says_so() {
    let Some(mut app) = headless_app() else { return };
    let made = app.run_line("param name=wall value=3");
    let p = made.blocks.created.iter().find(|b| b.kind == "parameter");
    assert!(p.is_some_and(|p| p.name.as_deref() == Some("wall") && p.id.starts_with("p:")), "{made:?}");
    let edited = app.run_line("param name=wall value=4");
    assert!(edited.blocks.changed.iter().any(|b| b.kind == "parameter"), "{edited:?}");
}

#[test]
fn every_example_runs_on_a_new_document() {
    for (verb, lines) in crate::examples::EXAMPLES {
        let Some(mut app) = headless_app() else { return };
        assert!(crate::catalogue::describe(verb).is_some(), "`{verb}` is an example for no verb");
        for line in lines.lines() {
            let out = app.run_line(line);
            assert!(out.ok, "the example for `{verb}` was refused at `{line}`: {:?}", out.error);
        }
    }
}

// ------------------------------------------------------------- a curve's row, and its joins

/// Where `at` falls in the viewport, which must have been drawn and must show it.
fn on_screen(app: &App, at: Point3<f64>) -> egui::Pos2 {
    let rect = app.viewport_rect.expect("the viewport has been drawn");
    let aspect = rect.width() / rect.height();
    let pos = app.camera.project_to_screen(aspect, rect, at.cast::<f32>()).expect("in front of the camera");
    assert!(rect.contains(pos), "{at} is off the viewport at {pos:?}");
    pos
}

/// How tall `block`'s row is drawn, on a frame of its own.
fn row_height(app: &mut App, block: Block) -> f32 {
    app.drawn_frames.clear();
    frame(app, Vec::new());
    app.drawn_frames.iter().find(|f| f.block == block).map(|f| f.rect.height()).expect("the row was drawn")
}

/// Two lines apart, framed, and the list settled.
fn two_lines(app: &mut App) -> (CurveId, CurveId) {
    frame(app, Vec::new());
    app.run_command("line p0=0,0,0 p1=40,0,0 'a'");
    app.run_command("line p0=0,30,0 p1=40,30,0 'b'");
    app.select_curve(app.curve_named("a").expect("a"));
    app.run_command("frameall");
    app.selected_curve = None;
    for _ in 0..40 {
        frame(app, Vec::new());
    }
    (app.curve_named("a").expect("a"), app.curve_named("b").expect("b"))
}

/// **A curve clicked in the viewport to be edited opens its own row** (the owner, 2026-11-05),
/// not only the rows above it, which is all the list used to open for a selection. The other
/// line's row is the control: drawn in the same list, never clicked, still closed.
#[test]
fn a_curve_clicked_in_the_viewport_opens_its_own_row() {
    let Some(mut app) = headless_app() else { return };
    let (a, b) = two_lines(&mut app);
    let closed = row_height(&mut app, Block::Curve(a));
    assert_eq!(closed, row_height(&mut app, Block::Curve(b)), "both rows start closed");
    let at = on_screen(&app, Point3::new(20.0, 0.0, 0.0));
    left_click(&mut app, at);
    assert_eq!(app.selected_curve, Some(a), "the click picked the line");
    for _ in 0..40 {
        frame(&mut app, Vec::new());
    }
    assert!(row_height(&mut app, Block::Curve(a)) > closed + 10.0, "the clicked curve's row is open");
    assert_eq!(row_height(&mut app, Block::Curve(b)), closed, "and the other's is not");
}

/// **A curve selected without a click on it leaves its row as it was**: a line typed, a paste,
/// a script of twenty lines would otherwise leave every row it made open behind it.
#[test]
fn a_curve_selected_without_a_click_keeps_its_row_closed() {
    let Some(mut app) = headless_app() else { return };
    let (a, _) = two_lines(&mut app);
    let closed = row_height(&mut app, Block::Curve(a));
    app.select_curve(a);
    for _ in 0..40 {
        frame(&mut app, Vec::new());
    }
    assert_eq!(row_height(&mut app, Block::Curve(a)), closed);
}

/// **A join is two ends meeting with nothing else there, held by a point or not.** An open chain
/// of three has two; a third end at a corner takes that corner's away; a closed loop of four has
/// four before it is held and the same four after.
#[test]
fn a_join_is_two_ends_meeting_and_nothing_else_there() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=40,0,0 'a'",
        "line p0=40,0,0 p1=40,20,0 'b'",
        "line p0=40,20,0 p1=0,20,0 'c'",
    ] {
        app.run_command(line);
    }
    assert_eq!(joins_in(&app.doc).len(), 2, "an open chain of three");
    app.run_command("line p0=40,0,0 p1=80,-10,0 'stray'");
    assert_eq!(joins_in(&app.doc).len(), 1, "three ends at a corner are not a join");
    app.run_command("delete 'stray'");
    app.run_command("line p0=0,20,0 p1=0,0,0 'd'");
    frame(&mut app, Vec::new());
    let held = app.doc.parts[&app.active_part].graph.nodes().filter(|(_, n)| matches!(n, Node::Point { .. })).count();
    assert_eq!(held, 4, "the closed loop was held with a point at each corner");
    assert_eq!(joins_in(&app.doc).len(), 4, "and each corner is still a join");
}

/// An open chain: a line, then a cubic leaving its end at an angle.
fn line_then_cubic(app: &mut App) -> (CurveId, CurveId) {
    frame(app, Vec::new());
    app.run_command("line p0=0,0,0 p1=40,0,0 'a'");
    app.run_command("curve p=40,0,0;50,10,0;60,10,0;70,0,0 'b'");
    (app.curve_named("a").expect("a"), app.curve_named("b").expect("b"))
}

fn joint_continuity(app: &App) -> Vec<fieldform_scene::JointContinuity> {
    app.doc
        .relations
        .values()
        .filter_map(|r| match r.relation {
            fieldform_scene::Relation::Joint { continuity, .. } => Some(continuity),
            _ => None,
        })
        .collect()
}

/// **G1, G2 and G0 at an open chain's join, each one undo step** -- where joint points are never
/// made, which is why the join is read from the ends and not from a point. G1 lays the cubic's
/// second point on the line's direction; G2 its third as well, a line having no curvature to
/// match; G0 takes the relation out and moves nothing back.
#[test]
fn an_open_chains_join_takes_g1_g2_and_g0_each_as_one_step() {
    let Some(mut app) = headless_app() else { return };
    let (a, b) = line_then_cubic(&mut app);
    let point = |app: &App, i: usize| app.doc.curve(b).expect("b").control_points()[i];

    app.run_command("continuity at=a.end g=1");
    assert_eq!(joint_continuity(&app), vec![fieldform_scene::JointContinuity::G1]);
    let master = app.doc.relations.values().find_map(|r| match r.relation {
        fieldform_scene::Relation::Joint { master, .. } => Some(master.curve),
        _ => None,
    });
    assert_eq!(master, Some(a), "the earlier curve leads");
    assert!(point(&app, 1).y.abs() < 1e-9 && point(&app, 1).x > 40.0, "tangent: {}", point(&app, 1));
    assert!(point(&app, 2).y.abs() > 1.0, "G1 leaves the third point alone");

    app.run_command("continuity at=b.start g=2");
    assert_eq!(joint_continuity(&app), vec![fieldform_scene::JointContinuity::G2], "replaced, not added");
    assert!(point(&app, 2).y.abs() < 1e-9, "no curvature off a line: {}", point(&app, 2));

    let moved = point(&app, 2);
    app.run_command("continuity at=a.end g=0");
    assert!(joint_continuity(&app).is_empty(), "G0 is no relation");
    assert_eq!(point(&app, 2), moved, "and breaking one moves nothing");

    app.run_command("undo");
    assert_eq!(joint_continuity(&app), vec![fieldform_scene::JointContinuity::G2], "one undo, one step");
    app.run_command("undo");
    assert_eq!(joint_continuity(&app), vec![fieldform_scene::JointContinuity::G1]);
}

/// **A continuity is refused, by name, where there is no join**: an end that meets nothing, and
/// a corner where three ends meet. Nothing is made either time.
#[test]
fn continuity_is_refused_where_two_ends_do_not_meet() {
    let Some(mut app) = headless_app() else { return };
    line_then_cubic(&mut app);
    let warned = app.console_warnings;
    app.run_command("continuity at=a.start g=1");
    assert_eq!(app.console_warnings, warned + 1, "a's start meets nothing");
    app.run_command("line p0=40,0,0 p1=40,-30,0 'stray'");
    let warned = app.console_warnings;
    app.run_command("continuity at=a.end g=1");
    assert_eq!(app.console_warnings, warned + 1, "three ends meet there");
    app.run_command("continuity at=a.end g=3");
    assert_eq!(app.console_warnings, warned + 2, "G3 is not on offer");
    assert!(joint_continuity(&app).is_empty());
}



/// An open chain of two cubics, meeting at an angle at 40,0,0: either can lead the other.
fn two_cubics(app: &mut App) -> (CurveId, CurveId) {
    frame(app, Vec::new());
    app.run_command("curve p=0,0,0;10,-10,0;30,-20,0;40,0,0 'a'");
    app.run_command("curve p=40,0,0;50,10,0;60,10,0;70,0,0 'b'");
    (app.curve_named("a").expect("a"), app.curve_named("b").expect("b"))
}

/// The open chain framed, its join right-clicked and North picked: the list of which curve leads.
fn open_lead_list(app: &mut App) -> (CurveId, CurveId, egui::Pos2) {
    let (a, b) = two_cubics(app);
    app.select_curve(a);
    app.run_command("frameall");
    app.selected_curve = None;
    for _ in 0..10 {
        frame(app, Vec::new());
    }
    let at = on_screen(app, Point3::new(40.0, 0.0, 0.0));
    frame(app, vec![egui::Event::PointerMoved(at)]);
    right_click(app, at);
    let menu = app.marking.as_ref().expect("the right-click opened a menu");
    assert_eq!(menu.subject, marking::Subject::Joint, "the join's ring, with nothing selected");
    assert!(menu.join.is_some_and(|j| j.iter().any(|e| e.curve == a)), "on the join of a and b");
    let (dx, dy) = marking::slot_offset(0, 84.0);
    left_click(app, at + egui::vec2(dx, dy));
    (a, b, at)
}

/// Walks the pointer down the lead list and says where each row's preview showed, by the curve
/// that leads in it, in the order met.
fn lead_rows(app: &mut App) -> Vec<(CurveId, egui::Pos2)> {
    // A frame first, so the rect read is where the list is drawn now and not where it last was.
    frame(app, Vec::new());
    let area = app.egui_ctx.memory(|m| m.area_rect(egui::Id::new("join lead list"))).expect("the list is drawn");
    let mut rows: Vec<(CurveId, egui::Pos2)> = Vec::new();
    let mut y = area.top();
    while y < area.bottom() {
        // Down the middle: the rows' left edge is inside the frame's margin.
        let p = egui::pos2(area.center().x, y);
        frame(app, vec![egui::Event::PointerMoved(p)]);
        if let Some(fieldform_scene::Relation::Joint { master, .. }) = app.join_preview.as_ref().map(|p| p.relation.clone())
            && rows.last().is_none_or(|(m, _)| *m != master.curve)
        {
            rows.push((master.curve, p));
        }
        y += 2.0;
    }
    rows
}

/// **G1 on a join asks which curve leads, and shows each before it is kept** (the owner,
/// 2026-11-07: "as u hover over one it automatically changes so u can see the effect of your
/// choice in real time"). Hovering a row moves the follower's points with no relation stored and
/// nothing on the undo stack; leaving the rows puts them back; a click keeps the one under it as
/// one step, echoed with its lead so it replays.
#[test]
fn g1_on_a_join_lists_each_lead_shows_it_on_hover_and_keeps_the_one_clicked() {
    let Some(mut app) = headless_app() else { return };
    let (a, b, _) = open_lead_list(&mut app);
    assert_eq!(app.marking.as_ref().and_then(|m| m.lead_for), Some(fieldform_scene::JointContinuity::G1));
    assert!(joint_continuity(&app).is_empty(), "nothing is kept by opening the list");
    let points = |app: &App, id: CurveId| app.doc.curve(id).expect("curve").control_points().to_vec();
    let (a_before, b_before, history) = (points(&app, a), points(&app, b), app.stack.history());

    let rows = lead_rows(&mut app);
    assert_eq!(rows.iter().map(|(m, _)| *m).collect::<Vec<_>>(), vec![a, b], "a row per curve, the earlier first");
    let (_, a_leads) = rows[0];
    frame(&mut app, vec![egui::Event::PointerMoved(a_leads)]);
    assert_ne!(points(&app, b), b_before, "a leading: b turns to leave along a, shown");
    assert_eq!(points(&app, a), a_before, "and a stays");
    assert!(joint_continuity(&app).is_empty() && app.stack.history() == history, "shown, not kept");
    let (_, b_leads) = rows[1];
    frame(&mut app, vec![egui::Event::PointerMoved(b_leads)]);
    assert_eq!(points(&app, b), b_before, "b leading: b is as it was");
    assert_ne!(points(&app, a), a_before, "and a turns to meet it");

    let area = app.egui_ctx.memory(|m| m.area_rect(egui::Id::new("join lead list"))).expect("still open");
    frame(&mut app, vec![egui::Event::PointerMoved(area.right_bottom() + egui::vec2(40.0, 40.0))]);
    assert!(app.join_preview.is_none(), "off the rows, nothing is shown");
    assert_eq!((points(&app, a), points(&app, b)), (a_before.clone(), b_before.clone()), "and everything is back");

    left_click(&mut app, a_leads);
    assert!(app.marking.is_none() && app.join_preview.is_none(), "kept, and the list is gone");
    let kept: Vec<_> = app.doc.relations.values().map(|r| r.relation.clone()).collect();
    assert!(
        matches!(kept.as_slice(), [fieldform_scene::Relation::Joint { master, continuity: fieldform_scene::JointContinuity::G1, .. }] if master.curve == a),
        "{kept:?}"
    );
    assert_ne!(points(&app, b), b_before, "kept as it was shown");
    let echoed = app.echo_log.iter().rev().find(|l| l.starts_with("continuity")).cloned().expect("echoed");
    assert_eq!(echoed, "continuity at=\"a.end\" g=1 lead=\"a\"");
    app.run_command("undo");
    assert!(joint_continuity(&app).is_empty(), "one undo, one step");
    assert_eq!((points(&app, a), points(&app, b)), (a_before, b_before));

    let Some(mut replay) = headless_app() else { return };
    two_cubics(&mut replay);
    replay.run_command(&echoed);
    assert_eq!(joint_continuity(&replay), vec![fieldform_scene::JointContinuity::G1], "the echo replays");
}

/// **Esc leaves the join as it was**, with a row hovered and its preview up; and so does an undo
/// pressed with the list open, which acts on the document rather than on the preview.
#[test]
fn leaving_the_lead_list_puts_back_what_its_preview_moved() {
    let Some(mut app) = headless_app() else { return };
    let (a, b, _) = open_lead_list(&mut app);
    let points = |app: &App, id: CurveId| app.doc.curve(id).expect("curve").control_points().to_vec();
    let before = (points(&app, a), points(&app, b));
    let rows = lead_rows(&mut app);
    frame(&mut app, vec![egui::Event::PointerMoved(rows[1].1)]);
    assert!(app.join_preview.is_some());
    frame(&mut app, vec![key_event(egui::Key::Escape)]);
    assert!(app.marking.is_none() && app.join_preview.is_none());
    assert_eq!((points(&app, a), points(&app, b)), before, "Esc put it back");
    assert!(joint_continuity(&app).is_empty());

    let (a, b, _) = (a, b, open_lead_list_again(&mut app));
    let rows = lead_rows(&mut app);
    frame(&mut app, vec![egui::Event::PointerMoved(rows[0].1)]);
    assert!(app.join_preview.is_some());
    app.invoke_command(by("edit.undo"));
    frame(&mut app, Vec::new());
    assert!(app.join_preview.is_none(), "the undo ended the preview");
    assert!(app.doc.curve(b).is_none() || points(&app, b) == before.1, "and undid the curve, not the preview");
    let _ = a;
}

/// The ring opened again on the same chain, North picked.
fn open_lead_list_again(app: &mut App) -> egui::Pos2 {
    let at = on_screen(app, Point3::new(40.0, 0.0, 0.0));
    frame(app, vec![egui::Event::PointerMoved(at)]);
    right_click(app, at);
    let (dx, dy) = marking::slot_offset(0, 84.0);
    left_click(app, at + egui::vec2(dx, dy));
    assert!(app.marking.as_ref().is_some_and(|m| m.lead_for.is_some()));
    at
}

/// **A lead the kernel refuses is listed greyed, never shown**: a straight line cannot turn to
/// follow a curve at G1, so on a line meeting a cubic only the line can lead, and hovering the
/// other row moves nothing.
#[test]
fn a_refused_lead_is_listed_but_never_previewed() {
    let Some(mut app) = headless_app() else { return };
    let (a, b) = line_then_cubic(&mut app);
    let join = joins_in(&app.doc)[0].ends;
    let b_leads = fieldform_scene::Relation::Joint {
        master: if join[0].curve == b { join[0] } else { join[1] },
        slave: if join[0].curve == a { join[0] } else { join[1] },
        continuity: fieldform_scene::JointContinuity::G1,
    };
    assert!(fieldform_scene::relation::refuse_reason_replacing(&app.doc, &b_leads, None).is_err(), "the line cannot follow");
    app.select_curve(a);
    app.run_command("frameall");
    app.selected_curve = None;
    for _ in 0..10 {
        frame(&mut app, Vec::new());
    }
    open_lead_list_again(&mut app);
    let rows = lead_rows(&mut app);
    assert_eq!(rows.iter().map(|(m, _)| *m).collect::<Vec<_>>(), vec![a], "only the line can lead");
    // And the reason is drawn under the greyed row, not only on its hover.
    let why = app.joint_refusal(&b_leads, None).expect("refused");
    let drawn = texts_drawn(&mut app, Vec::new());
    assert!(drawn.iter().any(|(t, _)| t.starts_with(&why[..why.len().min(24)])), "{why:?} is on screen: {:?}", drawn.iter().map(|(t, _)| t).collect::<Vec<_>>());
}

/// **A joint refused for want of a control point says what degree frees one** (reported
/// 2026-11-11: three cubics round a loop, the last joint "refuses to go anything other than G0").
/// With G1 from one to two and G2 from two to three, the closing join is refused both ways: one
/// following three would make a ring, and three following one wants the third point its G2 owns.
/// The second says to raise three to degree 4, and at degree 4 the G1 is made.
#[test]
fn a_joint_refused_for_a_shared_point_names_the_degree_that_frees_one() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "curve p=0,0,0;20,-10,0;40,-10,0;60,0,0 'one'",
        "curve p=60,0,0;70,20,0;60,40,0;40,60,0 'two'",
        "curve p=40,60,0;20,60,0;5,40,0;0,0,0 'three'",
        "continuity at=one.end g=1 lead=one",
        "continuity at=two.end g=2 lead=two",
    ] {
        app.run_command(line);
    }
    let relations = app.doc.relations.len();
    app.run_command("continuity at=three.end g=1");
    assert_eq!(app.doc.relations.len(), relations, "refused both ways");
    let said = last_said(&app);
    assert!(said.contains("raise three to degree 4"), "{said}");
    assert!(said.contains("cannot follow"), "both reasons are said: {said}");
    app.run_command("degree degree=4 target=three");
    app.run_command("continuity at=three.end g=1 lead=one");
    assert_eq!(app.doc.relations.len(), relations + 1, "{}", last_said(&app));
    // G2 at both ends of one curve wants degree 5, and says so.
    app.run_command("continuity at=three.end g=2 lead=one");
    assert!(last_said(&app).contains("raise three to degree 5"), "{}", last_said(&app));
}

/// Two cubics framed, the first clicked, then dragged by its middle across the view and let go.
/// Where the second's start and the first's end ended up.
fn drag_the_first_cubic(app: &mut App, a: CurveId, b: CurveId) -> (Point3<f64>, Point3<f64>) {
    app.select_curve(a);
    app.run_command("frameall");
    for _ in 0..10 {
        frame(app, Vec::new());
    }
    // a at t = 1/2: (0 + 3*10 + 3*30 + 40)/8, (0 - 30 - 60 + 0)/8.
    let at = on_screen(app, Point3::new(20.0, -11.25, 0.0));
    left_click(app, at);
    assert_eq!(app.selected_curve, Some(a));
    frame(app, vec![press(at, egui::PointerButton::Primary, true)]);
    let mut pos = at;
    for _ in 0..6 {
        pos += egui::vec2(0.0, -15.0);
        frame(app, vec![egui::Event::PointerMoved(pos)]);
    }
    frame(app, vec![press(pos, egui::PointerButton::Primary, false)]);
    frame(app, Vec::new());
    let end_of_a = *app.doc.curve(a).expect("a").control_points().last().expect("points");
    let start_of_b = app.doc.curve(b).expect("b").control_points()[0];
    (end_of_a, start_of_b)
}

/// **Ends linked by a continuity stay together when either curve is dragged, until the join is
/// broken** (the owner, 2026-11-07: "the linked points should stay magnetically connected until u
/// RMB click and break it"). Dragging the first of two cubics set G1 carries the second's start
/// with it and keeps the tangent; the same drag with nothing set, and after Break, parts them.
#[test]
fn ends_linked_by_a_continuity_move_together_until_the_join_is_broken() {
    let tangent_kept = |app: &App, a: CurveId, b: CurveId| {
        let pa = app.doc.curve(a).expect("a").control_points().to_vec();
        let pb = app.doc.curve(b).expect("b").control_points().to_vec();
        let (into, out) = (pa[3] - pa[2], pb[1] - pb[0]);
        into.normalize().dot(&out.normalize()) > 1.0 - 1e-9
    };

    // Linked: G1 holds the ends, and the drag takes both.
    let Some(mut app) = headless_app() else { return };
    let (a, b) = two_cubics(&mut app);
    app.run_command("continuity at=a.end g=1");
    let (end_of_a, start_of_b) = drag_the_first_cubic(&mut app, a, b);
    assert!((end_of_a - Point3::new(40.0, 0.0, 0.0)).norm() > 1.0, "the drag moved a: {end_of_a}");
    assert!((end_of_a - start_of_b).norm() < 1e-9, "b's start came with it: {end_of_a} {start_of_b}");
    assert!(tangent_kept(&app, a, b), "and still leaves along a");

    // The control: the same drag on ends that only touch parts them.
    let Some(mut loose) = headless_app() else { return };
    let (a, b) = two_cubics(&mut loose);
    let (end_of_a, start_of_b) = drag_the_first_cubic(&mut loose, a, b);
    assert!((end_of_a - start_of_b).norm() > 1.0, "nothing linked them: {end_of_a} {start_of_b}");

    // Broken: set, then let go, then dragged.
    let Some(mut broken) = headless_app() else { return };
    let (a, b) = two_cubics(&mut broken);
    broken.run_command("continuity at=a.end g=2");
    broken.run_command("continuity at=b.start -break");
    assert!(joint_continuity(&broken).is_empty(), "Break took the continuity out");
    let points = broken.part_graph().nodes().filter(|(_, n)| matches!(n, Node::Point { .. })).count();
    assert_eq!(points, 0, "and the point that held them");
    let (end_of_a, start_of_b) = drag_the_first_cubic(&mut broken, a, b);
    assert!((end_of_a - start_of_b).norm() > 1.0, "broken, they part: {end_of_a} {start_of_b}");
}

/// **G0 holds too, since that is what G0 is**, as one undo step, and the point it holds the join
/// with takes no row of its own in the list: it is the join's, worked from the join's ring.
#[test]
fn g0_holds_a_join_with_a_point_that_takes_no_row() {
    let Some(mut app) = headless_app() else { return };
    let (a, b) = two_cubics(&mut app);
    let history = app.stack.history();
    app.run_command("continuity at=a.end g=0");
    let end = |app: &App, c: CurveId, i: usize| app.doc.curve(c).expect("curve").placed_on(i);
    let n = app.doc.curve(a).expect("a").control_points().len();
    assert!(end(&app, a, n - 1).is_some() && end(&app, a, n - 1) == end(&app, b, 0), "both ends on one point");
    assert!(joint_continuity(&app).is_empty(), "positional only");
    let point = end(&app, a, n - 1).expect("held");
    let joints = crate::blocks::list::held_joint_points(&app.doc, app.active_part, &app.profiles);
    assert!(joints.contains(&point), "the join's point is not listed as a block");
    app.run_command("undo");
    assert_eq!(app.stack.history().0, history.0, "one step");
    assert!(end(&app, a, n - 1).is_none() && end(&app, b, 0).is_none());
}


/// Every piece of text one frame drew, with where it was drawn: a frame like [`frame`], with the
/// shapes kept rather than dropped, so a test can find a word on screen and click it.
fn texts_drawn(app: &mut App, events: Vec<egui::Event>) -> Vec<(String, egui::Rect)> {
    let ctx = app.egui_ctx.clone();
    let mut input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))),
        events,
        ..Default::default()
    };
    app.raw_input_hook(&ctx, &mut input);
    let mut eframe_frame = eframe::Frame::_new_kittest();
    let out = ctx.run_ui(input, |ui| app.ui(ui, &mut eframe_frame));
    fn walk(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
        match shape {
            egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.visual_bounding_rect())),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }
    let mut found = Vec::new();
    for clipped in &out.shapes {
        walk(&clipped.shape, &mut found);
    }
    found
}

/// **An offset lists no point, and its filled field clears from a right-click** (the owner,
/// 2026-11-07: "things like the offset tool should not list items like points ... as input", and
/// "u should be able to RMB click on it and CLEAR should be an option"). The point is in the part
/// and a sphere is listed beside it, so the list is not simply empty.
#[test]
fn an_offsets_field_lists_no_point_and_clears_from_a_right_click() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in ["sphere 'ball'", "sphere 'other' centre=40,0,0", "point at=80,0,0", "offset target=ball distance=2 'off'"] {
        app.run_command(line);
    }
    let off = app.node_named("off").expect("the offset");
    let listed: Vec<String> = crate::suggested(&app.operand_offers(off, &app.operand_choices(off)), "", false)
        .into_iter()
        .map(|(_, label, _)| label)
        .collect();
    assert!(listed.iter().any(|l| l == "other"), "a solid is listed: {listed:?}");
    assert!(!listed.iter().any(|l| l.starts_with("Point")), "and no point: {listed:?}");

    // The ball selected opens the offset's row above it, where its field is drawn.
    app.select_body(app.node_named("ball"));
    for _ in 0..40 {
        frame(&mut app, Vec::new());
    }
    let texts = texts_drawn(&mut app, Vec::new());
    let of = texts.iter().find(|(t, _)| t == "Of:").map(|(_, r)| *r).expect("the offset's Of: row is drawn");
    let field = egui::pos2(of.right() + 40.0, of.center().y);
    frame(&mut app, vec![egui::Event::PointerMoved(field)]);
    right_click(&mut app, field);
    let before: usize = texts.iter().filter(|(t, _)| t == "Clear").count();
    let after = texts_drawn(&mut app, Vec::new());
    let menu_clear = after
        .iter()
        .filter(|(t, _)| t == "Clear")
        .map(|(_, r)| *r)
        .max_by(|a, b| a.top().total_cmp(&b.top()))
        .expect("Clear on the menu");
    assert_eq!(after.iter().filter(|(t, _)| t == "Clear").count(), before + 1, "the right-click added a Clear");
    left_click(&mut app, menu_clear.center());
    let Some(Node::Offset { input, .. }) = app.part_graph().node(off) else { panic!("the offset is gone") };
    assert!(input.is_empty(), "the slot is empty");
    assert!(app.part_graph().node(app.node_named("ball").expect("ball")).is_some(), "and the ball is kept");
}

/// **The console dragged as small as it goes keeps its log off the entry field** (reported
/// 2026-11-07: "the console text overlays the input field and its not legible"). The log's newest
/// line is still drawn, and nothing of it reaches the field.
#[test]
fn the_console_dragged_to_its_smallest_keeps_its_log_off_the_entry_field() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for i in 0..30 {
        app.say(format!("line {i} of the log"));
    }
    let find = |texts: &[(String, egui::Rect)], t: &str| texts.iter().find(|(s, _)| s == t).map(|(_, r)| *r);
    let before = texts_drawn(&mut app, Vec::new());
    let heading = find(&before, "Console").expect("the console heading");
    // The divider is the panel's top edge, a little above the heading: try down from there until
    // a drag to the bottom of the window moves the heading.
    let mut dragged = false;
    for lift in (4..30).step_by(2) {
        let grip = egui::pos2(heading.left() + 40.0, heading.top() - lift as f32);
        frame(&mut app, vec![egui::Event::PointerMoved(grip)]);
        frame(&mut app, vec![press(grip, egui::PointerButton::Primary, true)]);
        for y in [grip.y + 100.0, grip.y + 300.0, 899.0] {
            frame(&mut app, vec![egui::Event::PointerMoved(egui::pos2(grip.x, y))]);
        }
        frame(&mut app, vec![press(egui::pos2(grip.x, 899.0), egui::PointerButton::Primary, false)]);
        frame(&mut app, Vec::new());
        let now = find(&texts_drawn(&mut app, Vec::new()), "Console").expect("still drawn");
        if now.top() > heading.top() + 20.0 {
            dragged = true;
            break;
        }
    }
    assert!(dragged, "found the console's divider and dragged it down");
    let texts = texts_drawn(&mut app, Vec::new());
    let field = find(&texts, "command").expect("the entry field's hint");
    let log: Vec<&(String, egui::Rect)> = texts.iter().filter(|(t, _)| t.starts_with("line ")).collect();
    assert!(log.iter().any(|(t, _)| t == "line 29 of the log"), "the newest line is still shown: {log:?}");
    for (t, r) in &log {
        assert!(r.bottom() <= field.top() + 0.5, "{t:?} at {r:?} runs onto the field at {field:?}");
    }
}

// ------------------------------------------------------------------------ the section block

/// **A section is a block of the document** (the owner, 2026-11-07: "it should be a block"):
/// made from the console where it is told, every change one undo step, and saved with the file,
/// so it is there, where it was, when the file is opened again.
#[test]
fn a_section_is_saved_with_the_document_and_each_change_is_one_undo_step() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere radius=20 'ball'");
    app.run_command("addsection origin=0,0,10 turn=90,0,0");
    let made = app.sections();
    assert_eq!(made.len(), 1);
    assert!((made[0].origin - Point3::new(0.0, 0.0, 10.0)).norm() < 1e-9, "where it was told");
    let n = made[0].normal();
    assert!((n - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-9, "turned 90 about X, +Z faces -Y: {n}");
    assert_eq!(app.section_half_spaces().len(), 1, "and it cuts");

    app.flip_section_at(0);
    assert!(app.sections()[0].flipped);
    app.run_command("undo");
    assert!(!app.sections()[0].flipped, "a flip is one step");
    app.run_command("section");
    assert!(!app.sections()[0].on);
    app.run_command("undo");
    assert!(app.sections()[0].on, "so is the switch");

    let dir = std::env::temp_dir().join(format!("fieldform-section-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("cut.fform");
    app.run_command(&format!("saveas file=\"{}\"", file.to_string_lossy().replace('\\', "/")));
    app.run_command("new");
    assert!(app.sections().is_empty(), "a new document has none");
    app.run_command(&format!("open file=\"{}\"", file.to_string_lossy().replace('\\', "/")));
    let back = app.sections();
    assert_eq!(back.len(), 1, "it came back with the file");
    assert!((back[0].origin - made[0].origin).norm() < 1e-9 && (back[0].normal() - n).norm() < 1e-9);
    assert!(back[0].on);
    let _ = std::fs::remove_dir_all(&dir);

    app.run_command("addsection");
    assert_eq!(app.sections().len(), 2);
    app.run_command("undo");
    assert_eq!(app.sections().len(), 1, "making one is one step");
}

/// **A section is placed the way a plane block is** (the owner, 2026-11-07: "same machinery as the
/// plane block"): its origin is a `Place`, so it can sit on a point and goes where the point goes,
/// read through the same function a plane's origin is read through.
#[test]
fn a_section_can_sit_on_a_point_and_follows_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("point at=5,6,7 'pin'");
    app.run_command("addsection");
    let pin = app.node_named("pin").expect("the point");
    let id = app.sections()[0].id;
    let sections: Vec<fieldform_scene::SectionCut> = app
        .doc
        .sections
        .iter()
        .map(|c| fieldform_scene::SectionCut { origin: fieldform_scene::Place::From { node: pin }, ..*c })
        .collect();
    app.set_sections(sections, Some(id));
    assert!((app.sections()[0].origin - Point3::new(5.0, 6.0, 7.0)).norm() < 1e-9, "on the point");
    app.run_command("set pin.at=1,2,3");
    assert!((app.sections()[0].origin - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-9, "and where it went");
}

/// **A dragged section is written where the drag left it**: turned and moved by the handles, the
/// record it is put back into reads as the same plane, which is what makes the drag's undo entry
/// land the section where it was dropped rather than somewhere near.
#[test]
fn a_dragged_section_is_written_where_the_drag_left_it() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("addsection origin=1,2,3 turn=10,20,30");
    let resolved = app.sections()[0];
    let record = app.doc.sections[0];
    let pivot = Point3::new(4.0, -1.0, 2.0);
    let turned = resolved.with(gizmo::Motion::Rotate { axis: Vector3::new(1.0, 1.0, 0.0), degrees: 37.0 }, pivot);
    let moved = turned.with(gizmo::Motion::Translate(Vector3::new(3.0, 0.0, -2.0)), pivot);
    let back = app.resolve_section(&crate::placed_section(record, &moved));
    assert!((back.origin - moved.origin).norm() < 1e-9, "{} vs {}", back.origin, moved.origin);
    assert!((back.normal() - moved.normal()).norm() < 1e-9, "{} vs {}", back.normal(), moved.normal());
    assert!(back.rotation.angle_to(&moved.rotation) < 1e-9, "the whole frame, not only the normal");
}

/// **Every number a section takes is a key of `addsection`**, or recorded as one it does not read,
/// with why: the drift test for a block that is a record of the document rather than a node.
#[test]
fn every_field_of_a_section_is_a_key_addsection_reads_or_a_recorded_gap() {
    use std::collections::BTreeSet;
    let cut = fieldform_scene::SectionCut {
        id: fieldform_scene::SectionCutId::new(),
        number: 1,
        part: fieldform_scene::PartId::new(),
        on: true,
        origin: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
        turn: fieldform_scene::Turn::default(),
        flipped: true,
    };
    let json = serde_json::to_value(cut).unwrap();
    let fields: BTreeSet<String> = json.as_object().unwrap().keys().cloned().collect();
    let verb = console_keys::verb("view.section_new").expect("addsection declares its keys");
    let console_keys::Keys::Declared { reads, not_read } = &verb.keys else { panic!("declared") };
    let covered: BTreeSet<&str> = reads
        .iter()
        .flat_map(|k| k.covers.iter().copied())
        .chain(not_read.iter().map(|(f, _)| *f))
        .collect();
    for field in &fields {
        assert!(covered.contains(field.as_str()), "a section's field {field:?} is reached by no `addsection` key and not recorded as unread");
    }
}

/// **A left click away from the palette closes it, having run nothing** (the owner, 2026-11-07:
/// Space's command input "should disappear" when you click away). The control: a click on the
/// palette's own field leaves it open.
#[test]
fn a_click_away_from_the_palette_closes_it_and_runs_nothing() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    frame(&mut app, vec![key_event(egui::Key::Space), egui::Event::Text(" ".into())]);
    type_text(&mut app, "sph");
    frame(&mut app, Vec::new());
    assert!(app.palette.is_some(), "Space opened it");
    let area = app.egui_ctx.memory(|m| m.area_rect(egui::Id::new("command palette area"))).expect("drawn");
    left_click(&mut app, area.left_top() + egui::vec2(40.0, 16.0));
    assert!(app.palette.is_some(), "a click on its own field keeps it");
    let nodes = app.part_graph().nodes().count();
    left_click(&mut app, egui::pos2(area.center().x, area.bottom() + 120.0));
    assert!(app.palette.is_none(), "a click away closes it");
    assert_eq!(app.part_graph().nodes().count(), nodes, "and nothing ran");
}

/// **A new document starts in chrome under the Horizon environment** (the owner, 2026-11-07), on
/// launch and from File › New alike; and **a file saved without a material opens as it was**, in the
/// neutral preset, because the choice is written into the new part rather than into what no
/// material means.
#[test]
fn a_new_document_starts_in_chrome_under_horizon_and_an_older_file_keeps_its_look() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    let look = |app: &App| app.doc.parts.get(&app.active_part).and_then(|p| p.material.as_ref()).and_then(|m| m.preset_name());
    assert_eq!(app.environment, fieldform_gpu::EnvironmentChoice::Horizon, "on launch");
    assert_eq!(look(&app), Some("Chrome"));
    app.run_command("material preset=glass");
    assert_eq!(look(&app), Some("Glass"), "changed, so the next line has something to start over from");
    // Saved as far as the test is concerned, so `new` has nothing to ask about.
    app.mark_saved();
    app.run_command("new");
    assert_eq!(look(&app), Some("Chrome"), "File › New starts in chrome again");

    let dir = std::env::temp_dir().join(format!("fieldform-look-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("older.fform");
    let older = fieldform_scene::Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
    assert!(older.parts.values().all(|p| p.material.is_none()), "written with no material");
    fieldform_io::save_new(&older, &file).expect("writes");
    app.run_command(&format!("open file=\"{}\"", file.to_string_lossy().replace('\\', "/")));
    assert!(
        app.doc.parts.get(&app.active_part).is_some_and(|p| p.material.is_none()),
        "an older file keeps the neutral look it was saved with"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// **The menus that read the part sit in Evaluate** (the owner, 2026-11-08): Section, New section,
/// Measure, Thickness and the curvature comb, and none of them in View any more, which is about
/// how things look. Flip section is gone from every menu.
#[test]
fn the_evaluate_menu_holds_the_sections_the_measures_and_the_comb() {
    let bar = commands::menu_bar();
    let ids = |title: &str| -> Vec<&'static str> {
        let (_, items) = bar.iter().find(|(t, _)| *t == title).expect("the menu is there");
        commands::menu_commands(items).iter().map(|c| c.id).collect()
    };
    let evaluate = ids("Evaluate");
    for id in ["view.section", "view.section_new", "view.measure", "view.thickness", "evaluate.comb"] {
        assert!(evaluate.contains(&id), "{id} is in Evaluate: {evaluate:?}");
        assert!(!ids("View").contains(&id), "{id} has left View");
    }
    assert!(
        bar.iter().all(|(_, items)| !commands::menu_commands(items).iter().any(|c| c.id == "view.section_flip")),
        "no Flip section in any menu"
    );
}

/// A curve, in a part with an open cubic `bend` and nothing else.
fn app_with_a_bend() -> Option<App> {
    let mut app = headless_app()?;
    frame(&mut app, Vec::new());
    app.run_command("curve p=0,0,0;10,15,0;30,15,0;40,0,0 'bend'");
    frame(&mut app, Vec::new());
    Some(app)
}

/// The console's last few lines, newest first, for a failure to say what the app said.
fn last_said(app: &App) -> String {
    app.console_log.iter().rev().take(6).map(|l| l.text.clone()).collect::<Vec<_>>().join(" | ")
}

fn the_comb(app: &App) -> fieldform_scene::CurvatureComb {
    assert_eq!(app.doc.combs.len(), 1, "one comb: {}", last_said(app));
    app.doc.combs[0].clone()
}

/// **A comb is fitted once, when it is made, and not again on its own**: a control point moved
/// leaves its scale where it was -- a comb that re-fitted itself would draw a flattened bulge as it
/// drew the bulge -- and `-fit` re-fits it, as one undo step.
#[test]
fn a_comb_is_fitted_once_and_not_refitted_when_the_curve_moves() {
    let Some(mut app) = app_with_a_bend() else { return };
    app.run_command("comb curve=bend");
    let made = the_comb(&app);
    let curve = app.doc.curve(made.curve).cloned().expect("the bend");
    assert!((made.scale - fieldform_scene::fitted_scale(&curve, made.teeth)).abs() < 1e-12, "fitted when made");
    assert_eq!(app.selected_block, Some(outputs::BlockRef::Comb(made.id)), "and selected, so the next scrub is on it");

    app.run_command("set bend.p1=10,40,0");
    frame(&mut app, Vec::new());
    assert!((app.doc.curve(made.curve).unwrap().control_points()[1] - Point3::new(10.0, 40.0, 0.0)).norm() < 1e-9, "moved: {}", last_said(&app));
    assert_eq!(the_comb(&app).scale, made.scale, "a moved point leaves the scale alone");

    // `comb fit` is about the comb in hand, so it is selected again after the edit took the
    // selection to the curve.
    app.select_block(outputs::BlockRef::Comb(made.id));
    app.run_command("comb -fit");
    let fitted = the_comb(&app).scale;
    assert_ne!(fitted, made.scale, "fit re-fits to the curve as it is now");
    app.run_command("undo");
    assert_eq!(the_comb(&app).scale, made.scale, "a fit is one step");
}

/// The keys a comb takes, and the ones it refuses by name: a curve that is not there, a tooth
/// count out of range, a negative scale. A refusal adds nothing.
#[test]
fn a_comb_takes_its_keys_from_the_console_and_refuses_what_is_out_of_range() {
    let Some(mut app) = app_with_a_bend() else { return };
    app.run_command("comb curve=bend teeth=96 scale=12 -flip 'check'");
    let comb = the_comb(&app);
    assert_eq!((comb.teeth, comb.scale, comb.flip, comb.title()), (96, 12.0, true, "check"));
    app.run_command("comb block=check flip=false -hidden");
    let comb = the_comb(&app);
    assert!(!comb.flip && comb.hidden, "block= changes the comb that exists");
    for refused in ["comb curve=nosuch", "comb curve=bend teeth=1", "comb curve=bend scale=-1", "comb block=nosuch -fit"] {
        app.run_command(refused);
        assert_eq!(app.doc.combs.len(), 1, "{refused} adds nothing");
    }
    app.select_body(None);
    app.selected_curve = None;
    app.selected_block = None;
    app.run_command("comb");
    assert_eq!(app.doc.combs.len(), 1, "a comb with no curve in hand is refused, naming the key");
}

/// **A comb's row sits directly below its curve's**, and **the comb is not a reader**: adding one
/// to a profile an extrude reads leaves every other row where it was, nested as it was.
#[test]
fn adding_a_comb_to_an_extruded_profile_moves_no_row() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=40,0,0 'a'",
        "line p0=40,0,0 p1=40,20,0 'b'",
        "line p0=40,20,0 p1=0,20,0 'c'",
        "line p0=0,20,0 p1=0,0,0 'd'",
        "extrude profile=a height=10 'slab'",
        "curve p=0,50,0;10,65,0;30,65,0;40,50,0 'bend'",
        "sphere radius=5 centre=100,0,0 'ball'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    fn shape(rows: &[crate::blocks::list::BlockRow], depth: usize, out: &mut Vec<String>) {
        for row in rows {
            out.push(format!("{depth} {:?}", row.block));
            shape(&row.inside, depth + 1, out);
        }
    }
    let rows = |app: &App| {
        let list = app.list_readings();
        let mut out = Vec::new();
        shape(&app.nested_blocks(&list.nesting), 0, &mut out);
        out
    };
    let before = rows(&app);
    app.run_command("comb curve=a");
    app.run_command("comb curve=bend");
    frame(&mut app, Vec::new());
    let after = rows(&app);
    let without: Vec<String> = after.iter().filter(|r| !r.contains("Comb(")).cloned().collect();
    assert_eq!(without, before, "every other row where it was, nested as it was");
    // Each comb is the row after the one that holds its curve: the slab's profile is inside the
    // slab, so its comb is below the slab; the bend's is below the bend.
    let slab = format!("0 {:?}", crate::blocks::list::Block::Node(app.node_named("slab").unwrap()));
    let slab_at = after.iter().position(|r| *r == slab).expect("the slab's row");
    let next_top = after.iter().skip(slab_at + 1).position(|r| r.starts_with("0 ")).map(|i| i + slab_at + 1);
    assert!(next_top.is_some_and(|i| after[i].contains("Comb(")), "the profile's comb is the next top-level row after the slab: {after:?}");
    let at = |needle: &str| after.iter().position(|r| r.contains(needle)).unwrap_or_else(|| panic!("{needle} in {after:?}"));
    let bend = app.curve_named("bend").unwrap();
    let combs: Vec<usize> = after.iter().enumerate().filter(|(_, r)| r.contains("Comb(")).map(|(i, _)| i).collect();
    assert_eq!(combs.len(), 2);
    assert!(combs.contains(&(at(&format!("Curve({bend:?})")) + 1)), "the bend's comb is directly below it: {after:?}");
}

/// **Deleting the curve deletes its comb in the same step**, and undo brings both back.
#[test]
fn deleting_the_curve_deletes_its_comb_in_one_undo() {
    let Some(mut app) = app_with_a_bend() else { return };
    app.run_command("comb curve=bend");
    let bend = app.curve_named("bend").unwrap();
    app.delete_curve(bend);
    assert!(app.doc.combs.is_empty() && app.doc.curve(bend).is_none(), "both gone");
    app.run_command("undo");
    assert!(app.doc.curve(bend).is_some() && app.doc.combs.len() == 1, "both back in one undo");
}

/// **A hidden curve hides its comb**, and the comb's own eye hides it on a shown curve.
#[test]
fn a_hidden_curve_hides_its_comb() {
    let Some(mut app) = app_with_a_bend() else { return };
    app.run_command("comb curve=bend teeth=10");
    let drawn = app.comb_readings();
    assert_eq!(drawn.len(), 1);
    assert_eq!(drawn[0].1.teeth.len(), 10, "ten teeth on a single-span cubic");
    let bend = app.curve_named("bend").unwrap();
    app.hidden_curves.insert(bend);
    assert!(app.comb_readings().is_empty(), "the curve hidden, so is its comb");
    app.hidden_curves.remove(&bend);
    app.run_command("comb block='Curvature Comb' hidden=true");
    assert!(app.comb_readings().is_empty(), "its own eye: {}", last_said(&app));
    app.run_command("undo");
    assert_eq!(app.comb_readings().len(), 1);
}

/// **Every field of a comb is a key of `comb`**, or recorded as one it does not read, with why.
#[test]
fn every_field_of_a_comb_is_a_key_comb_reads_or_a_recorded_gap() {
    use std::collections::BTreeSet;
    let curve = fieldform_scene::Curve::polyline(
        CurveId::new(),
        "line",
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
        false,
    )
    .unwrap();
    let mut comb = fieldform_scene::CurvatureComb::new(fieldform_scene::PartId::new(), &curve);
    comb.name = Some("n".into());
    comb.whole_loop = true;
    comb.flip = true;
    comb.hidden = true;
    let json = serde_json::to_value(&comb).unwrap();
    let fields: BTreeSet<String> = json.as_object().unwrap().keys().cloned().collect();
    let verb = console_keys::verb("evaluate.comb").expect("comb declares its keys");
    let console_keys::Keys::Declared { reads, not_read } = &verb.keys else { panic!("declared") };
    let covered: BTreeSet<&str> = reads
        .iter()
        .flat_map(|k| k.covers.iter().copied())
        .chain(not_read.iter().map(|(f, _)| *f))
        .chain(["name"])
        .collect();
    for field in &fields {
        assert!(covered.contains(field.as_str()), "a comb's field {field:?} is reached by no `comb` key and not recorded as unread");
    }
}

/// **Open curves become one open profile only where a continuity is set at their join** (the
/// owner, 2026-11-11): touching is two curves, G0, G1 or G2 at the join makes them one row, a
/// third curve joins when its own join is set, and breaking a join takes the chain apart there.
#[test]
fn touching_open_curves_group_only_once_a_continuity_is_set_at_their_join() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=20,0,0 'a'",
        "curve p=20,0,0;30,10,0;40,10,0;50,0,0 'b'",
        "line p0=50,0,0 p1=70,-10,0 'c'",
    ] {
        app.run_command(line);
    }
    let id = |app: &App, n: &str| app.curve_named(n).unwrap();
    let (a, b, c) = (id(&app, "a"), id(&app, "b"), id(&app, "c"));
    app.refresh_profiles();
    assert!(app.profiles.chains.is_empty() && app.profiles.loops.is_empty(), "touching alone groups nothing");

    app.run_command("continuity at=a.end g=0");
    app.refresh_profiles();
    assert_eq!(app.profiles.chains, vec![(a, vec![a, b])], "{}", last_said(&app));
    assert!(app.profiles.is_open(a));
    assert_eq!(app.profile_label(a), "Open profile 1");
    assert_eq!(app.profiles.key_of(b), Some(a));

    app.run_command("continuity at=b.end g=0");
    app.refresh_profiles();
    assert_eq!(app.profiles.chains, vec![(a, vec![a, b, c])], "the third joins when its join is set");

    app.run_command("continuity at=a.end -break");
    app.refresh_profiles();
    assert_eq!(app.profiles.chains, vec![(b, vec![b, c])], "broken at a: what is left is keyed by its own first");
    assert_eq!(app.profiles.key_of(a), None);
}

/// **An open profile is a row**, folding its curves inside it as a closed one does, and a
/// closed loop stays a Profile and is never an open one.
#[test]
fn an_open_profile_is_one_row_and_a_closed_loop_is_never_open() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=20,0,0 'a'",
        "line p0=20,0,0 p1=20,20,0 'b'",
        "continuity at=a.end g=0",
        "line p0=100,0,0 p1=140,0,0 's1'",
        "line p0=140,0,0 p1=140,20,0 's2'",
        "line p0=140,20,0 p1=100,20,0 's3'",
        "line p0=100,20,0 p1=100,0,0 's4'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    app.refresh_profiles();
    let a = app.curve_named("a").unwrap();
    let s1 = app.curve_named("s1").unwrap();
    assert_eq!(app.profiles.loops.len(), 1);
    assert!(!app.profiles.is_open(s1) && app.profile_label(s1) == "Profile 1", "the square is closed");
    assert_eq!(app.profile_label(a), "Open profile 1", "counted on its own");
    let list = app.list_readings();
    let rows = app.nested_blocks(&list.nesting);
    let top: Vec<String> = rows.iter().map(|r| format!("{:?}", r.block)).collect();
    assert!(top.contains(&format!("{:?}", crate::blocks::list::Block::Profile(a))), "{top:?}");
    let b = app.curve_named("b").unwrap();
    assert!(!top.contains(&format!("{:?}", crate::blocks::list::Block::Curve(b))), "b is drawn inside the open profile: {top:?}");
}

/// **A comb on a whole open profile reads every curve as one**, and says at each joint the turn
/// and the curvature step: an exact quarter arc into the line tangent to it reads no turn and a
/// step of 1/r. Asked for by name, or by selecting the profile's row and making a comb.
#[test]
fn a_comb_on_an_open_profile_reads_every_curve_and_its_joints() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "curve2 p=25,0,0;25,25,0;0,25,0 w=1,0.70710678118654752,1 'arc'",
        "line p0=0,25,0 p1=-30,25,0 'run'",
        "continuity at=arc.end g=0",
        "comb profile='Open profile 1' 'whole'",
    ] {
        app.run_command(line);
    }
    let comb = the_comb(&app);
    assert!(comb.whole_loop, "{}", last_said(&app));
    let drawn = app.comb_readings();
    assert_eq!(drawn.len(), 1);
    let reading = &drawn[0].1;
    assert_eq!(reading.teeth.len(), 2 * comb.teeth as usize, "both curves read");
    assert_eq!(reading.joints.len(), 1);
    let joint = reading.joints[0];
    assert!(joint.angle_degrees < 1e-4, "tangent: {}", joint.angle_degrees);
    assert!((joint.curvature_step + 1.0 / 25.0).abs() < 1e-6, "steps from 1/r to nought: {}", joint.curvature_step);

    // Selecting the row and asking is the same request.
    app.run_command("delete");
    let key = app.profiles.key_of(app.curve_named("arc").unwrap()).unwrap();
    app.select_block(outputs::BlockRef::Profile(key));
    app.run_command("comb");
    assert!(app.doc.combs.iter().any(|c| c.whole_loop && c.curve == key), "{}", last_said(&app));

    // Broken apart, it reads its curve alone and says so; whole again, all of it.
    app.run_command("continuity at=arc.end -break");
    assert_eq!(app.comb_readings().last().unwrap().1.joints.len(), 0);
    app.run_command("continuity at=arc.end g=0");
    assert_eq!(app.comb_readings().last().unwrap().1.joints.len(), 1);
}

/// **A wall made from an open profile runs as far as its row and no further** (the owner,
/// 2026-11-11): two lines held at their join and a third that only touches the second make a row
/// of two, and the wall's chain is those two. Setting the second join brings the third into both.
#[test]
fn a_wall_from_an_open_profile_stops_where_its_row_does() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        "line p0=0,0,0 p1=20,0,0 'a'",
        "line p0=20,0,0 p1=20,20,0 'b'",
        "line p0=20,20,0 p1=40,20,0 'c'",
        "continuity at=a.end g=0",
        "extrude curve=a wall=2 height=10 'fence'",
    ] {
        app.run_command(line);
    }
    let id = |app: &App, n: &str| app.curve_named(n).unwrap();
    let (a, b, c) = (id(&app, "a"), id(&app, "b"), id(&app, "c"));
    app.refresh_profiles();
    let row = app.profiles.members(app.profiles.key_of(a).unwrap()).unwrap().to_vec();
    assert_eq!(row, vec![a, b]);
    let wall = fieldform_field::outline_chain(&app.doc, a, None).unwrap();
    assert_eq!(wall.curves, row, "the wall is the row, not the touching third: {}", last_said(&app));

    app.run_command("continuity at=b.end g=0");
    app.refresh_profiles();
    let row = app.profiles.members(app.profiles.key_of(a).unwrap()).unwrap().to_vec();
    assert_eq!(row, vec![a, b, c]);
    assert_eq!(fieldform_field::outline_chain(&app.doc, a, None).unwrap().curves, row);
}
