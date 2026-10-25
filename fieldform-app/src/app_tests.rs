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

    // A command that is not on offer says why and does nothing: Flip section with no section.
    assert!(app.sections.is_empty(), "no section until one is asked for");
    app.invoke_command(by("view.section_flip"));
    assert!(app.sections.is_empty(), "refused while there is no section");
    app.invoke_command(by("view.section"));
    assert_eq!(app.sections.len(), 1, "the first toggle makes one");
    app.invoke_command(by("view.section_flip"));
    assert!(app.sections[0].flipped);
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
    assert!(app.sections[0].on && app.section_selected == Some(0), "turning it on reaches for it");
    let placed = app.sections[0];

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
    assert!(app.sections[0].on, "picking leaves the section on");
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
    assert!(app.sections[0].on, "the cut is still on after the edit");
    assert_eq!(app.sections[0].origin, placed.origin, "and the plane did not move with the face");
    assert_eq!(app.sections[0].rotation, placed.rotation);
    assert_eq!(app.sections[0].flipped, placed.flipped);

    // The row gives the handles back and moves nothing; the face is no longer picked.
    app.select_section(0);
    assert!(app.sections[0].on && app.section_selected == Some(0));
    assert!(app.selection.is_empty());
    assert_eq!(app.sections[0].origin, placed.origin);
    // And the switch is the only thing that turns it off: the row stays, with nothing to move.
    app.invoke_command(by("view.section"));
    assert!(!app.sections[0].on && app.selected_section().is_none());
    assert!(app.section_half_spaces().is_empty(), "nothing is cut while none is on");
    app.invoke_command(by("view.section"));
    assert!(app.sections[0].on, "the toggle switches the current one back on, it makes no second");
    assert_eq!(app.sections.len(), 1);

    // A second section: on beside the first, numbered after it, and the manipulator's; both cut.
    app.invoke_command(by("view.section_new"));
    assert_eq!(app.sections.len(), 2);
    assert!(app.sections[0].on && app.sections[1].on, "several may be on at once");
    assert_eq!(app.section_half_spaces().len(), 2);
    assert_eq!((app.sections[0].number, app.sections[1].number), (1, 2));
    assert_eq!(app.section_selected, Some(1));
    // The first's row takes the handles and switches nothing off; its own switch switches it off.
    app.select_section(0);
    assert!(app.sections[0].on && app.sections[1].on && app.section_selected == Some(0));
    assert!(app.set_section_on(0, false));
    assert!(!app.sections[0].on && app.sections[1].on);
    assert_eq!(app.section_half_spaces().len(), 1);
    // Delete, with a section's row selected, removes that section and nothing of the part; the
    // one behind it keeps its number, and the claim goes with the row.
    let bodies = app.part_graph().roots().len();
    app.invoke_command(by("edit.delete_body"));
    assert_eq!(app.sections.len(), 1);
    assert_eq!(app.sections[0].number, 2);
    assert!(app.section_selected.is_none());
    assert_eq!(app.part_graph().roots().len(), bodies, "the part is untouched");
    app.remove_section(0);
    assert!(app.sections.is_empty() && app.section_half_spaces().is_empty());
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
    edit(&mut app, "Boolean_0", |n| {
        if let Node::Boolean { op, inputs } = n.clone()
            && let [a, b] = inputs[..]
        {
            *n = Node::Blend {
                op,
                a,
                b,
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
    assert!(only.contains(&"create.sphere") && !only.contains(&"view.section_flip"), "Flip section is not on offer");

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
        Node::Boolean { inputs, .. } => inputs.clone(),
        Node::Blend { a, b, .. } => vec![*a, *b],
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
        Node::Boolean { inputs, .. } => inputs[1] = ball.into(),
        Node::Blend { b, .. } => *b = ball.into(),
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
