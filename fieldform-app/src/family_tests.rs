// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Block families and their frames (backlog item 135): the table is the code, a family is a
//! function of kind alone, both views agree, and every row the list draws wears a frame inside
//! its holder's.
//!
//! The drawing half is held through the frames the list records under test
//! (`App::drawn_frames`), so a window is not needed to ask whether one frame encloses another —
//! which is the hand check the notes ask for, made a test.

use std::collections::{BTreeMap, BTreeSet};

use fieldform_scene::{BlockKey, CurveId, Family, Node, NodeId, ParamId};

use crate::app_tests::{frame, headless_app};
use crate::blocks::list::{Block, DrawnFrame};
use crate::console_keys::tests::samples;
use crate::theme::{family_colour, frame_paint};
use crate::App;
use eframe::egui;
use eframe::App as _;

/// A node's kind as the code spells it: the variant's name, read off its `Debug`.
fn kind_of(node: &Node) -> String {
    format!("{node:?}").chars().take_while(|c| c.is_alphanumeric()).collect()
}

/// **The table is the code**: one node of every kind, each in the family the notes' table puts
/// it in, written here as data. Every kind in the table must be met by a sample, so a sample left
/// out is caught as surely as a kind filed wrong; and the match in `Node::family` has no wildcard,
/// so a new kind does not compile until it is filed at all.
#[test]
fn every_kind_of_node_is_in_the_family_the_table_says() {
    use Family::*;
    let table: BTreeMap<&str, Option<Family>> = [
        ("Boolean", Some(Combine)),
        ("Blend", Some(Combine)),
        ("Morph", Some(Combine)),
        ("Sphere", Some(Create)),
        ("Cuboid", Some(Create)),
        ("SkewedBox", Some(Create)),
        ("Cylinder", Some(Create)),
        ("Cone", Some(Create)),
        ("Torus", Some(Create)),
        ("Capsule", Some(Create)),
        ("Coil", Some(Create)),
        ("Thread", Some(Create)),
        ("Hole", Some(Create)),
        ("Extrude", Some(Create)),
        ("Revolve", Some(Create)),
        ("Sweep", Some(Create)),
        ("Loft", Some(Create)),
        ("Pipe", Some(Create)),
        ("Lattice", Some(Create)),
        // Not in the notes' table, which was written the day this block arrived: it reads no
        // body and makes one from a formula, as a sphere makes one from a radius.
        ("Expression", Some(Create)),
        ("Offset", Some(Modify)),
        ("Shell", Some(Modify)),
        ("Twist", Some(Modify)),
        ("Attract", Some(Modify)),
        ("Transform", Some(Modify)),
        ("Mirror", Some(Modify)),
        ("Scale", Some(Modify)),
        ("Pattern", Some(Modify)),
        ("Infill", Some(Modify)),
        ("Cage", Some(Cage)),
        ("Mesh", Some(File)),
        ("Plane", Some(Reference)),
        ("Point", Some(Reference)),
        ("Ramp", Some(Reference)),
        ("Unknown", None),
    ]
    .into_iter()
    .collect();

    let mut nodes = samples();
    // The kinds no console verb makes, which the key table's samples therefore leave out.
    nodes.push(Node::Cage(fieldform_scene::CageId::new()));
    nodes.push(Node::Mesh(fieldform_scene::MeshId::new()));
    let unknown: Node = serde_json::from_value(serde_json::json!({ "kind": "thicken_of_the_future" }))
        .expect("a kind this build cannot read still loads, as Unknown");
    nodes.push(unknown);

    let mut met = BTreeSet::new();
    for node in &nodes {
        let kind = kind_of(node);
        let wanted = table.get(kind.as_str()).unwrap_or_else(|| panic!("{kind} is not in the table"));
        assert_eq!(node.family(), *wanted, "{kind}");
        met.insert(kind);
    }
    let missing: Vec<_> = table.keys().filter(|k| !met.contains(**k)).collect();
    assert!(missing.is_empty(), "no sample of {missing:?}");
}

/// The rows the graph does not hold, filed by `Block::family`: a named number is Reference, a
/// curve and a profile are Curve, and a surface, an import and the three outputs are File.
#[test]
fn every_kind_of_row_the_graph_does_not_hold_is_in_its_family() {
    let graph = fieldform_scene::Graph::empty();
    let rows = [
        (Block::Parameter(ParamId::new()), Family::Reference),
        (Block::Curve(CurveId::new()), Family::Curve),
        (Block::Profile(CurveId::new()), Family::Curve),
        (Block::Surface(fieldform_scene::QuadMeshId::new()), Family::File),
        (Block::Import(fieldform_scene::id::SourceId::new()), Family::File),
        (Block::Tessellation(fieldform_scene::TessellationId::new()), Family::File),
        (Block::Export(fieldform_scene::ExportId::new()), Family::File),
        (Block::Image(fieldform_scene::SaveImageId::new()), Family::File),
    ];
    for (block, family) in rows {
        assert_eq!(block.family(&graph), Some(family), "{block:?}");
    }
    assert_eq!(Block::Node(NodeId::new()).family(&graph), None, "a block no longer in the graph has none");
}

/// The legend there is: the hover line on each bar is the family and its rule.
#[test]
fn the_bar_says_the_family_and_its_rule() {
    assert_eq!(Family::Modify.hover_line(), "Modify \u{2014} reads one body and makes one");
    assert_eq!(Family::Combine.hover_line(), "Combine \u{2014} reads two or more bodies and makes one");
    for family in Family::ALL {
        let line = family.hover_line();
        assert!(line.starts_with(family.label()) && line.contains(family.rule()), "{line}");
    }
}

/// Opens every row above `name`'s by selecting it — the list follows a selection it did not
/// make, a level a frame or two at a time — and then draws one frame with the record cleared.
fn draw_with_open(app: &mut App, name: &str) {
    let id = app.node_named(name).unwrap_or_else(|| panic!("no block called {name}"));
    app.select_body(Some(id));
    for _ in 0..40 {
        tick(app);
    }
    app.drawn_frames.clear();
    tick(app);
}

thread_local! {
    /// Whether this test's frames hand their textures to the renderer, which only a test that
    /// draws a picture needs: the harness's `frame` drops a frame's output, font atlas and all.
    static UPLOADING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One frame, as the harness draws it or — where this test draws pictures — with its textures
/// handed to the renderer, from the first frame on, so a later picture finds the font atlas.
fn tick(app: &mut App) {
    if UPLOADING.with(std::cell::Cell::get) {
        let _ = run_frame(app, (1400, 1600));
    } else {
        frame(app, Vec::new());
    }
}

fn frame_of<'a>(frames: &'a [DrawnFrame], app: &App, name: &str) -> (usize, &'a DrawnFrame) {
    let id = app.node_named(name).unwrap_or_else(|| panic!("no block called {name}"));
    frames
        .iter()
        .enumerate()
        .find(|(_, f)| f.block == Block::Node(id))
        .unwrap_or_else(|| panic!("{name}'s row drew no frame"))
}

/// A part to open: a union of two shells, each of a sphere, and a plane beside them.
fn nested_part(app: &mut App) {
    tick(app);
    for line in [
        "sphere radius=5 'a'",
        "sphere radius=5 centre=20,0,0 'b'",
        "shell target=a thickness=1 'sa'",
        "shell target=b thickness=1 'sb'",
        "boolean -u in=sa,sb 'u'",
        "plane 'p'",
    ] {
        app.run_command(line);
    }
    tick(app);
}

/// **The hand check, as a test** (the notes' §8 step 9): every row the list draws has a frame and
/// a bar; a boolean's frame encloses its operands' and theirs their sphere's, each inset inside its
/// holder's and none touching; frames side by side do not overlap; and each is painted in its
/// family — Combine round Modify round Create, concentric, and the plane dashed.
#[test]
fn every_row_is_framed_inside_its_holders_frame_in_its_family() {
    let Some(mut app) = headless_app() else { return };
    nested_part(&mut app);
    draw_with_open(&mut app, "a");
    let frames = app.drawn_frames.clone();
    assert!(!frames.is_empty(), "the list drew no frames at all");
    let dark = app.egui_ctx.global_style().visuals.dark_mode;

    let (u_at, u) = frame_of(&frames, &app, "u");
    let (sa_at, sa) = frame_of(&frames, &app, "sa");
    let (_, a) = frame_of(&frames, &app, "a");
    assert_eq!(sa.parent, Some(u_at), "a shell's row is drawn inside the union's");
    assert_eq!(a.parent, Some(sa_at), "the sphere's row is drawn inside its shell's");
    let inset = crate::theme::FRAME_PADDING - 0.5;
    for (outer, inner, what) in [(u, sa, "union round shell"), (sa, a, "shell round sphere")] {
        assert!(outer.rect.shrink(inset).contains_rect(inner.rect), "{what}: {:?} in {:?}", inner.rect, outer.rect);
    }
    for (f, family) in [(u, Family::Combine), (sa, Family::Modify), (a, Family::Create)] {
        assert_eq!(f.paint.stroke.color, family_colour(family, dark), "{:?}'s stroke", f.block);
        assert_eq!(f.paint.bar.map(|b| b.0), Some(family_colour(family, dark)), "{:?}'s bar", f.block);
        assert!(!f.paint.dashed);
    }
    let (_, p) = frame_of(&frames, &app, "p");
    assert!(p.paint.dashed && p.paint.bar.is_some_and(|(_, hollow)| hollow), "a plane is Reference: dashed, hollow bar");

    // Every frame has a bar; frames drawn inside the same holder never overlap.
    for (i, f) in frames.iter().enumerate() {
        assert!(f.paint.bar.is_some(), "frame {i} ({:?}) has no bar", f.block);
        assert!(f.rect.is_positive(), "frame {i} ({:?}) was never given a rectangle", f.block);
        for (j, g) in frames.iter().enumerate().skip(i + 1) {
            if f.parent == g.parent {
                assert!(!f.rect.intersects(g.rect.shrink(0.5)), "frames {i} and {j} overlap: {:?} and {:?}", f.block, g.block);
            }
        }
        if let Some(parent) = f.parent {
            assert!(frames[parent].rect.contains_rect(f.rect), "frame {i} spills out of its holder");
        }
    }
}

/// **A family is a function of kind alone**: selecting, renaming, declaring and hiding a block
/// leave every row's bar the colour it was — a hidden body's dimmed with it, and still its family.
#[test]
fn selection_renaming_declaring_and_hiding_never_change_a_rows_family() {
    let Some(mut app) = headless_app() else { return };
    nested_part(&mut app);
    draw_with_open(&mut app, "a");
    let dark = app.egui_ctx.global_style().visuals.dark_mode;
    let bars = |app: &App| -> BTreeMap<String, Option<egui::Color32>> {
        app.drawn_frames
            .iter()
            .map(|f| (format!("{:?}", f.block), f.paint.bar.map(|b| b.0)))
            .collect()
    };
    let before = bars(&app);

    let a = app.node_named("a").unwrap();
    let u = app.node_named("u").unwrap();
    let rename = crate::blocks::list::rename_block(&app.doc, a, "renamed").expect("a block may be renamed");
    assert!(app.set_ownership(rename));
    let declare = crate::blocks::list::make_variable(&app.doc, BlockKey::Node(app.node_named("b").unwrap()), || "bee".into())
        .expect("b is not a variable yet");
    assert!(app.set_ownership(declare));
    draw_with_open(&mut app, "renamed");
    let after = bars(&app);
    for (block, bar) in &after {
        if let Some(was) = before.get(block) {
            assert_eq!(bar, was, "{block}'s bar changed with its state");
        }
    }

    // Hidden: the same family, at half strength.
    app.set_visible(u, false);
    draw_with_open(&mut app, "renamed");
    let (_, hidden) = frame_of(&app.drawn_frames, &app, "u");
    let combine = family_colour(Family::Combine, dark);
    assert_eq!(hidden.paint.bar.map(|b| b.0), Some(combine.gamma_multiply(crate::theme::DIMMED)));
}

/// **Status over family**: a shell with nothing to shell draws the error stroke round its frame
/// and keeps the Modify bar.
#[test]
fn a_block_with_an_empty_input_wears_the_error_stroke_and_keeps_its_bar() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    app.run_command("sphere radius=5 'ball'");
    app.run_command("shell target=none thickness=2 'hollow'");
    frame(&mut app, Vec::new());
    app.drawn_frames.clear();
    frame(&mut app, Vec::new());
    let visuals = app.egui_ctx.global_style().visuals.clone();
    let (_, hollow) = frame_of(&app.drawn_frames, &app, "hollow");
    assert_eq!(hollow.paint.stroke.color, visuals.error_fg_color);
    assert_eq!(hollow.paint.bar.map(|b| b.0), Some(family_colour(Family::Modify, visuals.dark_mode)));
    let (_, ball) = frame_of(&app.drawn_frames, &app, "ball");
    assert_eq!(ball.paint, frame_paint(Some(Family::Create), &crate::graph_view::model::Status::Ok, &visuals));
}

/// Holds every frame in `frames` to the rules the list draws by: a rectangle and a bar each (a
/// block this build cannot read is the one without); inside its holder's, inset by the holder's
/// padding so the two never touch; and no two frames in one holder overlapping.
fn assert_frames_nest(frames: &[DrawnFrame]) {
    let inset = crate::theme::FRAME_PADDING - 0.5;
    for (i, f) in frames.iter().enumerate() {
        assert!(f.rect.is_positive(), "frame {i} ({:?}) was never given a rectangle", f.block);
        assert!(f.paint.bar.is_some(), "frame {i} ({:?}) has no bar", f.block);
        if let Some(parent) = f.parent {
            let holder = frames[parent].rect;
            assert!(holder.shrink(inset).contains_rect(f.rect), "frame {i} ({:?}) is not inset in its holder's", f.block);
        }
        for (j, g) in frames.iter().enumerate().skip(i + 1) {
            if f.parent == g.parent {
                assert!(!f.rect.intersects(g.rect.shrink(0.5)), "frames {i} and {j} overlap: {:?} and {:?}", f.block, g.block);
            }
        }
    }
}

/// **The hand check on the deepest reference model**: FTC-07, an enclosure that is booleans of
/// booleans of planes, replayed line by line (its closing save left out, so the run writes nothing),
/// and the list opened down to its deepest block. Every frame sits inside its holder's and none
/// overlaps another — five levels and more of concentric frames.
#[test]
fn the_deepest_reference_model_draws_every_frame_inside_its_holders() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    let script = include_str!("../../benchmarks/reference-cad/ftc-07.ffc");
    for line in script.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line.starts_with("saveas") {
            continue;
        }
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    // The block with the longest chain of holders above it.
    let list = app.list_readings();
    let depth = |mut id: NodeId| {
        let mut n = 0;
        while let Some(&owner) = list.nesting.owners_of(id).first() {
            id = owner;
            n += 1;
        }
        n
    };
    let (deepest, levels) = app
        .part_graph()
        .nodes()
        .map(|(id, _)| (id, depth(id)))
        .max_by_key(|(id, d)| (*d, std::cmp::Reverse(id.raw())))
        .expect("the model has blocks");
    drop(list);
    assert!(levels >= 4, "FTC-07 nests only {levels} deep");
    app.select_body(Some(deepest));
    for _ in 0..(20 + 10 * levels) {
        frame(&mut app, Vec::new());
    }
    app.drawn_frames.clear();
    frame(&mut app, Vec::new());
    let frames = app.drawn_frames.clone();
    let (at, _) = frames
        .iter()
        .enumerate()
        .find(|(_, f)| f.block == Block::Node(deepest))
        .expect("the deepest block's row was opened and drawn");
    let mut chain = 0;
    let mut up = frames[at].parent;
    while let Some(p) = up {
        chain += 1;
        up = frames[p].parent;
    }
    assert_eq!(chain, levels, "its frame is drawn inside one frame per holder");
    assert_frames_nest(&frames);
}

/// One frame of the whole interface, `size` points at one pixel a point, its textures handed to
/// the app's renderer, and its shapes tessellated for drawing.
fn run_frame(app: &mut App, size: (u32, u32)) -> (Vec<egui::ClippedPrimitive>, f32) {
    let ctx = app.egui_ctx.clone();
    let mut input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size.0 as f32, size.1 as f32))),
        ..Default::default()
    };
    app.raw_input_hook(&ctx, &mut input);
    let mut eframe_frame = eframe::Frame::_new_kittest();
    let out = ctx.run_ui(input, |ui| app.ui(ui, &mut eframe_frame));
    let ppp = out.pixels_per_point;
    let jobs = ctx.tessellate(out.shapes, ppp);
    let state = app.wgpu_render_state.clone();
    let mut renderer = state.renderer.write();
    for (id, delta) in &out.textures_delta.set {
        renderer.update_texture(&state.device, &state.queue, *id, delta);
    }
    for id in &out.textures_delta.free {
        renderer.free_texture(id);
    }
    (jobs, ppp)
}

/// One frame of the whole interface, drawn by egui's own wgpu renderer — the app's, so the
/// viewport's paint callbacks find their resources — into a texture and read back as RGBA.
fn interface_image(app: &mut App, size: (u32, u32)) -> image::RgbaImage {
    use eframe::egui_wgpu::{wgpu, ScreenDescriptor};
    let (jobs, ppp) = run_frame(app, size);
    let state = app.wgpu_render_state.clone();
    let (device, queue) = (&state.device, &state.queue);
    let mut renderer = state.renderer.write();
    let format = state.target_format;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("interface image"),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let screen = ScreenDescriptor { size_in_pixels: [size.0, size.1], pixels_per_point: ppp };
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut commands = renderer.update_buffers(device, queue, &mut encoder, &jobs, &screen);
    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("interface image"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
    }
    let row = (size.0 * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("interface image readback"),
        size: u64::from(row * size.1),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.1) },
        },
        wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
    );
    commands.push(encoder.finish());
    queue.submit(commands);
    drop(renderer);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("the device finishes");
    let bytes = slice.get_mapped_range();
    let bgra = matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut picture = image::RgbaImage::new(size.0, size.1);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let at = (y * row + x * 4) as usize;
            let p = &bytes[at..at + 4];
            let rgba = if bgra { [p[2], p[1], p[0], 255] } else { [p[0], p[1], p[2], 255] };
            picture.put_pixel(x, y, image::Rgba(rgba));
        }
    }
    picture
}

/// **The list as it is drawn**, in both themes, written where `FIELDFORM_LIST_IMAGE` names a
/// folder — `notebook-dark.png` and `notebook-light.png`, the panel cropped from the window. The
/// hand check's picture: the frames the tests above measure, seen. Skipped when the variable is
/// unset, so an ordinary run writes nothing.
#[test]
fn the_list_drawn_to_an_image_when_asked() {
    let Some(out) = std::env::var_os("FIELDFORM_LIST_IMAGE").map(std::path::PathBuf::from) else { return };
    let Some(mut app) = headless_app() else { return };
    UPLOADING.with(|u| u.set(true));
    const SIZE: (u32, u32) = (1400, 1600);
    nested_part(&mut app);
    for line in [
        "param name=reach value=11",
        "line p0=0,0,0 p1=40,0,0 'c1'",
        "line p0=40,0,0 p1=40,20,0 'c2'",
        "line p0=40,20,0 p1=0,20,0 'c3'",
        "line p0=0,20,0 p1=0,0,0 'c4'",
        "extrude profile=c1 height=10 'slab'",
        "cube size=reach 'box'",
        "shell target=none thickness=2 'hollow'",
    ] {
        app.run_command(line);
    }
    // A variable read by a field, so the union's second slot is drawn as a chip.
    let sb = app.node_named("sb").expect("sb");
    let declare = crate::blocks::list::make_variable(&app.doc, BlockKey::Node(sb), || "sb".into()).expect("sb is not a variable");
    assert!(app.set_ownership(declare));
    std::fs::create_dir_all(&out).expect("the output folder");
    for (theme, name) in [(egui::Theme::Dark, "notebook-dark.png"), (egui::Theme::Light, "notebook-light.png")] {
        app.egui_ctx.set_theme(theme);
        draw_with_open(&mut app, "a");
        let _ = interface_image(&mut app, SIZE);
        let picture = interface_image(&mut app, SIZE);
        // The list's own extent: the frames it drew, with a margin.
        let extent = app.drawn_frames.iter().fold(egui::Rect::NOTHING, |r, f| r.union(f.rect)).expand(12.0);
        let (x, y) = (extent.left().max(0.0) as u32, extent.top().max(0.0) as u32);
        let w = (extent.width() as u32).min(SIZE.0 - x);
        let h = (extent.height() as u32).min(SIZE.1 - y);
        image::imageops::crop_imm(&picture, x, y, w, h)
            .to_image()
            .save(out.join(name))
            .expect("the picture is written");
        picture.save(out.join(format!("window-{name}"))).expect("the window is written");
    }
    // And the graph view's cards, which wear the same bar and the same stroke rule.
    app.egui_ctx.set_theme(egui::Theme::Dark);
    app.run_command("view show=graph");
    app.run_command("frameall");
    for _ in 0..10 {
        tick(&mut app);
    }
    let _ = interface_image(&mut app, SIZE);
    interface_image(&mut app, SIZE).save(out.join("graph-dark.png")).expect("the graph is written");
}

/// **Both views agree**: every node the graph view draws carries the family the notebook's row
/// for the same block is framed in.
#[test]
fn the_graph_and_the_notebook_agree_on_every_blocks_family() {
    use crate::graph_view::model::Kind;
    let Some(mut app) = headless_app() else { return };
    nested_part(&mut app);
    app.run_command("param name=reach value=11");
    for line in [
        "line p0=0,0,0 p1=40,0,0 'c1'",
        "line p0=40,0,0 p1=40,20,0 'c2'",
        "line p0=40,20,0 p1=0,20,0 'c3'",
        "line p0=0,20,0 p1=0,0,0 'c4'",
        "extrude profile=c1 height=10 'slab'",
    ] {
        app.run_command(line);
    }
    frame(&mut app, Vec::new());
    let (model, _) = app.graph_model();
    let graph = app.part_graph().clone();
    let mut checked = BTreeSet::new();
    for n in &model.nodes {
        let row = match n.kind {
            Kind::Block => Block::Node(n.id),
            Kind::Curve(c) => Block::Curve(c),
            Kind::Profile(c) => Block::Profile(c),
            Kind::Number => {
                let id = app.doc.parameter_named(&n.label).expect("a number node is a named number");
                Block::Parameter(id)
            }
            // A surface node stands for a quad mesh whose id the node does not carry; its family
            // is asked the same way, so the row's answer is the one for any surface.
            Kind::Surface => Block::Surface(fieldform_scene::QuadMeshId::new()),
        };
        assert_eq!(n.family, row.family(&graph), "{} ({:?})", n.label, n.kind);
        checked.insert(format!("{:?}", n.family));
    }
    for family in ["Some(Combine)", "Some(Modify)", "Some(Create)", "Some(Curve)", "Some(Reference)"] {
        assert!(checked.contains(family), "no node of {family} was compared");
    }
}
