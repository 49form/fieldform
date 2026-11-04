// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Save Image block and `snap`, driven through a real `App` without a window (backlog item
//! 117): the five properties the design marks as silent -- a picture that looks fine and is
//! wrong -- and the console's own behaviours. A machine with no adapter skips these and says so.

use super::*;
use crate::app_tests::{frame, headless_app};
use std::path::{Path, PathBuf};

/// A folder of this test's own, empty.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fieldform-image-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs frames until no picture is pending, or the wait is clearly a hang.
fn finish(app: &mut App) {
    let started = std::time::Instant::now();
    while app.pending_snap.is_some() {
        assert!(started.elapsed() < std::time::Duration::from_secs(180), "the picture never finished");
        frame(app, Vec::new());
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    frame(app, Vec::new());
}

/// Types `line`, drives it to its end, and answers what the console said after it.
fn snap(app: &mut App, line: &str) -> Vec<(bool, String)> {
    let before = app.console_log.len();
    app.run_command(line);
    finish(app);
    app.console_log.iter().skip(before).map(|l| (l.ok, l.text.clone())).collect()
}

fn wrote(said: &[(bool, String)]) -> bool {
    said.iter().any(|(ok, t)| *ok && t.starts_with("Wrote "))
}

fn load(path: &Path) -> image::RgbaImage {
    image::open(path).unwrap_or_else(|e| panic!("{} did not read: {e}", path.display())).to_rgba8()
}

fn path_arg(dir: &Path, name: &str) -> String {
    format!("\"{}\"", dir.join(name).to_string_lossy().replace('\\', "/"))
}

/// A sphere in an empty document, with the viewport settled on it.
fn sphere_app() -> Option<App> {
    let mut app = headless_app()?;
    frame(&mut app, Vec::new());
    app.run_command("sphere radius=10 'ball'");
    // A rung that accumulates, so a silhouette has a fractional edge, and the sphere filling
    // the view, so every count below is about it and not about empty frame.
    app.run_command("display quality=balanced");
    app.run_command("frameall");
    for _ in 0..3 {
        frame(&mut app, Vec::new());
    }
    Some(app)
}

#[test]
fn a_snap_with_no_quality_is_the_viewports_own_picture_as_it_always_was() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("displayed");
    let said = snap(&mut app, &format!("snap file={}", path_arg(&dir, "ref.png")));
    assert!(wrote(&said), "{said:?}");
    let img = load(&dir.join("ref.png"));
    assert_eq!((img.width(), img.height()), app.viewport_pixels(), "the window's size is its size");
    let refused = snap(&mut app, &format!("snap file={} size=100", path_arg(&dir, "x.png")));
    assert!(refused.iter().any(|(ok, t)| !ok && t.contains("size=")), "{refused:?}");
    assert!(!dir.join("x.png").exists());
    let unknown = snap(&mut app, &format!("snap file={} sharpness=9", path_arg(&dir, "y.png")));
    assert!(unknown.iter().any(|(ok, t)| !ok && t.contains("doesn't read")), "{unknown:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// T1 and T2: the same view at W×H and at 4W×4H frame the scene identically, and the larger
/// one has no holes inside the silhouette.
#[test]
fn a_still_frames_the_scene_as_the_viewport_does_at_any_size_and_has_no_holes() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("framing");
    let small = snap(&mut app, &format!("snap file={} quality=still width=300 height=200 bg=transparent", path_arg(&dir, "small.png")));
    assert!(wrote(&small), "{small:?}");
    let big = snap(&mut app, &format!("snap file={} quality=still width=1200 height=800 bg=transparent", path_arg(&dir, "big.png")));
    assert!(wrote(&big), "{big:?}");
    let (s, b) = (load(&dir.join("small.png")), load(&dir.join("big.png")));
    assert_eq!((s.width(), s.height(), b.width(), b.height()), (300, 200, 1200, 800));
    let covered = |img: &image::RgbaImage, x: u32, y: u32| img.get_pixel(x, y).0[3] >= 128;
    // Box-downsample the big one's coverage by four and hold it to the small one's, away from
    // the small one's own edges: a silhouette may land a pixel either way, a framing error
    // slides the whole thing.
    let mut disagreements = 0;
    let mut compared = 0;
    for y in 0..200u32 {
        for x in 0..300u32 {
            let near_edge = (y.saturating_sub(1)..=(y + 1).min(199))
                .any(|yy| (x.saturating_sub(1)..=(x + 1).min(299)).any(|xx| covered(&s, xx, yy) != covered(&s, x, y)));
            if near_edge {
                continue;
            }
            let mut sum = 0u32;
            for yy in 0..4 {
                for xx in 0..4 {
                    sum += u32::from(b.get_pixel(x * 4 + xx, y * 4 + yy).0[3]);
                }
            }
            compared += 1;
            if (sum >= 16 * 128) != covered(&s, x, y) {
                disagreements += 1;
            }
        }
    }
    assert!(compared > 10_000, "the sphere fills a fair part of the frame: {compared}");
    assert_eq!(disagreements, 0, "the 4x picture frames the scene as the 1x one does");
    // And inside the silhouette the big picture is solid: a ray that ran out of steps at the
    // finer size would leave a hole here.
    let mut holes = 0;
    for y in 2..198 {
        for x in 2..298 {
            let inside = (y - 2..=y + 2).all(|yy| (x - 2..=x + 2).all(|xx| covered(&s, xx, yy)));
            if !inside {
                continue;
            }
            for yy in 0..4 {
                for xx in 0..4 {
                    if b.get_pixel(x * 4 + xx, y * 4 + yy).0[3] != 255 {
                        holes += 1;
                    }
                }
            }
        }
    }
    assert_eq!(holes, 0, "no hole inside the silhouette at four times the size");
    let _ = std::fs::remove_dir_all(&dir);
}

/// T3: a picture drawn in tiles is the picture drawn whole, byte for byte.
#[test]
fn a_tiled_render_is_the_untiled_one_byte_for_byte() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("tiles");
    let line = |name: &str| format!("snap file={} quality=still width=640 height=480 bg=color:#303030", path_arg(&dir, name));
    let whole = snap(&mut app, &line("whole.png"));
    assert!(wrote(&whole), "{whole:?}");
    app.traced.limit_image_tile(Some(90_000));
    let tiled = snap(&mut app, &line("tiled.png"));
    assert!(wrote(&tiled), "{tiled:?}");
    assert!(tiled.iter().any(|(_, t)| t.contains("tiles")), "the line says it was tiled: {tiled:?}");
    app.traced.limit_image_tile(None);
    let (w, t) = (load(&dir.join("whole.png")), load(&dir.join("tiled.png")));
    let mut worst = 0u8;
    let mut differing = 0usize;
    for (a, b) in w.pixels().zip(t.pixels()) {
        let d = a.0.iter().zip(b.0.iter()).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        worst = worst.max(d);
        differing += usize::from(d > 0);
    }
    eprintln!("tiled against whole: {differing} pixels differ, worst {worst} LSB");
    assert_eq!(worst, 0, "a tiled render is the untiled one bit for bit ({differing} pixels differ)");
    let _ = std::fs::remove_dir_all(&dir);
}

/// T4: a transparent picture placed on black and on white matches the opaque renders over
/// those backgrounds -- no dark or light halo along the silhouette.
#[test]
fn a_transparent_image_composites_onto_black_and_white_without_a_halo() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("alpha");
    let line = |name: &str, bg: &str| format!("snap file={} quality=still width=400 height=300 ss=2 bg={bg}", path_arg(&dir, name));
    for (name, bg) in [("alpha.png", "transparent"), ("black.png", "color:#000000"), ("white.png", "color:#ffffff")] {
        let said = snap(&mut app, &line(name, bg));
        assert!(wrote(&said), "{said:?}");
    }
    let (a, black, white) = (load(&dir.join("alpha.png")), load(&dir.join("black.png")), load(&dir.join("white.png")));
    let mut worst = (0u8, 0u8);
    let mut edge_pixels = 0;
    let mut over_two = (0, 0);
    for y in 0..300 {
        for x in 0..400 {
            let p = a.get_pixel(x, y).0;
            let alpha = f32::from(p[3]) / 255.0;
            if p[3] == 0 || p[3] == 255 {
                continue;
            }
            edge_pixels += 1;
            for (c, &value) in p.iter().take(3).enumerate() {
                let on_black = (f32::from(value) * alpha).round() as u8;
                let on_white = (f32::from(value) * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                let db = on_black.abs_diff(black.get_pixel(x, y).0[c]);
                let dw = on_white.abs_diff(white.get_pixel(x, y).0[c]);
                worst = (worst.0.max(db), worst.1.max(dw));
                over_two = (over_two.0 + usize::from(db > 2), over_two.1 + usize::from(dw > 2));
            }
        }
    }
    eprintln!("edge pixels {edge_pixels}: worst over black {} LSB, over white {} LSB; over two LSB: {} and {} channel readings", worst.0, worst.1, over_two.0, over_two.1);
    assert!(edge_pixels > 200, "a sphere has an edge: {edge_pixels}");
    // A halo is a systematic darkening or lightening along the whole edge; a tonemap that is
    // not linear leaves an edge pixel a few values off, which is not that. The bound below is
    // the measured worst case with room, and the numbers above are what to re-read if it trips.
    assert!(worst.0 <= 24 && worst.1 <= 24, "an edge pixel is {} / {} LSB off its opaque render", worst.0, worst.1);
    // The interior and the outside are exact.
    for (x, y) in [(200, 150), (0, 0), (399, 299)] {
        let p = a.get_pixel(x, y).0;
        assert!(p[3] == 0 || p[3] == 255, "({x},{y}) is wholly in or out: {p:?}");
        if p[3] == 255 {
            assert_eq!(&p[..3], &black.get_pixel(x, y).0[..3], "a hit pixel is the same colour whatever is behind it");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// T5: a flat background is the bytes that were typed, at a width that is not a multiple of
/// sixty-four, with the rows where they belong.
#[test]
fn a_flat_background_is_the_bytes_typed_and_an_odd_width_is_not_sheared() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("flat");
    let said = snap(&mut app, &format!("snap file={} quality=still width=1001 height=301 bg=color:#204060 format=png", path_arg(&dir, "flat.png")));
    assert!(wrote(&said), "{said:?}");
    let img = load(&dir.join("flat.png"));
    assert_eq!((img.width(), img.height()), (1001, 301));
    let mut background = 0;
    let mut other = 0;
    for p in img.pixels() {
        if p.0 == [0x20, 0x40, 0x60, 255] {
            background += 1;
        } else {
            other += 1;
        }
    }
    assert!(background > 1001 * 301 / 2, "most of the frame is the background: {background} of {}", 1001 * 301);
    assert!(other > 1000, "and the sphere is in it: {other}");
    // Not sheared: the left and right columns are background on every row, and the sphere sits
    // about the middle of every row it crosses.
    for y in 0..301 {
        assert_eq!(img.get_pixel(0, y).0, [0x20, 0x40, 0x60, 255], "row {y} starts with the background");
        assert_eq!(img.get_pixel(1000, y).0, [0x20, 0x40, 0x60, 255], "row {y} ends with the background");
    }
    let (mut sum_x, mut count) = (0u64, 0u64);
    for (x, _, p) in img.enumerate_pixels() {
        if p.0 != [0x20, 0x40, 0x60, 255] {
            sum_x += u64::from(x);
            count += 1;
        }
    }
    let centre = sum_x as f64 / count as f64;
    assert!((centre - 500.0).abs() < 4.0, "the sphere is centred: {centre}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_block_is_made_from_a_line_runs_from_snap_block_and_numbers_its_files() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("block");
    let said = snap(&mut app, &format!("saveimage width=320 height=240 bg=transparent ss=2 folder={} 'hero'", path_arg(&dir, "")));
    assert!(said.iter().any(|(ok, t)| *ok && t.contains("Added hero")), "{said:?}");
    let block = app.doc.images.iter().find(|b| b.title() == "hero").cloned().expect("the block is in the document");
    assert_eq!((block.width, block.height, block.supersample), (320, 240, 2));
    assert_eq!(block.background, fieldform_scene::ImageBackground::Transparent);
    assert_eq!(app.selected_block, Some(outputs::BlockRef::Image(block.id)), "and selected");
    let first = snap(&mut app, "snap block=hero");
    assert!(wrote(&first), "{first:?}");
    let second = snap(&mut app, "snap block=hero");
    assert!(wrote(&second), "{second:?}");
    let names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    assert!(names.contains(&"untitled_hero_0001.png".to_string()) && names.contains(&"untitled_hero_0002.png".to_string()), "{names:?}");
    let bytes = std::fs::read(dir.join("untitled_hero_0001.png")).unwrap();
    let chunks = image_export::png_text_chunks(&bytes);
    assert!(chunks.contains(&("Comment".to_string(), "snap block=\"hero\"".to_string())), "{chunks:?}");
    assert!(chunks.iter().any(|(k, v)| k == "Software" && v.starts_with("FieldForm ")), "{chunks:?}");
    let img = load(&dir.join("untitled_hero_0001.png"));
    assert_eq!((img.width(), img.height()), (320, 240));
    let missing = snap(&mut app, "snap block=nobody");
    assert!(missing.iter().any(|(ok, t)| !ok && t.contains("nobody")), "{missing:?}");
    app.run_command("undo");
    app.run_command("undo");
    frame(&mut app, Vec::new());
    assert!(app.doc.images.is_empty(), "undo takes the block back");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_request_the_design_refuses_is_refused_by_name_before_anything_is_drawn() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("refusals");
    for (line, word) in [
        ("quality=still bg=transparent format=jpg", "alpha"),
        ("quality=still bits=16 format=jpg", "16-bit"),
        ("quality=still ss=3", "ss=3"),
        ("quality=still overlays=grid", "overlays"),
        ("width=50", "quality=still"),
        ("quality=still bg=blue", "bg=blue"),
    ] {
        let said = snap(&mut app, &format!("snap file={} {line}", path_arg(&dir, "x.png")));
        assert!(said.iter().any(|(ok, t)| !ok && t.contains(word)), "{line}: {said:?}");
        assert!(!dir.join("x.png").exists(), "{line} wrote nothing");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_field_of_the_block_is_a_key_saveimage_reads_or_a_recorded_gap() {
    use std::collections::BTreeSet;
    let mut block = fieldform_scene::SaveImage::new(fieldform_scene::PartId::new(), 10, 10);
    block.name = Some("x".into());
    block.view = Some("iso".into());
    let json = serde_json::to_value(&block).unwrap();
    let fields: BTreeSet<String> = json.as_object().unwrap().keys().cloned().collect();
    let verb = console_keys::verb("image.block").expect("saveimage declares its keys");
    let console_keys::Keys::Declared { reads, not_read } = &verb.keys else { panic!("declared") };
    let covered: BTreeSet<&str> = reads
        .iter()
        .flat_map(|k| k.covers.iter().copied())
        .chain(reads.iter().filter(|k| k.word == "name").map(|_| "name"))
        .chain(not_read.iter().map(|(f, _)| *f))
        .collect();
    for field in &fields {
        assert!(covered.contains(field.as_str()), "the block's field {field:?} is reached by no `saveimage` key and not recorded as unread");
    }
}

#[test]
fn a_still_with_a_view_leaves_the_persons_camera_where_it_was() {
    let Some(mut app) = sphere_app() else { return };
    let dir = scratch("view");
    let eye = app.camera.eye;
    let said = snap(&mut app, &format!("snap file={} quality=still width=200 height=150 view=top", path_arg(&dir, "top.png")));
    assert!(wrote(&said), "{said:?}");
    assert_eq!(app.camera.eye, eye, "a Still aims a copy of the camera");
    let said = snap(&mut app, &format!("snap file={} view=top", path_arg(&dir, "shown.png")));
    assert!(wrote(&said), "{said:?}");
    assert_ne!(app.camera.eye, eye, "the displayed picture moves the one camera, as it always did");
    let _ = std::fs::remove_dir_all(&dir);
}

/// §8 step 10's measurement: a 5000-wide Still with `ss=2` of a gyroid-infilled cube, the
/// time it took and how it was cut up, printed for the spec's §10.5. Ignored, since it is a
/// measurement rather than a property; run it by name with `--ignored --nocapture`.
#[test]
#[ignore]
fn measure_a_five_thousand_wide_still_of_a_gyroid_cube() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    for line in [
        // A gyroid that can be seen: a sheet lattice cut to a cube, not an infill behind a skin.
        "cube 'block' size=100,100,100 centre=0,0,0",
        "tpms cell=gyroid cell_size=10 phase=sheet thickness=1 'g'",
        "boolean -i in=block,g 'part'",
        "display quality=full",
        "frameall",
    ] {
        let before = app.console_log.len();
        app.run_command(line);
        frame(&mut app, Vec::new());
        for l in app.console_log.iter().skip(before) {
            eprintln!("{} {}", if l.ok { "  " } else { "!!" }, l.text);
        }
    }
    let dir = scratch("measure");
    for (name, keys) in [
        ("one.png", "quality=still"),
        ("five-thousand.png", "quality=still width=5000 ss=2"),
        ("five-thousand-alpha.png", "quality=still width=5000 ss=2 bg=transparent"),
    ] {
        let started = std::time::Instant::now();
        let said = snap(&mut app, &format!("snap file={} {keys}", path_arg(&dir, name)));
        let wall = started.elapsed();
        for (ok, text) in &said {
            eprintln!("{} {text}", if *ok { "  " } else { "!!" });
        }
        eprintln!("   wall {:.1} s for {name} ({keys})", wall.as_secs_f32());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
