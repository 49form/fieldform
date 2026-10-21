// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a settled frame's march is made of: the primary ray, the occlusion samples, the shadow.
//!
//! The plan's §5 items all went at the primary march, because that is all a draft is. A still
//! adds five occlusion samples and a shadow march of up to forty-eight readings to every pixel
//! that hit, and nothing has measured those since V0 — which timed drafts only. This splits a
//! still's march into its three parts, on the card's own clock, with the viewport's settled
//! grid and masks in place, so that whatever is done next is done to the part that costs.
//!
//! The split is by subtraction, which the three parts allow because they are sequential within a
//! pixel: the march alone; the march with occlusion and a shadow that gives up after one
//! reading; the march with both in full.
//!
//! **And the still at one pixel in four** (loop spec P5), with the occlusion and the shadow
//! marched for a quarter of the pixels and for those the lighting pass cannot fill in from them:
//! what that costs, beside what every pixel costs -- which is how the viewport draws a still --
//! and what it does to the finished picture, eight jittered frames each way, as they are shown,
//! in display levels of eight bits. The three parts above are measured at every pixel, so that
//! they split the march as they always did. The cost is given for the march and for the whole
//! frame, march and lighting, since the filling in is the lighting pass's; and how many of the
//! pixels that hit were marched. Beside the picture's change is the yardstick it is held to: how
//! far the still measured at every pixel moves between its fourth frame and its eighth, which is
//! what Full's eight frames are paid for over Balanced's four.
//!
//! **And a second table, on whether a still could stop early** (P5's other half): frame by frame
//! from the fourth, how far each moved the picture, by the plan's measure -- the mean change over
//! a 64 × 64 sample -- and by two stricter ones, and how far a still stopped there would be from
//! the one that ran to eight. See `converging`.
//!
//! `cargo run --release -p fieldform-io --example still_cost -- <file.fieldform>…`

use fieldform_gpu::{
    generate, BakeSettings, Camera, CurveUse, EnvironmentMaps, Look, Occlusion, Renderer,
    ShadingResources, Studio,
};
use fieldform_scene::{Document, Graph, Material};
use nalgebra::Point3;

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 732;

/// The march's GPU time for one presented frame, median of nine after five thrown away.
fn march_ms(
    renderer: &Renderer,
    shading: &mut ShadingResources,
    camera: &Camera,
    budget: u32,
    look: &Look,
) -> f64 {
    timed(renderer, shading, camera, budget, look).march
}

/// One presented frame on the card's own clock, as medians of nine after five thrown away: the
/// march pass, and the march with the lighting pass after it.
///
/// **The lighting pass is in the second because one in four put work there.** It fills in the
/// three pixels of each block the march did not measure, reading the G-buffer about each, so a
/// still at one in four is cheaper by what the march saves less what the lighting pass spends,
/// and timing the march alone credited it with the first and never charged it the second.
struct Timing {
    march: f64,
    frame: f64,
}

fn timed(
    renderer: &Renderer,
    shading: &mut ShadingResources,
    camera: &Camera,
    budget: u32,
    look: &Look,
) -> Timing {
    let (mut march, mut frame) = (Vec::new(), Vec::new());
    for round in 0..14 {
        renderer.present(shading, camera, WIDTH, HEIGHT, budget, look, 0).expect("presents");
        renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("the GPU finishes");
        let mut cost = None;
        for _ in 0..50 {
            if let Some((_, c)) = shading.costs().pop() {
                cost = c.march_ms.zip(c.shade_ms);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            let _ = renderer.device().poll(wgpu::PollType::Poll);
        }
        if round >= 5 {
            let (m, s) = cost.expect("this measurement needs GPU timestamps");
            march.push(f64::from(m));
            frame.push(f64::from(m + s));
        }
    }
    let median = |mut times: Vec<f64>| {
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    };
    Timing { march: median(march), frame: median(frame) }
}

fn measure(label: &str, doc: &Document, graph: &Graph) -> Vec<String> {
    let Some((lo, hi)) = fieldform_field::part_extent(doc, graph).as_box() else {
        println!("| {label} | — | — | — | — | — | no finite bounds to frame |");
        return Vec::new();
    };
    let program = generate(doc, graph).expect("the graph generates");
    let floor = ((hi - lo).norm() * 1e-5).max(1e-4);
    let curves: Option<Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)>> = program
        .curves
        .iter()
        .map(|use_of| match use_of {
            CurveUse::Spine(id) => Some((doc.curves[id].tessellate_to_angle(1.0, floor), None)),
            _ => None,
        })
        .collect();
    let Some(curves) = curves else {
        println!("| {label} | — | — | — | — | — | wants more than spines, which is all this supplies |");
        return Vec::new();
    };
    let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0))
        .expect("a GPU, which is where the march runs");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
        .expect("the lighting pipeline builds");

    let centre = Point3::from((lo.coords + hi.coords) * 0.5);
    let radius = (hi - lo).norm() * 0.5;
    // As the viewport has it once the field has stood still.
    renderer.build_skip_grid(centre - nalgebra::Vector3::repeat(radius), radius * 2.0, 64, true);

    let scale = fieldform_field::step_multiplier(doc, graph);
    let budget = (1024.0 * scale).ceil() as u32;
    let mut camera = Camera::framing((lo, hi), f64::from(WIDTH) / f64::from(HEIGHT));
    camera.start_at_scene = true;
    camera.value_scale = scale;
    let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);

    // Every pixel measured, as the viewport draws a still, so that the subtraction splits the
    // still's march as it was split before P5; one in four is the next column's.
    let full = Occlusion::for_scene(look.lights[0].direction, radius);
    let half = Occlusion { half_resolution: true, ..full };
    // A shadow that gives up at once: its first reading is past its reach.
    let occlusion_only = Occlusion { shadow_reach: 1e-9, ..full };
    // Each of the four twice over, in turn, and the lower of the two kept. Taken once and in
    // order, the first carried the card's clocks coming up and read twice what it should, and
    // every difference after it came out negative.
    let (mut primary, mut with_occlusion) = (f64::INFINITY, f64::INFINITY);
    let mut with_both = Timing { march: f64::INFINITY, frame: f64::INFINITY };
    let mut with_half = Timing { march: f64::INFINITY, frame: f64::INFINITY };
    let lower = |kept: &mut Timing, new: Timing| {
        kept.march = kept.march.min(new.march);
        kept.frame = kept.frame.min(new.frame);
    };
    for _ in 0..4 {
        camera.occlusion = Some(full);
        let _ = march_ms(&renderer, &mut shading, &camera, budget, &look);
    }
    for _ in 0..2 {
        camera.occlusion = None;
        primary = primary.min(march_ms(&renderer, &mut shading, &camera, budget, &look));
        camera.occlusion = Some(occlusion_only);
        with_occlusion = with_occlusion.min(march_ms(&renderer, &mut shading, &camera, budget, &look));
        camera.occlusion = Some(full);
        lower(&mut with_both, timed(&renderer, &mut shading, &camera, budget, &look));
        camera.occlusion = Some(half);
        lower(&mut with_half, timed(&renderer, &mut shading, &camera, budget, &look));
    }

    // **How many pixels one in four marches**, which is a quarter plus every pixel it could not
    // fill from its own surface: those with a reading of their own after a frame, of those that
    // hit. A pixel the lighting pass fills holds no reading.
    camera.occlusion = Some(half);
    renderer.present(&mut shading, &camera, WIDTH, HEIGHT, budget, &look, 0).expect("presents");
    let extras = renderer.read_extras(&shading).expect("the extras read back");
    let found = renderer.render(&camera, WIDTH, HEIGHT, budget).expect("renders");
    let hits = found.iter().filter(|h| h.is_hit()).count();
    let own = found.iter().zip(&extras).filter(|(h, [open, _])| h.is_hit() && *open >= 0.0).count();

    // **What one pixel in four does to the picture**, which no frame time can say: the still as
    // the viewport finishes it, eight jittered frames averaged, with the occlusion and shadow
    // measured at every pixel and at one in four, compared as they are shown. Beside it, the
    // yardstick: how far the every-pixel still moves from its fourth frame to its eighth, which is
    // the difference between Balanced's still and Full's and the one the eighth frame is paid for.
    camera.occlusion = Some(full);
    let every = still_frames(&renderer, &mut shading, &camera, budget, &look, 8);
    let finished = every.last().expect("eight frames");
    let yardstick = Apart::of(&every[3], finished);
    camera.occlusion = Some(half);
    let shown = still_frames(&renderer, &mut shading, &camera, budget, &look, 8);
    let quarter = shown.last().expect("eight frames");
    let apart = Apart::of(finished, quarter);
    camera.occlusion = Some(full);

    // And the same still as the march stood before the grid, the masks and the long step: what
    // those three are worth to a settled frame, where the plan only ever measured drafts.
    renderer.clear_skip_grid();
    renderer.set_relaxation(1.0);
    let mut before = f64::INFINITY;
    for _ in 0..2 {
        before = before.min(march_ms(&renderer, &mut shading, &camera, budget, &look));
    }

    println!(
        "| {label} | {:.1} % | {primary:.1} ms | {:.1} ms | {:.1} ms | **{:.1} ms**, of which the shadow is {:.0} % | {before:.1} ms | **{:.1} ms** | {:.1} → **{:.1} ms** | {:.3} | {} | {} |",
        100.0 * hits as f64 / f64::from(WIDTH * HEIGHT),
        with_occlusion - primary,
        with_both.march - with_occlusion,
        with_both.march,
        100.0 * (with_both.march - with_occlusion) / with_both.march.max(1e-9),
        with_half.march,
        with_both.frame,
        with_half.frame,
        own as f64 / hits.max(1) as f64,
        apart.describe(),
        yardstick.describe(),
    );
    // The still as the viewport draws it, which is every pixel's: 0070d0c read this table at one
    // in four, when that was the viewport's still.
    converging(label, &every)
}

/// **Whether a still could stop before its eighth frame** (loop spec P5's first half), as rows of
/// the second table: for each frame from the fourth, how far it moved the picture, by the plan's
/// measure and by two stricter ones, and how far a still stopped there is from the eighth.
///
/// The plan's measure is the mean change over a 64 × 64 sample of the image; the stricter are the
/// largest change in that sample, and the share of 8 × 8 tiles in which any pixel moved by a whole
/// level -- what a still that stopped tile by tile would still be tracing. All in display levels of
/// eight bits, each frame against the one before it, as they are shown.
fn converging(label: &str, shown: &[Vec<u8>]) -> Vec<String> {
    let moved = |a: &[u8], b: &[u8], i: usize| {
        (0..3).map(|c| a[i * 4 + c].abs_diff(b[i * 4 + c])).max().unwrap_or(0)
    };
    let sample: Vec<usize> = (0..64u32)
        .flat_map(|j| (0..64u32).map(move |i| (i, j)))
        .map(|(i, j)| ((((2 * j + 1) * HEIGHT) / 128) * WIDTH + ((2 * i + 1) * WIDTH) / 128) as usize)
        .collect();
    let (tiles_x, tiles_y) = (WIDTH.div_ceil(8), HEIGHT.div_ceil(8));
    let last = shown.last().expect("frames");
    (4..=shown.len())
        .map(|frames| {
            let (now, before) = (&shown[frames - 1], &shown[frames - 2]);
            let mean = sample.iter().map(|&i| f64::from(moved(now, before, i))).sum::<f64>()
                / sample.len() as f64;
            let largest = sample.iter().map(|&i| moved(now, before, i)).max().unwrap_or(0);
            let mut live = vec![false; (tiles_x * tiles_y) as usize];
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    if moved(now, before, (y * WIDTH + x) as usize) >= 1 {
                        live[((y / 8) * tiles_x + x / 8) as usize] = true;
                    }
                }
            }
            let tiles = live.iter().filter(|t| **t).count();
            let stopped = Apart::of(now, last);
            format!(
                "| {label} | {frames} | {mean:.3} | {largest} | {:.1} % | {:.3} % / {} |",
                100.0 * tiles as f64 / live.len() as f64,
                100.0 * stopped.over_four as f64 / stopped.pixels.max(1) as f64,
                stopped.worst,
            )
        })
        .collect()
}

/// The still as it is shown after each of `frames` jittered frames: display-encoded bytes.
fn still_frames(
    renderer: &Renderer,
    shading: &mut ShadingResources,
    camera: &Camera,
    budget: u32,
    look: &Look,
    frames: u32,
) -> Vec<Vec<u8>> {
    let mut jittered = *camera;
    (0..frames)
        .map(|frame| {
            jittered.jitter = fieldform_gpu::halton_jitter(frame);
            renderer
                .present(shading, &jittered, WIDTH, HEIGHT, budget, look, frame)
                .expect("presents");
            renderer.read_display(shading).expect("reads back").rgba
        })
        .collect()
}

/// How far apart two pictures of one view are, pixel by pixel, in display levels of eight bits:
/// a pixel is as far apart as its furthest channel.
struct Apart {
    over_one: usize,
    over_four: usize,
    worst: u8,
    mean: f64,
    pixels: usize,
}

impl Apart {
    fn of(a: &[u8], b: &[u8]) -> Self {
        let mut apart = Apart { over_one: 0, over_four: 0, worst: 0, mean: 0.0, pixels: 0 };
        for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let d = (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap_or(0);
            apart.over_one += usize::from(d > 1);
            apart.over_four += usize::from(d > 4);
            apart.worst = apart.worst.max(d);
            apart.mean += f64::from(d);
            apart.pixels += 1;
        }
        apart.mean /= apart.pixels.max(1) as f64;
        apart
    }

    fn describe(&self) -> String {
        let share = |n: usize| 100.0 * n as f64 / self.pixels.max(1) as f64;
        format!(
            "{:.3} % / {:.3} % / {} / {:.3}",
            share(self.over_one),
            share(self.over_four),
            self.worst,
            self.mean
        )
    }
}

fn main() {
    println!("{WIDTH} × {HEIGHT}, the model framed, the 64³ grid with its masks, one still frame.\n");
    println!("| scene | pixels that hit | primary march | five occlusion samples | the shadow march | the whole march | the whole march with no grid, no masks and plain steps | the whole march, occlusion and shadow at one pixel in four | the frame, march and lighting: every pixel → one in four | share of the pixels that hit marched at one in four | eight frames at one in four against every pixel: more than 1 level / more than 4 / worst / mean | the yardstick, every pixel's fourth frame against its eighth: the same four |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut converging = Vec::new();
    for path in std::env::args().skip(1) {
        match fieldform_io::open(std::path::Path::new(&path)) {
            Ok((doc, _)) => {
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(path.clone());
                let graphs: Vec<Graph> = doc
                    .parts
                    .values()
                    .filter(|p| !p.graph.roots().is_empty())
                    .map(|p| p.graph.clone())
                    .collect();
                for graph in graphs {
                    converging.extend(measure(&format!("`{name}`"), &doc, &graph));
                }
            }
            Err(e) => println!("could not open {path}: {e}"),
        }
    }
    println!("\nThe still as the viewport draws it, every pixel measured, frame by frame from the fourth.\n");
    println!("| scene | frames | the 64 × 64 sample's mean change, levels | its largest | 8 × 8 tiles with a pixel that moved a level | stopped here, against eight frames: more than 4 levels / worst |");
    println!("|---|---|---|---|---|---|");
    for row in converging {
        println!("{row}");
    }
}
