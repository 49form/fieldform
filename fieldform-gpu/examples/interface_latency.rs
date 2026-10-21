// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How long an interface frame waits behind the picture's march, what bounding the march would
//! buy, and whose cadence the march should run on — the measurements the loop spec's P10 asks
//! for before anything is built.
//!
//! **The mechanism, stated so the number can be read against it.** The viewport's march is
//! submitted from inside egui's own frame (`traced/draw.rs`), into the one queue the window is
//! presented through. Everything the interface then queues — egui's render pass, and the
//! presentation after it — sits behind that march, so a frame cannot appear until the march
//! finishes. A 55 ms draft is a 55 ms interface frame: the "sluggish" of the loop spec's §1
//! row 4.
//!
//! Two mitigations already exist and are not what this measures. The **draft ladder** shrinks a
//! draft until it fits 50 ms, and **one march in flight** stops a second being queued behind a
//! running one. Both reduce how *often* the interface waits. Neither changes how long it waits
//! when it does, which is what a band would.
//!
//! **What the first table measures.** Each arm submits some march work and then a trivial
//! submission standing for the interface's own frame — a buffer copy, microseconds of work.
//! Submissions complete in order, so the time until the trivial one reports is what an interface
//! frame would have waited. The arms differ only in how much march went in front of it:
//!
//! - **whole frame**: the draft as the viewport submits one today, every row in one packet.
//! - **banded**: one band of it, cut as the viewport cuts one.
//!
//! The difference between those two is the prize, and the cost is that the picture then takes
//! several frames to complete rather than one.
//!
//! **A band is sized as the viewport sizes one: from the dearest row, never under a wave.**
//! `traced/mod.rs`'s `band_rows` aims a band at `INTERFACE_BUDGET_MS` from the dearest row cost
//! it has measured, never fewer pixels than one wave of the card, in whole tiles. This walks the
//! frame the same way and settles on the same band. The first run of this example divided the
//! frame's cost by its rows instead — a mean, which the loop spec's own P10 calls the wrong
//! band — and the bands it measured on the heavy documents were 19–23 ms, over a vertical blank,
//! which is not a band the viewport would have cut. The dearest band of a sweep is still what
//! the table reports, since a band across the background and a band across the model cost
//! seventeen times apart for the same rows, and an interface frame has to survive the worst.
//!
//! **The second table is the other half of P10: whose cadence the march runs on.** The viewport
//! submits its march from inside the interface's own frame, at most one in flight. The loop spec's
//! design is to trace on a cadence of its own and have the interface show the last finished image
//! at whatever rate it likes. Both are run here against one model of eframe's frame: the input is
//! read, the application runs, and egui's paint then waits for a swapchain image before submitting
//! its own work — an image there is once fewer than two frames are waiting to be shown (wgpu's
//! default depth, which the application leaves alone), and a frame is shown at the first vertical
//! blank, sixty a second, after its work has finished. Each arm is that same interface; only who
//! submits the march, and when, differs:
//!
//! - **in the frame (today)**: the band goes up while the application runs, so before the paint
//!   waits for its image.
//! - **in the paint**: the band goes up with egui's own work, after the image, which is where a
//!   paint callback's command buffers are submitted.
//! - **own cadence**: a thread submits the next band the moment the last has finished.
//!
//! What each reports is what a person feels: frames shown a second, and how long a frame's input
//! took to reach the screen — from the input being read to the vertical blank it was shown at —
//! then the picture's own rate, and last what each thread's *submission* cost it on the CPU.
//!
//! **No thread here blocks in the device while another may submit.** wgpu-core takes the device's
//! fence for reading across a blocking `poll` and for writing across `Queue::submit`, so a thread
//! that waits for its band with `PollType::Wait` holds every other thread's submit until the band
//! has finished. The first run of this table did that in the own-cadence arm, and the interface's
//! submit — a 256-byte copy — read 17.6 ms median and 26.5 ms worst on a 26 ms draft, 4.9 / 12.6
//! ms banded, where every other arm read it in microseconds. The arm had measured a CPU lock
//! between two threads and the result was recorded as the one queue's cost (2026-10-13,
//! `ee23a9c`). So every wait here is a poll that never blocks, with a sleep between polls — the
//! interface's wait for its frame as much as the tracer's for its band, since the fence is held
//! the same way whichever thread waits — and the last column says, in microseconds, whether that
//! is still so.
//!
//! **Which rows are the viewport's own code path.** The viewport bands a draft only at the
//! ladder's last rung, 240 px, and only when a whole draft there misses `DRAFT_BUDGET_MS`; at that
//! width a band is at least 144 rows, so a draft is at most two bands. A banded row at any larger
//! size is the model's arm — what the design would do if the viewport banded there — and is read
//! as that, not as the code path.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example interface_latency -- <doc.fform> [px]`
//!
//! The pixel size defaults to 240, the draft ladder's floor, because the floor is where the
//! question lives: a document whose draft misses its budget *there* is one the ladder cannot
//! shrink any further.

use fieldform_gpu::{
    generate, BakeSettings, Camera, EnvironmentMaps, Look, Renderer, ShadingResources, Studio,
};
use fieldform_scene::Material;
use nalgebra::{Point3, Vector3};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The viewport's own step budget, as `traced/rebuild.rs` sets it: the base, times what a driven
/// value costs, with headroom.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;

/// What one submission may cost if the interface is to keep sixty frames a second while it is
/// queued ahead of their presentation. Two thirds of a 16.7 ms frame, leaving egui its own share.
const INTERFACE_BUDGET_MS: f64 = 11.0;

/// The fewest pixels a band may hold, one wave of the card: `traced/mod.rs`'s `MIN_BAND_PIXELS`.
const MIN_BAND_PIXELS: u32 = 1 << 15;
/// Rows in one tile of the per-tile body mask, which a band's count is rounded up to:
/// `traced/mod.rs`'s `TILE_ROWS`.
const TILE_ROWS: u32 = 8;
/// The last rung of the viewport's draft ladder, the only size at which it bands a draft.
const LADDER_FLOOR_PX: u32 = 240;

/// The display's refresh, which a swapchain presenting with vsync paces every frame to.
const VBLANK: Duration = Duration::from_micros(16_667);
/// Frames submitted and not yet shown before the next has to wait for an image: wgpu's default
/// `desired_maximum_frame_latency`, which eframe passes through and the application leaves alone.
const FRAMES_IN_FLIGHT: usize = 2;
/// How long each arm of the cadence table runs in one round.
const CADENCE_SECONDS: f64 = 3.0;
/// Rounds of the cadence arms, interleaved, so a card that warms or clocks down during the run
/// reads in every arm rather than in whichever happened to run last.
const CADENCE_ROUNDS: usize = 3;
/// How long a thread sleeps between two polls of the device while it waits for the card. Coarse
/// against a submission, fine against a vertical blank, which is what the latency is read to.
const POLL_EVERY: Duration = Duration::from_micros(100);

/// Who submits the march, and when, relative to the interface's own frame.
#[derive(Clone, Copy, PartialEq)]
enum Schedule {
    /// While the application runs, before the paint waits for an image: the viewport today.
    InTheFrame,
    /// With egui's own work, once the paint has its image.
    InThePaint,
    /// On a thread of its own, the next band the moment the last has finished.
    OwnCadence,
}

impl Schedule {
    fn label(self) -> &'static str {
        match self {
            Schedule::InTheFrame => "in the frame (today)",
            Schedule::InThePaint => "in the paint",
            Schedule::OwnCadence => "own cadence",
        }
    }
}

/// What one arm of the cadence table saw, summed over its rounds.
#[derive(Default)]
struct Cadence {
    /// From each shown frame's input being read to the vertical blank it was shown at, in
    /// milliseconds.
    latency: Vec<f64>,
    seconds: f64,
    bands: usize,
    /// What submitting a band cost the thread that submitted it, in microseconds.
    submit_us: Vec<f64>,
    /// What the interface's own submission cost its thread, in microseconds. Microseconds when
    /// the threads share only the queue; milliseconds when one of them holds the device's fence
    /// across a wait, which is the fault this column exists to show.
    interface_us: Vec<f64>,
}

/// The value a fraction `p` of the way up values already sorted ascending, by nearest rank.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// A band as the viewport cuts one — `band_rows` and `tile_aligned_rows` in `traced/mod.rs` —
/// given the dearest row cost measured at this size: toward `budget_ms` at that cost, never fewer
/// pixels than one wave of the card, rounded up to whole tiles and no taller than the frame.
///
/// The viewport also caps a band at half again the band before it, which is a transient on the
/// way up from the floor; this is the band it settles to. An infinite row cost asks for the floor.
fn viewport_band_rows(budget_ms: f64, dearest_ms_per_row: f64, width: u32, height: u32) -> u32 {
    let floor = MIN_BAND_PIXELS.div_ceil(width.max(1)).clamp(1, height);
    let wanted = (budget_ms / dearest_ms_per_row.max(1e-6)) as u32;
    wanted.clamp(floor, height).max(1).next_multiple_of(TILE_ROWS).min(height)
}

/// The bands of a frame `height` rows tall, cut `rows` at a time from the top, the last shorter.
fn bands_of(rows: u32, height: u32) -> Vec<Range<u32>> {
    (0..height.div_ceil(rows)).map(|k| k * rows..((k + 1) * rows).min(height)).collect()
}

/// A flag the card sets, through `on_submitted_work_done`, once the most recent submission has
/// finished. Set on whichever thread next polls the device, so a thread waiting on it must poll.
fn finished_flag(queue: &wgpu::Queue) -> Arc<AtomicBool> {
    let finished = Arc::new(AtomicBool::new(false));
    let set = Arc::clone(&finished);
    queue.on_submitted_work_done(move || set.store(true, Ordering::Release));
    finished
}

/// Waits until `finished` says so, polling the device without ever blocking inside it.
///
/// **Never `PollType::Wait` from a thread another thread submits beside.** wgpu-core holds the
/// device's fence for reading across a blocking wait and takes it for writing in `Queue::submit`,
/// so a blocking wait here would hold the other thread's submit — the interface's frame, or the
/// tracer's band — until this thread's work had finished, and the table would measure that lock
/// rather than the queue. `PollType::Poll` holds the fence for the microseconds of a fence read,
/// and the sleep between polls is what keeps the wait from being a spin.
fn wait_polling(device: &wgpu::Device, mut finished: impl FnMut() -> bool) {
    loop {
        let _ = device.poll(wgpu::PollType::Poll);
        if finished() {
            return;
        }
        std::thread::sleep(POLL_EVERY);
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("a .fform path");
    let px: u32 = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(LADDER_FLOOR_PX);
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&path)).expect("the document opens");
    let part = doc.parts.values().next().expect("a part");
    let graph = &part.graph;

    let program = generate(&doc, graph).expect("the part generates a shader");
    let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
        Ok(renderer) => renderer,
        Err(e) => {
            eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
            return;
        }
    };
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);

    let (centre, radius) = match fieldform_field::part_extent(&doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => {
            (nalgebra::center(&lo, &hi), (hi - lo).norm() * 0.5)
        }
        _ => (Point3::origin(), 50.0),
    };
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    let multiplier = fieldform_field::step_multiplier(&doc, graph);
    let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
        .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);
    let mut camera = camera;
    camera.value_scale = multiplier;

    // The lighting pass, because `present_rows` is the path the viewport submits through and the
    // only one that submits a march **without waiting for it** -- which is the whole point here.
    // `render` reads the frame back and waits, which would hide the queue this measures.
    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
        .expect("the lighting pipeline builds");
    let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);

    // A trivial submission standing for the interface's own frame: egui's render pass and the
    // presentation after it are a fraction of a millisecond of GPU work on any card that can run
    // this at all. What it measures is the queue, not itself.
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("interface work"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("interface work"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let interface_frame = || {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&a, 0, &b, 0, 256);
        queue.submit(Some(encoder.finish()));
        finished_flag(&queue)
    };

    let mut march = |rows: Range<u32>| {
        renderer
            .present_rows(&mut shading, &camera, px, px, budget, &look, 0, rows)
            .expect("the frame presents");
    };
    // Warm: the first submission of any program pays a pipeline the rest do not.
    march(0..px);
    device.poll(wgpu::PollType::wait_indefinitely()).expect("the queue drains");

    // One submission alone on the queue, the least of `runs`: what the bands are sized from and
    // what the arms below are read against. Single-threaded here, so the blocking wait holds
    // nobody's submit.
    let mut alone = |rows: Range<u32>, runs: usize| {
        let mut best = f64::INFINITY;
        for _ in 0..runs {
            device.poll(wgpu::PollType::wait_indefinitely()).expect("the queue drains");
            let start = Instant::now();
            march(rows.clone());
            device.poll(wgpu::PollType::wait_indefinitely()).expect("the queue drains");
            best = best.min(start.elapsed().as_secs_f64() * 1000.0);
        }
        best
    };
    let whole_ms = alone(0..px, 5);

    // **The band the viewport settles to.** It starts at the floor and resizes from the dearest
    // row it has measured, so the walk here is the same: bands of the current size are timed, the
    // dearest row they show sizes the next, until the size holds. The dearest row cost depends on
    // the band it was measured over -- finer bands find denser rows -- which is why this is a
    // fixed point rather than one measurement.
    let floor = viewport_band_rows(INTERFACE_BUDGET_MS, f64::INFINITY, px, px);
    let mut rows = floor;
    let mut dearest_per_row = f64::INFINITY;
    for _ in 0..4 {
        dearest_per_row = bands_of(rows, px)
            .into_iter()
            .map(|band| alone(band.clone(), 3) / f64::from(band.end - band.start))
            .fold(0.0, f64::max);
        let next = viewport_band_rows(INTERFACE_BUDGET_MS, dearest_per_row, px, px);
        if next == rows {
            break;
        }
        rows = next;
    }
    let bands = bands_of(rows, px);
    let wanted = (INTERFACE_BUDGET_MS / dearest_per_row.max(1e-6)) as u32;

    println!("{}, {px} px: the whole draft is {whole_ms:.1} ms", path_tail(&path));
    println!(
        "the dearest row costs {:.3} ms, so a band toward {INTERFACE_BUDGET_MS} ms is {rows} rows{} \
         and the frame is {} bands",
        dearest_per_row,
        if wanted < floor {
            format!(" -- the floor of one wave at this width, where the budget alone would be {wanted}")
        } else {
            String::new()
        },
        bands.len(),
    );
    println!();
    println!("| what the interface queued behind | its frame waited | frames a second |");
    println!("|---|---|---|");

    // `render` is the readback path and waits for the frame itself, which would hide the queue.
    // The arms below submit the march without waiting and then time the trivial submission, which
    // is the thing an interface frame actually is.
    // **Every band, not only the first.** A band of the top rows is mostly sky, and reading the
    // interface against that would flatter it by the difference between the cheapest band and
    // the dearest. What an interface frame has to survive is its worst, so the bands are walked
    // and the dearest is what the table reports.
    let mut arms: Vec<(String, Range<u32>)> =
        vec![("the whole draft, as today".to_string(), 0..px)];
    for (k, band) in bands.iter().enumerate() {
        arms.push((format!("band {} of {}", k + 1, bands.len()), band.clone()));
    }
    let mut dearest: Option<(String, f64)> = None;
    for (label, rows) in arms {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            // Drained first, so each reading starts from an idle queue rather than from the
            // tail of the last one. One thread here, so the blocking wait holds nobody's submit.
            device.poll(wgpu::PollType::wait_indefinitely()).expect("the queue drains");
            let start = Instant::now();
            march(rows.clone());
            let _finished = interface_frame();
            device.poll(wgpu::PollType::wait_indefinitely()).expect("the queue drains");
            best = best.min(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("| {label} | {best:.1} ms | {:.0} |", 1000.0 / best.max(1e-6));
        if label.starts_with("band") && dearest.as_ref().is_none_or(|(_, ms)| best > *ms) {
            dearest = Some((label, best));
        }
    }
    if let Some((label, ms)) = dearest {
        println!();
        println!(
            "The dearest band is {label} at {ms:.1} ms, so banding holds an interface frame to that"
        );
    }
    println!();
    println!(
        "The second row is what an interface frame would wait if a draft went up in bands, and the\n\
         cost is that the picture takes {} frames to complete rather than one.",
        bands.len()
    );

    // **Whose cadence.** The same bands, and the whole frame for comparison, under each schedule.
    let sweeps: [(&str, Vec<Range<u32>>); 2] = [("whole", vec![0..px]), ("banded", bands.clone())];
    let schedules = [Schedule::InTheFrame, Schedule::InThePaint, Schedule::OwnCadence];
    let mut seen: Vec<Vec<Cadence>> = sweeps
        .iter()
        .map(|_| schedules.iter().map(|_| Cadence::default()).collect())
        .collect();
    for _ in 0..CADENCE_ROUNDS {
        for (s, (_, sweep)) in sweeps.iter().enumerate() {
            for (k, schedule) in schedules.iter().enumerate() {
                device.poll(wgpu::PollType::wait_indefinitely()).expect("the queue drains");
                let _ = shading.costs();
                cadence(
                    *schedule,
                    &mut seen[s][k],
                    sweep,
                    &device,
                    &interface_frame,
                    &|shading: &mut ShadingResources, rows: Range<u32>| {
                        renderer
                            .present_rows(shading, &camera, px, px, budget, &look, 0, rows)
                            .expect("the frame presents");
                    },
                    &mut shading,
                );
            }
        }
    }
    println!();
    println!(
        "Whose cadence: {CADENCE_ROUNDS} interleaved rounds of {CADENCE_SECONDS} s an arm, a {:.1} ms \
         vertical blank, {FRAMES_IN_FLIGHT} frames in flight, bands of {rows} rows",
        VBLANK.as_secs_f64() * 1000.0
    );
    println!();
    println!(
        "| sweep | march submitted | frames shown a second | input to screen, median | 95th | worst | \
         pictures a second | submitting a band | the interface's submit, median / worst |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for (s, (name, sweep)) in sweeps.iter().enumerate() {
        for (k, schedule) in schedules.iter().enumerate() {
            let c = &mut seen[s][k];
            c.latency.sort_by(f64::total_cmp);
            c.submit_us.sort_by(f64::total_cmp);
            c.interface_us.sort_by(f64::total_cmp);
            println!(
                "| {name} ({} a picture) | {} | {:.0} | {:.1} ms | {:.1} ms | {:.1} ms | {:.1} | {:.0} µs | {:.0} / {:.0} µs |",
                sweep.len(),
                schedule.label(),
                c.latency.len() as f64 / c.seconds,
                percentile(&c.latency, 0.5),
                percentile(&c.latency, 0.95),
                c.latency.last().copied().unwrap_or(f64::NAN),
                c.bands as f64 / sweep.len() as f64 / c.seconds,
                percentile(&c.submit_us, 0.5),
                percentile(&c.interface_us, 0.5),
                c.interface_us.last().copied().unwrap_or(f64::NAN),
            );
        }
    }
    let floor_rows = viewport_band_rows(INTERFACE_BUDGET_MS, f64::INFINITY, LADDER_FLOOR_PX, LADDER_FLOOR_PX);
    println!();
    println!(
        "The viewport bands a draft only at the ladder's last rung, {LADDER_FLOOR_PX} px, where a band \
         is at least {floor_rows} rows and a draft at most {} bands; a banded row at any other size \
         is the model's arm, not the code path's.",
        LADDER_FLOOR_PX.div_ceil(floor_rows)
    );
}

/// Runs the interface for [`CADENCE_SECONDS`] under one schedule and adds what it saw to `into`.
///
/// **The model of the window, in eframe's order.** A frame reads its input and runs the
/// application; its paint then waits for a swapchain image, which there is once fewer than
/// [`FRAMES_IN_FLIGHT`] frames are waiting to be shown, and submits its own work; and it is shown
/// at the first vertical blank after that work has finished and after the frame before it was
/// shown. So a frame waits behind whatever march is queued ahead of its work, and a frame that
/// waits too long holds the next one's image — which is how a slow march becomes a slow
/// interface. The march keeps one band in flight under every schedule, asked of the lighting pass
/// exactly as the viewport asks it, and every wait on the card is a poll ([`wait_polling`]).
fn cadence(
    schedule: Schedule,
    into: &mut Cadence,
    sweep: &[Range<u32>],
    device: &wgpu::Device,
    interface_frame: &(dyn Fn() -> Arc<AtomicBool> + Sync),
    march: &(dyn Fn(&mut ShadingResources, Range<u32>) + Sync),
    shading: &mut ShadingResources,
) {
    let start = Instant::now();
    let stop = AtomicBool::new(false);
    let marched = AtomicUsize::new(0);
    let mut submit_us = Vec::new();
    let mut interface_us = Vec::new();
    let mut latency = Vec::new();
    let (traced, mut shading) = if schedule == Schedule::OwnCadence {
        (Some(shading), None)
    } else {
        (None, Some(shading))
    };
    std::thread::scope(|scope| {
        // The trace's own thread: it owns the lighting pass for the arm and submits the next
        // band as soon as the card has finished the last, whatever the interface is doing. It
        // learns that the way the viewport does, from the pass's own arrival, and polls for it.
        let tracer = traced.map(|shading| {
            let (stop, marched) = (&stop, &marched);
            scope.spawn(move || {
                let mut us = Vec::new();
                let mut next = 0;
                while !stop.load(Ordering::Relaxed) {
                    let at = Instant::now();
                    march(shading, sweep[next % sweep.len()].clone());
                    us.push(at.elapsed().as_secs_f64() * 1e6);
                    next += 1;
                    wait_polling(device, || {
                        let _ = shading.costs();
                        !shading.busy()
                    });
                    marched.fetch_add(1, Ordering::Relaxed);
                }
                us
            })
        });
        let mut next = 0;
        let mut band = |shading: &mut ShadingResources, submit_us: &mut Vec<f64>| {
            // One march in flight, read the way the viewport reads it.
            let _ = device.poll(wgpu::PollType::Poll);
            let _ = shading.costs();
            if !shading.busy() {
                let at = Instant::now();
                march(shading, sweep[next % sweep.len()].clone());
                submit_us.push(at.elapsed().as_secs_f64() * 1e6);
                next += 1;
                marched.fetch_add(1, Ordering::Relaxed);
            }
        };
        let mut waiting: VecDeque<(Instant, Arc<AtomicBool>)> = VecDeque::new();
        let mut shown_at = start;
        let show = |begun: Instant, finished: Arc<AtomicBool>, shown_at: &mut Instant| {
            wait_polling(device, || finished.load(Ordering::Acquire));
            // The first vertical blank after the work finished, and never two frames at one.
            let done = Instant::now();
            let blanks = ((done - start).as_secs_f64() / VBLANK.as_secs_f64()).ceil() as u32;
            let blank = (start + VBLANK * blanks).max(*shown_at + VBLANK);
            if let Some(rest) = blank.checked_duration_since(Instant::now()) {
                std::thread::sleep(rest);
            }
            *shown_at = blank;
            (blank - begun).as_secs_f64() * 1000.0
        };
        while start.elapsed().as_secs_f64() < CADENCE_SECONDS {
            // The input is read, and the application runs.
            let begun = Instant::now();
            if schedule == Schedule::InTheFrame {
                band(shading.as_deref_mut().expect("the frame marches"), &mut submit_us);
            }
            // The paint's image: there is one once fewer frames than the swapchain's depth are
            // waiting to be shown.
            while waiting.len() >= FRAMES_IN_FLIGHT {
                let (begun, finished) = waiting.pop_front().expect("a frame is waiting");
                latency.push(show(begun, finished, &mut shown_at));
            }
            if schedule == Schedule::InThePaint {
                band(shading.as_deref_mut().expect("the frame marches"), &mut submit_us);
            }
            let at = Instant::now();
            let finished = interface_frame();
            interface_us.push(at.elapsed().as_secs_f64() * 1e6);
            waiting.push_back((begun, finished));
        }
        // What is still waiting was begun inside the arm, so it is shown, and counted, inside it:
        // the march carries on meanwhile, as it would have.
        while let Some((begun, finished)) = waiting.pop_front() {
            latency.push(show(begun, finished, &mut shown_at));
        }
        let seconds = start.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
        if let Some(tracer) = tracer {
            submit_us.extend(tracer.join().expect("the tracer finishes"));
        }
        into.seconds += seconds;
    });
    into.latency.extend(latency);
    into.bands += marched.load(Ordering::Relaxed);
    into.submit_us.extend(submit_us);
    into.interface_us.extend(interface_us);
}

fn path_tail(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}
