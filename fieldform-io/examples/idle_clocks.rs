// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the first draft after an idle costs, and what keeps it from costing that.
//!
//! Plan §5 V4. V0 found the first draft after the viewport has sat converged running three to
//! four times slower than the drafts that follow it, at the same camera and size: the card's
//! clocks coming up from idle. It is why a first drag feels worse than the drag itself. The
//! target is the first draft after five idle seconds within 1.5× the steady median.
//!
//! This measures the candidates rather than choosing one by argument:
//!
//! - nothing, which is today;
//! - a **trickle** through the idle: a few rows of the draft re-traced every so often, so the
//!   card never sees an empty queue for long. It costs power for as long as the window sits;
//! - a **warm-up** just before the draft: nothing through the idle, then a burst of the same
//!   few rows for a fraction of a second, which is what the pointer crossing the viewport buys
//!   before the button goes down.
//!
//! Every figure is the march pass's own GPU time from the card's timestamps, not the wall clock.
//!
//! `cargo run --release -p fieldform-io --example idle_clocks -- <file.fieldform>`

use std::time::{Duration, Instant};

use fieldform_gpu::{
    generate, BakeSettings, Camera, CurveUse, EnvironmentMaps, Look, Renderer, ShadingResources,
    Studio,
};
use fieldform_scene::Material;
use nalgebra::Point3;

const WIDTH: u32 = 720;
const HEIGHT: u32 = 488;
const IDLE: Duration = Duration::from_secs(5);

struct Rig {
    renderer: Renderer,
    shading: ShadingResources,
    camera: Camera,
    look: Look,
    budget: u32,
}

impl Rig {
    /// Traces rows `rows` of the draft, waits for the card, and returns the march's GPU time.
    ///
    /// From the top of the frame, which is work only because `Camera::framing` fills the frame
    /// with the model. The application marches the *middle* rows for its warm-up: on a model
    /// that sits inside the frame the top rows all miss the scene's sphere and cost nothing,
    /// which the first version of it found out by warming nothing.
    fn trace(&mut self, rows: std::ops::Range<u32>) -> f32 {
        self.renderer
            .present_rows(&mut self.shading, &self.camera, WIDTH, HEIGHT, self.budget, &self.look, 0, rows)
            .expect("the rows present");
        self.renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("the GPU finishes");
        // The timestamps resolve a moment after the work does.
        for _ in 0..50 {
            if let Some((_, cost)) = self.shading.costs().pop() {
                return cost.march_ms.expect("this measurement needs GPU timestamps");
            }
            std::thread::sleep(Duration::from_millis(1));
            let _ = self.renderer.device().poll(wgpu::PollType::Poll);
        }
        panic!("the submission's cost never arrived");
    }

    fn draft(&mut self) -> f32 {
        self.trace(0..HEIGHT)
    }

    /// A whole draft `longest` pixels along its longer edge, as a lower rung of the ladder is.
    fn draft_at(&mut self, longest: u32) -> f32 {
        let (width, height) = (longest, (longest * HEIGHT).div_ceil(WIDTH));
        self.renderer
            .present(&mut self.shading, &self.camera, width, height, self.budget, &self.look, 0)
            .expect("the draft presents");
        self.renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("the GPU finishes");
        for _ in 0..50 {
            if let Some((_, cost)) = self.shading.costs().pop() {
                return cost.march_ms.expect("this measurement needs GPU timestamps");
            }
            std::thread::sleep(Duration::from_millis(1));
            let _ = self.renderer.device().poll(wgpu::PollType::Poll);
        }
        panic!("the submission's cost never arrived");
    }

    /// Sits for `total`, tracing `rows` rows every `every` — or nothing at all, given `None`.
    /// Returns how many trickles ran and the GPU time they took between them.
    /// The warm-up as the application can actually run it: one band in flight, and the next
    /// not before the interface's next frame, sixteen milliseconds after the last began.
    fn paced(&mut self, total: Duration, rows: u32) -> (u32, f32) {
        let until = Instant::now() + total;
        let (mut ran, mut spent) = (0, 0.0);
        while Instant::now() < until {
            let began = Instant::now();
            spent += self.trace(0..rows);
            ran += 1;
            std::thread::sleep(Duration::from_millis(16).saturating_sub(began.elapsed()));
        }
        (ran, spent)
    }

    fn sit(&mut self, total: Duration, trickle: Option<(Duration, u32)>) -> (u32, f32) {
        let until = Instant::now() + total;
        let (mut ran, mut spent) = (0, 0.0);
        match trickle {
            None => std::thread::sleep(total),
            Some((every, rows)) => {
                while Instant::now() < until {
                    spent += self.trace(0..rows);
                    ran += 1;
                    std::thread::sleep(every);
                }
            }
        }
        (ran, spent)
    }
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let path = std::env::args().nth(1).expect("a .fieldform file to draw");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("the document opens");
    let graph = doc
        .parts
        .values()
        .find(|p| !p.graph.roots().is_empty())
        .map(|p| p.graph.clone())
        .expect("a part with a body in it");
    let (lo, hi) = fieldform_field::part_extent(&doc, &graph).as_box().expect("finite bounds");
    let program = generate(&doc, &graph).expect("the graph generates");
    let floor = ((hi - lo).norm() * 1e-5).max(1e-4);
    let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
        .curves
        .iter()
        .map(|use_of| match use_of {
            CurveUse::Spine(id) => (doc.curves[id].tessellate_to_angle(1.0, floor), None),
            other => panic!("this measurement supplies spines only, and the graph wants {other:?}"),
        })
        .collect();
    let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0))
        .expect("a GPU, which is what is being measured");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    let shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
        .expect("the lighting pipeline builds");
    let camera = Camera::framing((lo, hi), f64::from(WIDTH) / f64::from(HEIGHT));
    let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);
    let budget = (512.0 * 2.0 * fieldform_field::step_multiplier(&doc, &graph)).ceil() as u32;
    let mut rig = Rig { renderer, shading, camera, look, budget };

    // Steady: back to back, the first few thrown away as the warm-up they are.
    for _ in 0..8 {
        rig.draft();
    }
    let mut steady: Vec<f32> = (0..24).map(|_| rig.draft()).collect();
    let steady = median(&mut steady);
    println!("{WIDTH} × {HEIGHT} draft, step budget {budget}. Steady median of 24: **{steady:.1} ms**.\n");
    if std::env::args().nth(2).as_deref() == Some("paced") {
        println!("| after the 5 s idle, one band a 16 ms frame | for | first draft, three runs | median | against steady | GPU time spent |");
        println!("|---|---|---|---|---|---|");
        for (label, rows) in [("a sixth of the draft", HEIGHT / 6), ("a quarter", HEIGHT / 4), ("a third", HEIGHT / 3)] {
            for warm in [250u64, 400] {
                let mut firsts = Vec::new();
                let mut kept = (0, 0.0);
                for _ in 0..3 {
                    for _ in 0..6 {
                        rig.draft();
                    }
                    rig.sit(IDLE, None);
                    kept = rig.paced(Duration::from_millis(warm), rows);
                    firsts.push(rig.draft());
                }
                let shown = firsts.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", ");
                let first = median(&mut firsts);
                println!(
                    "| {label} ({rows} rows) | {warm} ms | {shown} ms | **{first:.0} ms** | {:.2}× | {:.0} ms in {} bands |",
                    first / steady,
                    kept.1,
                    kept.0
                );
            }
        }
        return;
    }

    println!("| through the 5 s idle | then | first draft, three runs | median | against steady | drafts 2–4 (last run) | GPU time spent keeping warm |");
    println!("|---|---|---|---|---|---|---|");

    let ms = Duration::from_millis;
    let cases: [(&str, Option<(Duration, u32)>, &str, Option<Duration>); 9] = [
        ("nothing", None, "straight to the draft", None),
        ("8 rows every 250 ms", Some((ms(250), 8)), "straight to the draft", None),
        ("8 rows every 100 ms", Some((ms(100), 8)), "straight to the draft", None),
        ("8 rows every 33 ms", Some((ms(33), 8)), "straight to the draft", None),
        ("32 rows every 100 ms", Some((ms(100), 32)), "straight to the draft", None),
        ("32 rows every 33 ms", Some((ms(33), 32)), "straight to the draft", None),
        ("nothing", None, "32 rows every 16 ms for 100 ms", Some(ms(100))),
        ("nothing", None, "32 rows every 16 ms for 250 ms", Some(ms(250))),
        ("nothing", None, "32 rows every 16 ms for 500 ms", Some(ms(500))),
    ];
    for (idle, trickle, then, warm) in cases {
        let mut firsts = Vec::new();
        let mut after = Vec::new();
        let mut kept = (0, 0.0);
        for _ in 0..3 {
            // Each run starts from the same place: clocks up, then the idle.
            for _ in 0..6 {
                rig.draft();
            }
            let (ran, spent) = rig.sit(IDLE, trickle);
            kept = (ran, spent);
            if let Some(warm) = warm {
                let (ran, spent) = rig.sit(warm, Some((ms(16), 32)));
                kept = (ran, spent);
            }
            firsts.push(rig.draft());
            after = (0..3).map(|_| rig.draft()).collect();
        }
        let shown = firsts.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", ");
        let first = median(&mut firsts);
        println!(
            "| {idle} | {then} | {shown} ms | **{first:.0} ms** | {:.2}× | {} ms | {} |",
            first / steady,
            after.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", "),
            if kept.0 == 0 { "—".to_string() } else { format!("{:.0} ms in {} traces", kept.1, kept.0) },
        );
    }

    // A warm-up that saturates: the same 32 rows back to back with no gap, so the card sees a
    // full queue for the whole of it. What the light bursts above could not say is whether it
    // is *load* the clocks answer to or merely work.
    println!("
| after the 5 s idle, 32 rows back to back for | first draft, three runs | median | against steady | GPU time spent |");
    println!("|---|---|---|---|---|");
    for warm in [100u64, 200, 300, 400, 600] {
        let mut firsts = Vec::new();
        let mut kept = (0, 0.0);
        for _ in 0..3 {
            for _ in 0..6 {
                rig.draft();
            }
            rig.sit(IDLE, None);
            kept = rig.sit(ms(warm), Some((ms(0), 32)));
            firsts.push(rig.draft());
        }
        let shown = firsts.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", ");
        let first = median(&mut firsts);
        println!(
            "| {warm} ms | {shown} ms | **{first:.0} ms** | {:.2}× | {:.0} ms in {} traces |",
            first / steady,
            kept.1,
            kept.0
        );
    }

    // How long the clocks stay up once the work stops, which is how soon a warm-up is owed again.
    println!("
| idle after steady drafts | first draft, three runs | median | against steady |");
    println!("|---|---|---|---|");
    for idle in [250u64, 500, 1000, 2000, 3000] {
        let mut firsts = Vec::new();
        for _ in 0..3 {
            for _ in 0..6 {
                rig.draft();
            }
            rig.sit(ms(idle), None);
            firsts.push(rig.draft());
        }
        let shown = firsts.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", ");
        let first = median(&mut firsts);
        println!("| {idle} ms | {shown} ms | **{first:.0} ms** | {:.2}× |", first / steady);
    }

    // The third candidate: accept the slow clocks and draw less with them. The first draft
    // after the idle goes out some rungs down the ladder, and the ladder climbs from there.
    println!("
| first draft after the 5 s idle at | that draft, three runs | median | the 720 px drafts after it (last run) |");
    println!("|---|---|---|---|");
    for longest in [720u32, 480, 360, 240] {
        let mut firsts = Vec::new();
        let mut after = Vec::new();
        for _ in 0..3 {
            for _ in 0..6 {
                rig.draft();
            }
            rig.sit(IDLE, None);
            firsts.push(rig.draft_at(longest));
            after = (0..4).map(|_| rig.draft()).collect();
        }
        let shown = firsts.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", ");
        println!(
            "| {longest} px | {shown} ms | **{:.0} ms** | {} ms |",
            median(&mut firsts),
            after.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(", "),
        );
    }
}
