// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a thin extrude costs -- to build, to draw and to compile -- against a thick extrude of
//! the same solid.
//!
//! **The promise this checks.** The wall was built from fibres rather than from the tube
//! coordinates its spec first described, on a table of how *right* each was, with the claim
//! that it would be no slower to draw and that the claim would be measured once there was a
//! shader to measure. A wall is paid for in three places, so there are three tables.
//!
//! **Building it, on the CPU.** A wall's bare edges are found by cutting every piece against
//! every other, so the build grows as the square of the chain where everything else about a
//! wall grows in step with it: 3.9 ms at 256 runs and 30 at a thousand, for a wall about its
//! curve. A loop has no bare edges to find, and is flat.
//!
//! **One block of each kind.** An extrude, a revolve and a sweep, each on a wall and on the
//! closed outline of the same band, beside a small sphere so that the source is one nothing
//! else generates. Cold: 0.94, 0.53 and 0.90 s thin against 0.25, 0.22 and 0.86 thick, the first
//! carrying the device's start-up. The frames are a profile's: 23.9 ms against 23.9, 25.6
//! against 25.0 and 30.2 against 29.8 at 1440 pixels, with the repeat at 30.0.
//!
//! The sweep's two arms are one solid in one place, which takes arranging. A wall is carried
//! along its path by the point midway between its curve's ends and a profile by its centroid,
//! so the section is an S, where those are the same point. The `steps x` column is what each
//! is divided by, 1.092 and 1.090. With a wall carried by its curve's first point, as it was
//! until this table was used to measure the change, it is 1.188 -- and a different solid, so
//! that its frames say less than that number does: 12.0 to 13.5 ms at 720 pixels against the
//! 10.1 it is now, and nothing to see at 1440.
//!
//! **Drawing one, by how long its curve is.** The same band four ways -- the curve given a
//! thickness, as the application prepares it, which is baked; the same with the bake withheld;
//! the band's own outline as a closed profile, baked; and that with its bake withheld -- and
//! the whole circle as a loop, which is the wall's other branch. At 1440 pixels:
//!
//! - **A walked wall costs what a walked profile costs**, and a little less, since it walks
//!   its `n` runs where the outline of the same band has `2n + 2` edges: 120.9 ms against 155.0
//!   at 256 chords. That is not the construction's doing. Tube coordinates walk the same curve
//!   to find their nearest point.
//! - **And a thick extrude is never walked**, so until a wall could be baked a thin extrude on
//!   a long chain was five times the frame of the same solid drawn thick: 120.9 against 21.7.
//!   That was this example's first finding, and the owner ruled for the bake on it.
//! - **Baked, the thin column is flat and is the thick one's**: 23.0, 22.2, 22.3 and 22.2 ms
//!   from four chords to 256, against 22.2, 21.5, 21.8 and 21.7.
//!
//! **Compiling a union of them.** `extrude_spokes_cost`'s fan, each blade as a rectangle
//! extruded or as the line down its middle given the rectangle's width, which is the same solid
//! exactly -- and the thin fan twice, written out a call an operand and walked as a run.
//! Written out is what a thin extrude was when this was first run, and it is what
//! `union-compile-8fa06b7.md` measured on pipes: 6 s cold for two blades, 14 for three, 25 for
//! four, **178 for eight**. As a run (`wall_extrude_run_distance`) it is 2.1 to 2.4 s at every
//! count, and the frame at eight goes from 41.1 ms to 33.1, against 30.3 for eight thick.
//! `WALL_RUN_FLOOR` has the table and what it says about where a run should start.
//!
//! **The controls.** The last row of each GPU table is a scene already in it, built again: its
//! driver column is the disk cache answering, and its frame columns bound the run's noise.
//! One blade is a second control -- a lone extrude has no run, so its two thin arms are one
//! source. In the run these figures are from, with another session building beside it, the
//! repeats moved 15% at 720 pixels and 3% at 1440, which is why 1440 is the column quoted.
//!
//! **A driver figure is a compile only the first time.** A source this machine has compiled
//! comes back from the driver's own cache in about twenty milliseconds whatever it holds, so a
//! second run of this shows the frame columns again and nothing of the compile. And what the
//! cache keys on is what the driver is handed: a function added to the prelude that a source
//! never reaches leaves that source a hit, so a change has to be to code a shader *calls*
//! before its compile can be read again. The figures above are all first readings, from one
//! run, taken after walls were given their bake.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example wall_cost`
//! -- then `-- build` for the first table alone, which needs no GPU, `-- blocks` to stop after
//! the second, or `-- 1 2 4 8 16` to choose the fan's counts.

use fieldform_gpu::{generate, generate_with_run_floor, Camera, CurveUse, Renderer};
use fieldform_scene::{
    BooleanOp, Curve, CurveId, Document, Frame, Graph, Node, Outline, ParamRef, Section, Sweep,
    WallEnds, WallSide,
};
use nalgebra::{Point3, Vector3};

const FRAMES: u32 = 20;
const ROUNDS: u32 = 6;
/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const SKIP_CELLS_SETTLED: u32 = 64;
const SPINE_DEGREES: f64 = 1.0;
const TOLERANCE_MM: f64 = 0.5;

const HEIGHT_MM: f64 = 40.0;
/// The band the first table is about: so far round a circle, so thick.
const ARC_RADIUS_MM: f64 = 80.0;
const ARC_DEGREES: f64 = 270.0;
const ARC_THICKNESS_MM: f64 = 4.0;
/// `extrude_spokes_cost`'s blade, so the second table lines up with that one.
const INNER_MM: f64 = 20.0;
const OUTER_MM: f64 = 200.0;
const BLADE_WIDTH_MM: f64 = 12.0;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

struct Scene {
    label: String,
    doc: Document,
    graph: Graph,
    /// The bake withheld, so a thick extrude walks its outline as a wall walks its chain.
    walked: bool,
    /// A union's operands written out a call each rather than walked as a run, which is what a
    /// thin extrude was before it had a run of its own and is the arm the loop is read against.
    as_calls: bool,
}

impl Scene {
    fn program(&self) -> fieldform_gpu::ShaderProgram {
        if self.as_calls {
            generate_with_run_floor(&self.doc, &self.graph, usize::MAX)
        } else {
            generate(&self.doc, &self.graph)
        }
        .expect("generates")
    }
}

fn polyline(doc: &mut Document, points: Vec<Point3<f64>>, closed: bool) -> CurveId {
    let id = CurveId::new();
    doc.curves.insert(id, Curve::polyline(id, "curve", points, closed).expect("a polyline"));
    id
}

/// Taken both ways about its plane, so the solid is the same whichever way a plane faces: a
/// straight line has no plane of its own and takes one from a rule, and a rectangle takes its
/// from the way it winds.
fn extrude(outline: Outline) -> Node {
    Node::Extrude { rims: Default::default(),
        outline,
        sweep: Sweep::Normal,
        height: lit(HEIGHT_MM),
        start: lit(0.0),
        sides: fieldform_scene::Sides::Symmetric,
        draft: lit(0.0),
        corners: Default::default(),
    }
}

fn wall(curve: CurveId, thickness: f64) -> Outline {
    wall_to(curve, thickness, WallSide::Centre)
}

fn wall_to(curve: CurveId, thickness: f64, side: WallSide) -> Outline {
    Outline::Wall { curve, thickness: lit(thickness), side, ends: WallEnds::Square }
}

fn on_circle(radius: f64, degrees: f64) -> Point3<f64> {
    let (s, c) = degrees.to_radians().sin_cos();
    Point3::new(radius * c, radius * s, 0.0)
}

#[derive(Clone, Copy)]
enum Arm {
    /// The arc as a chain, given a thickness, prepared the way the application does: baked.
    Thin,
    /// The same, with the bake withheld, which is what a wall was before it had one.
    ThinWalked,
    /// The band's own outline as a closed profile, prepared the way the application does.
    Thick,
    /// The same, with the bake withheld.
    ThickWalked,
    /// The whole circle as a loop, given a thickness: the wall's other branch, baked.
    ThinRing,
}

/// One extrude of a band `runs` chords long.
///
/// The thick arms draw the band's outline -- out along the outer edge and back along the inner
/// -- so they are the same solid as the thin one to within the rounds a wall puts on the
/// outside of its corners, which at sixty-four chords are slivers.
fn band(runs: usize, arm: Arm) -> Scene {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let at = |k: usize, radius: f64| on_circle(radius, ARC_DEGREES * k as f64 / runs as f64);
    let (label, outline) = match arm {
        Arm::Thin | Arm::ThinWalked => {
            let chain = (0..=runs).map(|k| at(k, ARC_RADIUS_MM)).collect();
            (
                if matches!(arm, Arm::Thin) { "thin" } else { "thin, walked" },
                wall(polyline(&mut doc, chain, false), ARC_THICKNESS_MM),
            )
        }
        Arm::Thick | Arm::ThickWalked => {
            let half = 0.5 * ARC_THICKNESS_MM;
            let outline = (0..=runs)
                .map(|k| at(k, ARC_RADIUS_MM + half))
                .chain((0..=runs).rev().map(|k| at(k, ARC_RADIUS_MM - half)))
                .collect();
            (
                if matches!(arm, Arm::Thick) { "thick" } else { "thick, walked" },
                Outline::closed(polyline(&mut doc, outline, true)),
            )
        }
        Arm::ThinRing => {
            let ring =
                (0..runs).map(|k| on_circle(ARC_RADIUS_MM, 360.0 * k as f64 / runs as f64)).collect();
            ("thin, a closed ring", wall(polyline(&mut doc, ring, true), ARC_THICKNESS_MM))
        }
    };
    graph.add(extrude(outline));
    Scene {
        label: format!("{runs:>3} chords, {label}"),
        doc,
        graph,
        walked: matches!(arm, Arm::ThickWalked | Arm::ThinWalked),
        as_calls: false,
    }
}

/// How a fan's blades are made, and how its union reads them.
#[derive(Clone, Copy, PartialEq)]
enum Blade {
    /// A rectangle extruded. Four or more in a union are a run, and walked.
    Thick,
    /// The line down the rectangle's middle, given its width. Two or more are a run.
    Thin,
    /// The same thin blades, written out a call each.
    ThinAsCalls,
}

/// `n` blades on their own spokes, unioned: each a rectangle extruded, or the line down its
/// middle given the rectangle's width. **The same solid exactly**, so the frame columns compare
/// what a reading costs and nothing about what is drawn.
fn fan(n: usize, blade: Blade) -> Scene {
    let thin = blade != Blade::Thick;
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let half = 0.5 * BLADE_WIDTH_MM;
    let ids: Vec<_> = (0..n)
        .map(|i| {
            let turn = std::f64::consts::TAU * i as f64 / n as f64;
            let (c, s) = (turn.cos(), turn.sin());
            // Along the spoke by `r`, across it by `w`.
            let at = |r: f64, w: f64| Point3::new(r * c - w * s, r * s + w * c, 0.0);
            let outline = if thin {
                let line = polyline(&mut doc, vec![at(INNER_MM, 0.0), at(OUTER_MM, 0.0)], false);
                wall(line, BLADE_WIDTH_MM)
            } else {
                let corners = vec![
                    at(INNER_MM, -half),
                    at(OUTER_MM, -half),
                    at(OUTER_MM, half),
                    at(INNER_MM, half),
                ];
                Outline::closed(polyline(&mut doc, corners, true))
            };
            graph.add(extrude(outline))
        })
        .collect();
    if ids.len() > 1 {
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: ids.into_iter().map(Into::into).collect() });
    }
    Scene {
        label: format!(
            "{n:>2} blades, {}",
            match blade {
                Blade::Thick => "thick",
                Blade::Thin => "thin",
                Blade::ThinAsCalls => "thin, as calls",
            }
        ),
        doc,
        graph,
        walked: false,
        as_calls: blade == Blade::ThinAsCalls,
    }
}

/// How long the wall itself takes to build, which every rebuild pays before anything is drawn.
///
/// A wall's bare edges are found by cutting every piece's outline against every other piece,
/// so the work grows as the square of the chain's length where everything else about a wall
/// grows in step with it. The application builds the wall several times a rebuild -- once to
/// compile the field, once for the numbers the shader is told, once for each of the three
/// polylines it uploads -- and a rebuild is on the interface's own thread.
fn build_cost() {
    println!("Building the wall, on the CPU (ms, the least of five)");
    println!(
        "{:<8} {:>16} {:>16} {:>16}",
        "chords", "about the curve", "to the left", "a closed ring"
    );
    for runs in [4usize, 16, 64, 256, 1024] {
        let built = |closed: bool, side: WallSide| {
            let mut doc = Document::new();
            let mut graph = Graph::empty();
            let sweep = if closed { 360.0 } else { ARC_DEGREES };
            let count = if closed { runs } else { runs + 1 };
            let chain = (0..count)
                .map(|k| on_circle(ARC_RADIUS_MM, sweep * k as f64 / runs as f64))
                .collect();
            let curve = polyline(&mut doc, chain, closed);
            let id = graph.add(extrude(wall_to(curve, ARC_THICKNESS_MM, side)));
            let mut best = f64::MAX;
            for _ in 0..5 {
                let began = std::time::Instant::now();
                let wall = fieldform_field::outline_wall(&doc, &graph, id, SPINE_DEGREES)
                    .expect("a wall the kernel builds");
                best = best.min(began.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(wall);
            }
            best
        };
        println!(
            "{:<8} {:>16.3} {:>16.3} {:>16.3}",
            runs,
            built(false, WallSide::Centre),
            built(false, WallSide::Left),
            built(true, WallSide::Centre),
        );
    }
    println!();
}

/// Which of the three blocks that take an outline.
#[derive(Clone, Copy)]
enum Block {
    Extruded,
    Revolved,
    Swept,
}

/// One block on a wall, or on the closed outline of the same band, **with a small sphere
/// beside it in a union**. The sphere is there for the driver's column and nothing else: a
/// lone block's source is the same text whatever its numbers, so once this machine has drawn
/// one thin revolve the driver answers for every other from its cache, and its compile can
/// never be read again. A block beside a sphere is a source nothing else here generates.
fn block(kind: Block, thin: bool) -> Scene {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    // A band drawn as its curve, or as the outline a profile would need for the same band:
    // out along one edge and back along the other, each point moved half a thickness square
    // to the curve in the plane the section lies in, whose normal is `plane`.
    //
    // **Square to the curve, and not along one direction for the whole of it.** It was one
    // direction until the pivot was measured with this table, and that made the thick arm a
    // different solid: a band sheared along a fixed line is thinner wherever the curve runs
    // nearer that line, and on the half ring it pinched to nothing at one point. Two arms that
    // are not one solid cannot say what the method costs.
    let section = |doc: &mut Document, curve: Vec<Point3<f64>>, plane: Vector3<f64>| {
        if thin {
            wall(polyline(doc, curve, false), ARC_THICKNESS_MM)
        } else {
            let last = curve.len() - 1;
            let across: Vec<Vector3<f64>> = (0..=last)
                .map(|i| {
                    let along = curve[(i + 1).min(last)] - curve[i.saturating_sub(1)];
                    along.cross(&plane).normalize() * (0.5 * ARC_THICKNESS_MM)
                })
                .collect();
            let outline = curve
                .iter()
                .zip(&across)
                .map(|(p, half)| p + half)
                .chain(curve.iter().zip(&across).rev().map(|(p, half)| p - half))
                .collect();
            Outline::closed(polyline(doc, outline, true))
        }
    };
    let (label, node) = match kind {
        Block::Extruded => {
            let arc = (0..=64)
                .map(|k| on_circle(ARC_RADIUS_MM, ARC_DEGREES * k as f64 / 64.0))
                .collect();
            ("extrude", extrude(section(&mut doc, arc, Vector3::z())))
        }
        Block::Revolved => {
            // A silhouette in the half-plane y = 0 that swells and narrows, as a vase's does.
            let silhouette = (0..=64)
                .map(|k| {
                    let t = k as f64 / 64.0;
                    let r = 60.0 + 25.0 * (t * std::f64::consts::TAU).sin();
                    Point3::new(r, 0.0, -60.0 + 120.0 * t)
                })
                .collect();
            let outline = section(&mut doc, silhouette, Vector3::y());
            ("revolve", Node::Revolve { rims: Default::default(), outline, frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One })
        }
        Block::Swept => {
            // An S as the section, and a path that bends gently out of its plane.
            //
            // **An S because it is the same about its own middle**, so the point midway
            // between its ends, which a wall is carried by, is also the centroid of the
            // band's outline, which a profile is carried by. The two arms are then one solid
            // in one place and differ only in how it is read. A half ring, which this was,
            // is carried from two different points and is two solids.
            let s_curve = (0..=16)
                .map(|k| {
                    let s = -1.0 + k as f64 / 8.0;
                    Point3::new(15.0 * s, 8.0 * (s * std::f64::consts::PI).sin(), 0.0)
                })
                .collect();
            let outline = section(&mut doc, s_curve, Vector3::z());
            let path = (0..=32)
                .map(|k| {
                    let u = (60.0 * k as f64 / 32.0).to_radians();
                    Point3::new(200.0 * (1.0 - u.cos()), 20.0 * u, 200.0 * u.sin())
                })
                .collect();
            let spine = polyline(&mut doc, path, false);
            ("sweep", Node::Sweep { rims: Default::default(), outline, spine, section: Section::Square, law: Default::default() })
        }
    };
    let solid = graph.add(node);
    let beside = graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(-150.0), lit(-150.0), lit(0.0)]),
        radius: lit(6.0),
    });
    graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![solid.into(), beside.into()] });
    Scene {
        label: format!("{label}, {}", if thin { "thin" } else { "thick" }),
        doc,
        graph,
        walked: false,
        as_calls: false,
    }
}

fn main() {
    build_cost();
    // `-- build` for that table alone, which needs no GPU and takes a second.
    if std::env::args().any(|a| a == "build") {
        return;
    }
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("an adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: fieldform_gpu::required_limits(&adapter),
                ..Default::default()
            })
            .await
            .expect("a device")
    });
    println!("{}\n", device.adapter_info().name);

    println!("One block on a wall, by which block it is");
    let mut blocks = Vec::new();
    for kind in [Block::Extruded, Block::Revolved, Block::Swept] {
        blocks.push(block(kind, true));
        blocks.push(block(kind, false));
    }
    // The control: a scene already in the list, built again.
    blocks.push(Scene { label: "sweep, thin (again)".into(), ..block(Block::Swept, true) });
    // Framed on the sweep, which reaches furthest.
    measure(&device, &queue, &blocks, 4);
    // `-- blocks` for that table alone.
    if std::env::args().any(|a| a == "blocks") {
        return;
    }
    println!();

    // The fan's counts on the command line, so the longest compile is asked for rather than
    // sat through: `-- 1 2 4 8 16`.
    let asked: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let counts = if asked.is_empty() { vec![1, 2, 3, 4, 8] } else { asked };

    println!("One extrude, by how long its curve is");
    let mut bands = Vec::new();
    for runs in [4, 16, 64, 256] {
        for arm in [Arm::Thin, Arm::ThinWalked, Arm::Thick, Arm::ThickWalked, Arm::ThinRing] {
            bands.push(band(runs, arm));
        }
    }
    // The control: a scene already in the list, built again.
    bands.push(Scene { label: "256 chords, thin (again)".into(), ..band(256, Arm::Thin) });
    let widest = bands.len() - 2;
    measure(&device, &queue, &bands, widest);

    println!("\nA fan of blades in one union, by how many");
    let mut fans = Vec::new();
    for &n in &counts {
        fans.push(fan(n, Blade::Thick));
        fans.push(fan(n, Blade::ThinAsCalls));
        fans.push(fan(n, Blade::Thin));
    }
    let last = *counts.last().expect("at least one count");
    fans.push(Scene { label: format!("{last:>2} blades, thin (again)"), ..fan(last, Blade::Thin) });
    let widest = fans.len() - 1;
    measure(&device, &queue, &fans, widest);
}

/// Builds every scene on one device, times each at two sizes, and prints the table.
///
/// The arrangement is `mask_frame_cost`'s, arrived at after three faults that each made two arms
/// look further apart than they were: one device for every scene, the image size as the
/// outermost loop so no renderer reallocates between calls, the scenes walked forward then
/// backward so the running order does not settle on one, and the least of the rounds as the
/// reading. Every scene is framed on the same box, so the pixels are the same pixels.
fn measure(device: &wgpu::Device, queue: &wgpu::Queue, scenes: &[Scene], widest: usize) {
    let extent = match fieldform_field::part_extent(&scenes[widest].doc, &scenes[widest].graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("an extrude is bounded"),
    };
    let radius = (extent.1 - extent.0).norm() * 0.5;
    let centre = nalgebra::center(&extent.0, &extent.1);
    let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
    let reach = Vector3::repeat(radius);

    struct Built {
        points: usize,
        baked: usize,
        lines: usize,
        params: usize,
        compile: f64,
        renderer: Renderer,
        multiplier: f64,
    }
    let mut built: Vec<Built> = Vec::new();
    for scene in scenes {
        let program = scene.program();
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("every program carries the marker");
        let lines = program.source[cut..].lines().count();
        let curves = curves_for(scene, &program);

        let began = std::time::Instant::now();
        let renderer = match Renderer::with_device(
            device.clone(),
            queue.clone(),
            &program,
            &[],
            &curves,
            &[],
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: no GPU ({e})", scene.label);
                return;
            }
        };
        let compile = began.elapsed().as_secs_f64() * 1e3;
        // Said as it happens and not only in the table: a cold compile of a long union is
        // minutes with nothing on screen, and a run that is waiting looks like one that hung.
        eprintln!("  {} — built in {compile:.0} ms", scene.label.trim());

        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| match p {
                ParamRef::Literal(v) => *v as f32,
                _ => 0.0,
            })
            .collect();
        renderer.set_params(&values);
        renderer.build_skip_grid(centre - reach, radius * 2.0, SKIP_CELLS_SETTLED, true);

        built.push(Built {
            points: curves.iter().map(|(points, _)| points.len()).sum(),
            baked: curves.iter().filter(|(_, bake)| bake.is_some()).count(),
            lines,
            params: program.params.len(),
            compile,
            renderer,
            multiplier: fieldform_field::step_multiplier(&scene.doc, &scene.graph),
        });
    }

    let mut best = vec![[f64::MAX; 2]; built.len()];
    for (at, edge) in [(0usize, 720u32), (1, 1440)] {
        for b in &built {
            let camera = framed(eye, centre, radius, b.multiplier);
            let _ = b.renderer.render(&camera, edge, edge, budget_for(b.multiplier));
        }
        for round in 0..ROUNDS {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..built.len()).collect()
            } else {
                (0..built.len()).rev().collect()
            };
            for i in order {
                let b = &built[i];
                let budget = budget_for(b.multiplier);
                let camera = framed(eye, centre, radius, b.multiplier);
                let start = std::time::Instant::now();
                for _ in 0..FRAMES {
                    b.renderer.render(&camera, edge, edge, budget).expect("renders");
                }
                let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES);
                best[i][at] = best[i][at].min(ms);
            }
        }
    }

    // `steps x` is what the field is divided by to stay a bound, and so how many times the
    // steps a true distance would take: one for an extrude or a revolve, and for a sweep the
    // stretch of its section round the path's tightest bend.
    println!(
        "{:<30} {:>7} {:>6} {:>6} {:>7} {:>8} {:>10} {:>9} {:>10}",
        "scene", "points", "baked", "lines", "params", "steps x", "driver ms", "720px ms",
        "1440px ms"
    );
    for (i, (scene, b)) in scenes.iter().zip(&built).enumerate() {
        println!(
            "{:<30} {:>7} {:>6} {:>6} {:>7} {:>8.3} {:>10.0} {:>9.1} {:>10.1}",
            scene.label,
            b.points,
            b.baked,
            b.lines,
            b.params,
            b.multiplier,
            b.compile,
            best[i][0],
            best[i][1],
        );
    }
}

fn budget_for(multiplier: f64) -> u32 {
    ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
        .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET)
}

fn framed(eye: Point3<f64>, centre: Point3<f64>, radius: f64, multiplier: f64) -> Camera {
    let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
    camera.value_scale = multiplier;
    camera
}

/// A path as the viewport flattens one: by the angle it turns through, down to a floor.
fn spine_points(doc: &Document, id: CurveId) -> Vec<Point3<f64>> {
    doc.curves[&id].tessellate_to_angle(SPINE_DEGREES, 0.05)
}

/// What the shader asks for, prepared the way the application prepares it: a closed profile in
/// its own plane with its bake, and a wall as the three polylines `Wall::packed` writes.
fn curves_for(
    scene: &Scene,
    program: &fieldform_gpu::ShaderProgram,
) -> Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> {
    let (doc, graph) = (&scene.doc, &scene.graph);
    program
        .curves
        .iter()
        .map(|use_of| match *use_of {
            CurveUse::Profile(node) => {
                // Which curve, what is left out of its walk, and whether it is read about an
                // axis rather than in a plane of its own.
                let (curve, ignore, about_axis) = match graph.node(node) {
                    Some(Node::Extrude { outline, sweep, .. }) => {
                        (outline.curve(), sweep.source(), false)
                    }
                    Some(Node::Sweep { outline, spine, .. }) => {
                        (outline.curve(), Some(*spine), false)
                    }
                    Some(Node::Revolve { outline, frame, .. }) => {
                        (outline.curve(), frame.curve_source(), true)
                    }
                    _ => unreachable!("a profile belongs to an extrude, a revolve or a sweep"),
                };
                let world =
                    fieldform_field::profile_loop_turning(doc, curve, SPINE_DEGREES, ignore)
                        .expect("a closed loop")
                        .points;
                let built = if about_axis {
                    // Every revolve here is about a default frame, which is the world's own.
                    fieldform_field::Profile::about_axis(&world, |p| p).expect("clear of the axis")
                } else {
                    let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                    let rotation = fieldform_field::rotation_from_z(normal);
                    let into = move |p: Point3<f64>| {
                        Point3::from(
                            rotation.inverse_transform_vector(&(p.coords - origin.coords)),
                        )
                    };
                    fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM).expect("planar")
                };
                let points: Vec<Point3<f64>> =
                    built.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                let reads = fieldform_field::profile_reads_out_to(doc, graph, node);
                let (lo, hi) = built.bounds();
                let bake = fieldform_field::bake_cells((hi - lo).max(), TOLERANCE_MM, reads)
                    .filter(|_| !scene.walked)
                    .map(|cells| built.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                (points, bake)
            }
            // Baked by the rule the field compiles under, as the application uploads one, and
            // the grid rides on the chain. Withheld for the walked arm, like a profile's.
            CurveUse::Wall(node, part) => {
                let wall = if scene.walked {
                    fieldform_field::outline_wall(doc, graph, node, SPINE_DEGREES)
                } else {
                    fieldform_field::outline_wall_baked(
                        doc,
                        graph,
                        node,
                        SPINE_DEGREES,
                        TOLERANCE_MM,
                        0.0,
                    )
                }
                .expect("a wall the kernel builds");
                let grid = match part {
                    fieldform_field::WallPart::Chain => wall.bake().cloned(),
                    _ => None,
                };
                (wall.packed(part), grid)
            }
            // A sweep's path, and the frame along it, computed from the very points the path
            // goes up as so that the two bindings index the same samples.
            CurveUse::Spine(id) => (spine_points(doc, id), None),
            CurveUse::Frame(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(node) else {
                    unreachable!("a frame belongs to a sweep");
                };
                let frame = fieldform_field::sweep_frame(doc, outline, *spine, &spine_points(doc, *spine))
                    .expect("a spine with a direction");
                (frame, None)
            }
            ref other => unreachable!("these blocks ask for nothing else: {other:?}"),
        })
        .collect()
}
