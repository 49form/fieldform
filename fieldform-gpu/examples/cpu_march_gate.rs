// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The GPU's march held to the CPU's: hit for hit, body for body, depth for depth.
//!
//! **What `gpu_matches_cpu` cannot see, and why this exists.** That gate asks, of every GPU hit,
//! whether the point lies on the CPU field's surface. A ray that tunnels through one wall and
//! stops on the next lies on a surface, so it passes. That is how V8's hop skipped its vouch
//! unseen from the day the grid and the over-relaxation were combined until a 2 mm shell showed
//! it (`tile-mask-design-01941c5.md` §1.7c). What catches a march that stops in the wrong place
//! is a march that stops in the right one: `fieldform_gpu::cpu_march` runs the same loop with the
//! same numbers and the same grid in `f32`, and this compares the two over whole frames.
//!
//! **The CPU marches with the GPU's own side tables, and the side tables are checked apart.** The
//! skip grid the replica steps by is read back from the renderer's fill, and the tile mask it
//! folds by is read back from the frame's own `tile_cull`, so a disagreement between the two
//! marches is the *march's* and nothing else's. Then each table is held to its own replica on the
//! CPU: every cell's clearance against `Grid::build`, and every tile's verdict on every body
//! against `ruled_out_on_the_cpu`'s five planes. The first run of the gate did neither -- it
//! built both on the CPU and marched with those, which held the GPU's loop to a grid and a mask
//! the shader never read, and could not have seen a plane oriented the wrong way (§1.7's first
//! fault) unless a body happened to stand behind it.
//!
//! **Two configurations, because the march has two.** Cull on, with each tile's own mask -- the
//! viewport's default -- and cull off. A fault in the vouch, the look-ahead, the refusal, or the
//! grid shows in both; a fault in the mask shows in one.
//!
//! **What counts as disagreement, and what is only precision.** The GPU evaluates the field in
//! `f32`; the replica evaluates it in `f64` and casts. On a grazing silhouette that lands a
//! candidate on opposite sides of the hit threshold, so one side hits and the other misses, or the
//! two hits sit a millimetre apart along a ray that is nearly in the surface. The first run of this
//! gate reported one to five such pixels on every frame -- **identical with the cull on and off,
//! every one of them a run of one** -- and the tunnelling it was built to catch was a run of
//! thirteen. So a disagreement is: a hit on a **different body**; or lost, invented or moved-past-
//! a-millimetre pixels in a **run of two or more** down a column; or such pixels anywhere in more
//! than one in ten thousand of the frame, which is a diffuse loss no run would show; or a tile
//! whose mask rules out a body the CPU's planes keep by more than `mask_tolerance_mm`; or a grid
//! cell the GPU reads clearer than the CPU by more than `GRID_TOLERANCE_MM`. Isolated pixels, and
//! table disagreements inside those tolerances, are *scatter*, counted and printed, and they do
//! not fail the gate.
//!
//! **The gate is made to cross its own branch.** `--unvouched` runs the CPU side with the loop as it
//! was before `dfdf029`, the hop that dropped its vouch. On the document that found the fault the
//! GPU now holds its hits and the replica loses them, and the gate must fail with a run -- which is
//! the arm that says the criterion above can see what it exists to see.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example cpu_march_gate [-- <doc.fform>…] [--cameras N] [--unvouched]`
//!
//! The built-in scenes run always; a document named on the command line runs as well. Pass the
//! document that showed a fault, as a user's own scene once did.

use fieldform_field::{GraphField, ScalarField};
use fieldform_gpu::cpu_march::{march, pixel_ray, ruled_out_on_the_cpu, scene_entry, Arm, Grid, Hop, V3};
use fieldform_gpu::{generate_for_measurement, Camera, Renderer, TileMaskReading};
use fieldform_scene::{BooleanOp, Document, Graph, Node, NodeId, ParamRef, Place};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 405;
const BUDGET: u32 = 512;
const GRID_CELLS: u32 = 64;

/// How far the GPU's mask may disagree with the CPU's planes before it is a fault rather than
/// precision. The five planes are built in `f32` from the camera on the GPU and in `f64` on the
/// CPU, and a body whose furthest corner sits within this of a plane may land on either side.
///
/// **It grows with the eye's distance from the origin, because the residual does.** A side plane
/// is the cross product of two corner rays a hundredth of a radian apart, so the rays' own `f32`
/// rounding (6e-8 of a unit vector) is divided by that angle, and the plane's error at the body
/// is that times the distance to it: `cull_plane_precision.rs` measures 0.003 mm with the eye
/// 647 mm out, 0.031 mm at 6.5 m, 0.31 mm at 65 m -- 4.8e-6 of the distance, every time. This is
/// four times that. The first run of the gate measured a plane 0.15 mm off at 647 mm, which was
/// not this residual but the second vector spelt `b.origin + b.dir - a.origin`; §1.7e.
///
/// **Why a disagreement under this cannot delete material:** the extents the planes test are
/// padded by `CULL_PAD_FLOOR_MM` at least (0.5 mm, `codegen.rs`), so the body itself is at least
/// that far inside the box whose corner the two sides place differently, and this stays under
/// the floor out to an eye 25 m from the origin. A body the GPU rules out that the CPU keeps by
/// *more* than this is a mask that deletes material.
fn mask_tolerance_mm(camera: &Camera) -> f64 {
    (camera.eye.coords.norm() * 2e-5).max(0.01)
}

/// How much clearer than the CPU the GPU may read a grid cell: the field in `f32` against the
/// field in `f64` cast once, at the cell's centre. Measured at 0.00003 mm on every built-in scene,
/// the lattice included; this is thirty times that. The other direction -- the GPU reading less
/// clearance -- costs a reading and never a hit, and is counted without failing.
const GRID_TOLERANCE_MM: f32 = 1e-3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The graph with every root but `keep` removed, so one body's field can be compiled on its own.
fn only_body(graph: &Graph, keep: NodeId) -> Graph {
    let mut out = graph.clone();
    for root in graph.roots() {
        if root == keep {
            continue;
        }
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(node) = out.node(id) {
                stack.extend(node.inputs());
                out.remove(id);
            }
        }
    }
    out
}

/// Dear bodies apart: four overlapping spheres each, on a grid.
fn spread(count: usize, pitch: f64, radius: f64) -> Graph {
    let mut graph = Graph::empty();
    let side = (count as f64).cbrt().ceil() as usize;
    for k in 0..count {
        let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
        let at = [x as f64 * pitch, y as f64 * pitch, z as f64 * pitch];
        let parts: Vec<_> = (0..4)
            .map(|j| {
                graph.add(Node::Sphere {
                    centre: Place::At([lit(at[0] + j as f64 * 0.5), lit(at[1]), lit(at[2])]),
                    radius: lit(radius),
                })
            })
            .collect();
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() });
    }
    graph
}

/// Thin shells apart: the geometry the fault was found in.
fn shells(count: usize, pitch: f64, radius: f64, wall: f64) -> Graph {
    let mut graph = Graph::empty();
    let side = (count as f64).cbrt().ceil() as usize;
    for k in 0..count {
        let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
        let at = [lit(x as f64 * pitch), lit(y as f64 * pitch), lit(z as f64 * pitch)];
        let outer = graph.add(Node::Sphere { centre: Place::At(at.clone()), radius: lit(radius) });
        let inner = graph.add(Node::Sphere { centre: Place::At(at), radius: lit(radius - wall) });
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![outer.into(), inner.into()] });
    }
    graph
}

/// Lattice-filled cubes apart, as `masked_matches_unmasked` measures them: lower-bound bodies, no
/// guard box, the cull their only skip -- and a gyroid is a thousand thin walls, so every frame
/// is silhouettes, which is where `f32` against `f64` scatters most. Here to say what that scatter
/// looks like on the hardest scene rather than only on spheres.
fn lattice_assembly(count: usize) -> Graph {
    use fieldform_scene::{CellMap, Frame, LatticeCell, LatticePhase};
    let mut graph = Graph::empty();
    let side = (count as f64).cbrt().ceil() as usize;
    let (pitch, half) = (120.0, 20.0);
    for k in 0..count {
        let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
        let at = [x as f64 * pitch, y as f64 * pitch, z as f64 * pitch];
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(at[0]), lit(at[1]), lit(at[2])]),
            size: [lit(half * 2.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let lattice = graph.add(Node::Lattice {
            cell: LatticeCell::Gyroid,
            map: CellMap::Rectangular { cell: lit(8.0) },
            phase: LatticePhase::sheet(lit(1.0)),
            frame: Frame::default(),
        });
        graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![cube.into(), lattice.into()] });
    }
    graph
}

/// The sweep's camera for one of its frames: a ring of eyes at a few elevations and distances,
/// orthographic on the odd turns.
fn camera_for(bounds: (Point3<f64>, Point3<f64>), turn: u32, turns: u32, lift: f64, away: f64) -> Camera {
    let centre = nalgebra::center(&bounds.0, &bounds.1);
    let scene_radius = (bounds.1 - bounds.0).norm() * 0.5;
    let a = turn as f64 * std::f64::consts::TAU / turns as f64;
    let radius = scene_radius * away;
    let eye = Point3::new(
        centre.x + radius * a.cos() * lift.cos(),
        centre.y + radius * a.sin() * lift.cos(),
        centre.z + radius * lift.sin(),
    );
    let mut camera =
        Camera::looking_at(eye, centre, Vector3::z(), (centre, scene_radius), 35.0, WIDTH as f64 / HEIGHT as f64);
    camera.start_at_scene = true;
    camera.orthographic = turn % 2 == 1;
    if camera.orthographic {
        camera.half_height = scene_radius;
    }
    camera
}

struct Verdict {
    lost: usize,
    gained: usize,
    moved: usize,
    owner: usize,
    jitter: usize,
    run: usize,
    at: (u32, u32),
    /// Tile-and-body pairs the GPU ruled out and the CPU's planes kept by more than
    /// `mask_tolerance_mm`: material the mask would delete. Fails.
    mask_wrong: usize,
    /// Pairs the two sides decide differently within the tolerance, either way: precision.
    mask_near: usize,
    /// Pairs the CPU rules out and the GPU keeps beyond the tolerance: a mask that skips less than
    /// it could, which draws right and is reported so a drift is seen.
    mask_timid: usize,
    /// The largest CPU margin on any pair the GPU ruled out, in millimetres -- the number
    /// `mask_tolerance_mm` is set against -- and the deepest the CPU ruled out a pair the GPU
    /// kept, so both directions of the disagreement are printed rather than only the one that
    /// fails.
    mask_margin: f64,
    timid_margin: f64,
}

/// One in ten thousand of a frame: 29 pixels at 720 by 405. The tunnelling run was 26 pixels in
/// one column; a diffuse loss of that size with no run would be a different fault, and this is
/// what sees it.
const SCATTER_ALLOWED: f64 = 1e-4;

impl Verdict {
    fn fails(&self) -> bool {
        let scattered = self.lost + self.gained + self.moved;
        self.owner > 0
            || self.run >= 2
            || scattered as f64 > SCATTER_ALLOWED * f64::from(WIDTH * HEIGHT)
            || self.mask_wrong > 0
    }
}

/// The GPU's mask against the CPU's planes, every tile, every body, for the frame `mask` was
/// recorded for.
fn hold_mask_to_the_planes(
    mask: &TileMaskReading,
    extents: &[(Point3<f64>, Point3<f64>)],
    camera: &Camera,
    v: &mut Verdict,
) {
    let tolerance = mask_tolerance_mm(camera);
    for ty in 0..mask.tiles.1 {
        for tx in 0..mask.tiles.0 {
            for (b, (lo, hi)) in extents.iter().enumerate() {
                let gpu = mask.ruled_out((tx, ty), b as u32);
                let (cpu, gaps) = ruled_out_on_the_cpu(camera, (WIDTH, HEIGHT), (tx, ty), *lo, *hi);
                if gpu == cpu {
                    continue;
                }
                // The deciding plane: the one the body's far corner is nearest to, which is the
                // one whose sign `f32` could have turned.
                let margin = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
                if margin.abs() > tolerance {
                    println!(
                        "    tile ({tx},{ty}) body {b}: GPU {}, CPU {} -- CPU gaps {:.4} {:.4} {:.4} {:.4} {:.4} mm; extent {:?} to {:?}",
                        if gpu { "ruled out" } else { "kept" },
                        if cpu { "ruled out" } else { "kept" },
                        gaps[0],
                        gaps[1],
                        gaps[2],
                        gaps[3],
                        gaps[4],
                        lo,
                        hi,
                    );
                }
                if gpu {
                    v.mask_margin = v.mask_margin.max(margin);
                    if margin > tolerance {
                        v.mask_wrong += 1;
                    } else {
                        v.mask_near += 1;
                    }
                } else {
                    v.timid_margin = v.timid_margin.max(-margin);
                    if -margin > tolerance {
                        v.mask_timid += 1;
                    } else {
                        v.mask_near += 1;
                    }
                }
            }
        }
    }
}

/// One frame, one configuration: the GPU's hits against the CPU march's, the CPU marching with
/// the GPU's own grid and -- culled -- the GPU's own mask for this very frame.
fn compare(
    renderer: &Renderer,
    bodies: &[GraphField],
    grid: &Grid,
    extents: &[(Point3<f64>, Point3<f64>)],
    camera: &Camera,
    cull: bool,
    hop: Hop,
    // The program's `SETTLES`, so the replica runs the loop the shader was generated with.
    settles: bool,
) -> Verdict {
    let mut camera = camera.clone();
    camera.cull = cull;
    let gpu = renderer.render(&camera, WIDTH, HEIGHT, BUDGET).expect("renders");
    // Straight after the render whose camera it was recorded for, and before anything else is
    // submitted: the buffer holds the last cull only.
    let mask = cull.then(|| renderer.read_tile_mask(WIDTH, HEIGHT).expect("reads the mask"));
    let epsilon = camera.epsilon as f32;
    let far = camera.far as f32;

    let mut v = Verdict {
        lost: 0,
        gained: 0,
        moved: 0,
        owner: 0,
        jitter: 0,
        run: 0,
        at: (0, 0),
        mask_wrong: 0,
        mask_near: 0,
        mask_timid: 0,
        mask_margin: 0.0,
        timid_margin: 0.0,
    };
    if let Some(mask) = &mask {
        hold_mask_to_the_planes(mask, extents, &camera, &mut v);
    }

    let rows: Vec<u32> = (0..HEIGHT).collect();
    let cpu: Vec<Vec<(Option<f32>, u32)>> = std::thread::scope(|s| {
        let handles: Vec<_> = rows
            .chunks(16)
            .map(|chunk| {
                let camera = &camera;
                let mask = mask.as_ref();
                s.spawn(move || {
                    let mut out = Vec::new();
                    for &y in chunk {
                        // Which bodies this row's tiles rule out, as the shader's own mask has
                        // it -- `load_tile_mask`, including its rule that an empty mask is no
                        // mask, which here changes nothing, since a mask that rules nothing out
                        // keeps every body.
                        let masks: Vec<Vec<bool>> = (0..WIDTH.div_ceil(8))
                            .map(|tx| {
                                (0..bodies.len())
                                    .map(|i| mask.is_some_and(|m| m.ruled_out((tx, y / 8), i as u32)))
                                    .collect()
                            })
                            .collect();
                        for x in 0..WIDTH {
                            let ruled = &masks[(x / 8) as usize];
                            let kept = |p: V3| -> f32 {
                                let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
                                bodies
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| !ruled[*i])
                                    .map(|(_, b)| b.distance(q) as f32)
                                    .fold(3.4e38f32, f32::min)
                            };
                            let (origin, dir) = pixel_ray(camera, (WIDTH, HEIGHT), x, y);
                            let start = if camera.start_at_scene {
                                scene_entry(camera, origin, dir)
                            } else {
                                Some(0.0)
                            };
                            let Some(start) = start else {
                                out.push((None, u32::MAX));
                                continue;
                            };
                            let arm = Arm { field: &kept, grid: Some(grid), relax: 1.4, hop, settles, exact: None };
                            let o = march(&arm, origin, dir, start, far, epsilon, BUDGET, false);
                            // The owner, as the fold assigns it: the kept body nearest at the hit.
                            let owner = o.hit.map_or(u32::MAX, |t| {
                                let p = origin + dir * t;
                                let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
                                bodies
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| !ruled[*i])
                                    .map(|(i, b)| (i as u32, b.distance(q) as f32))
                                    .fold((u32::MAX, 3.4e38f32), |best, (i, d)| if d < best.1 { (i, d) } else { best })
                                    .0
                            });
                            out.push((o.hit, owner));
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a row chunk")).collect()
    });
    let cpu: Vec<(Option<f32>, u32)> = cpu.into_iter().flatten().collect();

    let mut bad = vec![false; (WIDTH * HEIGHT) as usize];
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let g_hit = (g.depth >= 0.0).then_some(g.depth as f32);
        match (c.0, g_hit) {
            (Some(_), None) => {
                v.lost += 1;
                bad[i] = true;
            }
            (None, Some(_)) => {
                v.gained += 1;
                bad[i] = true;
            }
            (Some(a), Some(b)) => {
                if (a - b).abs() > 1.0 {
                    v.moved += 1;
                    bad[i] = true;
                } else if c.1 != g.body {
                    v.owner += 1;
                    bad[i] = true;
                } else if (a - b).abs() > 2.0 * epsilon {
                    v.jitter += 1;
                }
            }
            (None, None) => {}
        }
    }
    for x in 0..WIDTH {
        let mut run = 0usize;
        for y in 0..HEIGHT {
            if bad[(y * WIDTH + x) as usize] {
                run += 1;
                if run > v.run {
                    v.run = run;
                    v.at = (x, y + 1 - run as u32);
                }
            } else {
                run = 0;
            }
        }
    }
    v
}

/// The GPU's grid against the CPU's, cell for cell: how much clearer the GPU ever reads a cell
/// (the direction that can tunnel), how much less (the direction that only costs a reading), and
/// how many cells sit beyond `GRID_TOLERANCE_MM` either way.
struct GridDrift {
    gpu_clearer_by: f32,
    cpu_clearer_by: f32,
    beyond: usize,
}

fn hold_grid_to_the_field(gpu: &Grid, cpu: &Grid) -> GridDrift {
    let mut drift = GridDrift { gpu_clearer_by: 0.0, cpu_clearer_by: 0.0, beyond: 0 };
    for (g, c) in gpu.clear.iter().zip(&cpu.clear) {
        let d = g - c;
        drift.gpu_clearer_by = drift.gpu_clearer_by.max(d);
        drift.cpu_clearer_by = drift.cpu_clearer_by.max(-d);
        if d.abs() > GRID_TOLERANCE_MM {
            drift.beyond += 1;
        }
    }
    drift
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cameras: u32 = args
        .iter()
        .position(|a| a == "--cameras")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(6);
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--") && a.ends_with(".fform")).collect();
    let hop = if args.iter().any(|a| a == "--unvouched") { Hop::Unvouched } else { Hop::Vouched };

    let mut scenes: Vec<(String, Document, Graph)> = vec![
        ("10 spread spheres".to_string(), Document::new(), spread(10, 40.0, 10.0)),
        ("20 thin shells, 1 mm".to_string(), Document::new(), shells(20, 40.0, 10.0, 1.0)),
        ("8 shells, 2 mm on 100 mm, 300 apart".to_string(), Document::new(), shells(8, 300.0, 50.0, 2.0)),
        ("8 lattice cubes".to_string(), Document::new(), lattice_assembly(8)),
    ];
    for path in paths {
        let (doc, _) = fieldform_io::open(std::path::Path::new(path)).expect("opens");
        let graph = doc.parts_ordered().first().copied().expect("a part").graph.clone();
        scenes.push((path.clone(), doc, graph));
    }

    let mut shared: Option<(wgpu::Device, wgpu::Queue)> = None;
    let mut failures = 0usize;
    println!("| scene | camera | cull | lost | gained | moved | owner | jitter | longest run | mask wrong / near / timid | margin ruled out / kept |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    for (name, doc, graph) in &scenes {
        let program = generate_for_measurement(doc, graph, true, true).expect("generates");
        assert!(program.baked.is_empty(), "{name}: this gate has no atlas; the scene must be closed-form");
        let renderer = match &shared {
            None => match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
                    return;
                }
            },
            Some((device, queue)) => Renderer::with_device(device.clone(), queue.clone(), &program, &[], &[], &[])
                .expect("the renderer builds on the shared device"),
        };
        if shared.is_none() {
            shared = Some((renderer.device().clone(), renderer.queue().clone()));
        }
        let values: Vec<f32> =
            program.params.iter().map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32).collect();
        renderer.set_params(&values);

        let roots = doc.folded_bodies(graph);
        let bodies: Vec<GraphField> =
            roots.iter().map(|&r| GraphField::compile_graph(doc, &only_body(graph, r), 0.01).expect("compiles")).collect();
        let bounds = match fieldform_field::part_extent(doc, graph) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => panic!("{name}: unbounded; this gate wants a scene sphere"),
        };
        // The cull's own extents, as the shader reads them: six slots a body from `cull_base`.
        let base = program.cull_base.expect("two or more bodies cull");
        let slot = |b: usize, k: usize| -> f64 { f64::from(values[base + b * 6 + k]) };
        let extents: Vec<(Point3<f64>, Point3<f64>)> = (0..roots.len())
            .map(|b| (Point3::new(slot(b, 0), slot(b, 1), slot(b, 2)), Point3::new(slot(b, 3), slot(b, 4), slot(b, 5))))
            .collect();

        // The grid over the same cube the application fills: the GPU's fill read back, which is
        // the one the CPU marches with, and the CPU's own build, which is what the fill is held to.
        let centre = nalgebra::center(&bounds.0, &bounds.1);
        let scene_radius = (bounds.1 - bounds.0).norm() * 0.5;
        let reach = Vector3::repeat(scene_radius);
        renderer.build_skip_grid(centre - reach, scene_radius * 2.0, GRID_CELLS, true);
        renderer.device().poll(wgpu::PollType::wait_indefinitely()).expect("the grid fills");
        let reading = renderer.read_skip_grid(GRID_CELLS).expect("reads the grid");
        let grid = Grid::from_reading(centre - reach, scene_radius * 2.0, GRID_CELLS, reading.clearance);
        let all = |p: V3| -> f32 {
            let q = Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
            bodies.iter().map(|b| b.distance(q) as f32).fold(3.4e38f32, f32::min)
        };
        let built = Grid::build(centre - reach, scene_radius * 2.0, GRID_CELLS, &all);
        let drift = hold_grid_to_the_field(&grid, &built);
        let grid_fails = drift.gpu_clearer_by > GRID_TOLERANCE_MM;
        if grid_fails {
            failures += 1;
        }
        println!(
            "| {name} | grid {GRID_CELLS}³ | | | | | | | GPU clearer by {:.5} mm, CPU by {:.5}; {} cells beyond {GRID_TOLERANCE_MM}{} | | |",
            drift.gpu_clearer_by,
            drift.cpu_clearer_by,
            drift.beyond,
            if grid_fails { " **GPU OVER-READS**" } else { "" },
        );

        for turn in 0..cameras {
            let lift = if turn % 3 == 2 { 0.6 } else { 0.0 };
            let away = if turn % 2 == 0 { 2.4 } else { 1.3 };
            let camera = camera_for(bounds, turn, cameras, lift, away);
            for cull in [true, false] {
                let v = compare(&renderer, &bodies, &grid, &extents, &camera, cull, hop, program.settles);
                if v.fails() {
                    failures += 1;
                }
                println!(
                    "| {name} | {turn} {} | {} | {} | {} | {} | {} | {} | {} at ({},{}) | {} |",
                    if camera.orthographic { "ortho" } else { "persp" },
                    if cull { "on" } else { "off" },
                    v.lost,
                    v.gained,
                    v.moved,
                    v.owner,
                    v.jitter,
                    v.run,
                    v.at.0,
                    v.at.1,
                    if cull {
                        format!(
                            "{} / {} / {} | {:.5} / {:.5} mm{}",
                            v.mask_wrong,
                            v.mask_near,
                            v.mask_timid,
                            v.mask_margin,
                            v.timid_margin,
                            if v.mask_wrong > 0 { " **MASK DELETES**" } else { "" }
                        )
                    } else {
                        "– | –".to_string()
                    },
                );
            }
        }
    }

    println!();
    if hop == Hop::Unvouched {
        println!("control arm: the CPU marched the loop as it was before the hop vouched the step.");
    }
    if failures == 0 {
        println!(
            "the GPU's march agrees with the CPU's on every frame, marching the GPU's own grid and mask: \
             no body re-owned, no run of two, only scatter; and both tables agree with their replicas within tolerance."
        );
    } else {
        println!("** {failures} frame(s) or table(s) where the GPU disagrees with the CPU **");
        std::process::exit(1);
    }
}
