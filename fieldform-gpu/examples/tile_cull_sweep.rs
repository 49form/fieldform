// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Hunting a *sliver*: a run of pixels a tile column lost, as against scattered single pixels.
//!
//! `masked_matches_unmasked` reports one to three differing pixels and shows, on the CPU, that
//! none of them is a body the frustum ruled out — so they are `confirm_distance`'s look-ahead
//! landing elsewhere. Then a screenshot arrived with a **vertical bar** of background showing
//! through a sphere, five pixels wide and some fifty tall, at the edge where one sphere emerges
//! from behind another. That is not scatter. It is 8-pixel-quantised and coherent, which is the
//! shape of a tile column that ruled out a body it can see — the failure the other file says
//! cannot happen.
//!
//! One camera cannot find a rare thing, so this sweeps: the same scene from many angles and two
//! distances, and for every frame it measures **the longest contiguous run of differing pixels
//! down a column**, not the count. A run of one is the known residual; a run of eight is a tile.
//!
//! For the worst run found it then does what the other file does for a single pixel — runs
//! `tile_cull`'s own five planes on the CPU for the body the unmasked arm hit — which is what
//! separates "the frustum test is wrong" from "the march landed differently".
//!
//! The scene is the screenshot's: spheres that overlap on screen, so that one is a thin crescent
//! behind another's silhouette. That is where a ray grazes one body on its way to another, and
//! it is the geometry the reported artefact appeared in.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example tile_cull_sweep`

use fieldform_gpu::{generate_for_measurement, Camera, Renderer};
use fieldform_scene::{BooleanOp, Document, Graph, Node, ParamRef, Place};
use nalgebra::Point3;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// Spheres that overlap on screen from most angles: a cluster, not a grid. Each is a union of
/// four so it is dear enough to carry a guard box, as the reported scene's bodies were.
fn cluster() -> (Graph, (Point3<f64>, Point3<f64>)) {
    let mut graph = Graph::empty();
    let at: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [18.0, 4.0, 2.0],
        [9.0, 16.0, -3.0],
        [26.0, 18.0, 6.0],
        [-14.0, 10.0, 4.0],
        [6.0, -15.0, -6.0],
    ];
    for centre in at {
        let parts: Vec<_> = (0..4)
            .map(|j| {
                graph.add(Node::Sphere {
                    centre: Place::At([
                        lit(centre[0] + j as f64 * 0.4),
                        lit(centre[1]),
                        lit(centre[2]),
                    ]),
                    radius: lit(10.0),
                })
            })
            .collect();
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() });
    }
    (graph, (Point3::new(-26.0, -27.0, -18.0), Point3::new(38.0, 30.0, 18.0)))
}

/// `tile_cull`'s own five planes, on the CPU, for one tile and one body's six slots.
fn ruled_out_on_the_cpu(
    camera: &Camera,
    width: u32,
    height: u32,
    tile: (u32, u32),
    lo: Point3<f64>,
    hi: Point3<f64>,
) -> (bool, [f64; 5]) {
    const PAD: f64 = 1.0;
    let t = 8.0;
    let u0 = ((tile.0 as f64) * t - PAD) / width as f64 * 2.0 - 1.0;
    let u1 = (((tile.0 + 1) as f64) * t + PAD) / width as f64 * 2.0 - 1.0;
    let v0 = 1.0 - ((tile.1 as f64) * t - PAD) / height as f64 * 2.0;
    let v1 = 1.0 - (((tile.1 + 1) as f64) * t + PAD) / height as f64 * 2.0;

    let ray = |u: f64, v: f64| {
        let across = camera.right * (u * camera.half_height * camera.aspect)
            + camera.up * (v * camera.half_height);
        if camera.orthographic {
            (camera.eye + across, camera.forward.normalize())
        } else {
            (camera.eye, (camera.forward + across).normalize())
        }
    };
    let corner = [ray(u0, v0), ray(u1, v0), ray(u1, v1), ray(u0, v1)];
    let middle = ray((u0 + u1) * 0.5, (v0 + v1) * 0.5);

    let mut planes: Vec<(nalgebra::Vector3<f64>, Point3<f64>)> = Vec::new();
    for k in 0..4 {
        let a = corner[k];
        let b = corner[(k + 1) % 4];
        let mut n = a.1.cross(&((b.0 + b.1) - a.0));
        let away = corner[(k + 2) % 4];
        if n.dot(&((away.0 + away.1) - a.0)) < 0.0 {
            n = -n;
        }
        planes.push((n, a.0));
    }
    planes.push((middle.1, middle.0));

    let mut gaps = [0.0_f64; 5];
    let mut out = false;
    for (k, (n, through)) in planes.iter().enumerate() {
        let far = Point3::new(
            if n.x > 0.0 { hi.x } else { lo.x },
            if n.y > 0.0 { hi.y } else { lo.y },
            if n.z > 0.0 { hi.z } else { lo.z },
        );
        gaps[k] = n.dot(&(far - through)) / n.norm().max(1e-30);
        if gaps[k] < 0.0 {
            out = true;
        }
    }
    (out, gaps)
}

/// The same lattice-filled cubes `masked_matches_unmasked` measures, for `--lattice`: a lower-bound
/// body has no guard box, so the cull is the only skip it has, and a lattice reads *short*, which
/// is on the safe side of every vouch -- the expectation is scatter at most, and this is what
/// checks it rather than assuming it.
fn lattice_assembly(count: usize) -> (Graph, (Point3<f64>, Point3<f64>)) {
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
    let span = (side - 1) as f64 * pitch;
    (graph, (Point3::new(-half, -half, -half), Point3::new(span + half, span + half, span + half)))
}

fn main() {
    const WIDTH: u32 = 720;
    const HEIGHT: u32 = 405;
    // **A document named on the command line beats a synthetic scene**, and this is why: the
    // reported artefact would not appear in six solid spheres however many cameras were tried,
    // and the file that shows it is 2 mm shells on 100 mm spheres 300 mm apart, some of them
    // turned. A wall fifty times thinner than its body is not something a sweep guesses at.
    let named = std::env::args().nth(1).filter(|a| !a.starts_with("--"));
    let (doc, graph, bounds) = match &named {
        Some(path) => {
            let (doc, _) = fieldform_io::open(std::path::Path::new(path)).expect("opens");
            let graph = doc.parts_ordered().first().copied().expect("a part").graph.clone();
            let bounds = match fieldform_field::part_extent(&doc, &graph) {
                fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
                _ => panic!("this document is unbounded"),
            };
            println!("{path}");
            (doc, graph, bounds)
        }
        None if std::env::args().any(|a| a == "--lattice") => {
            let (graph, bounds) = lattice_assembly(8);
            println!("8 lattice cubes (synthetic)");
            (Document::new(), graph, bounds)
        }
        None => {
            let (graph, bounds) = cluster();
            (Document::new(), graph, bounds)
        }
    };
    // **Asked for by name, because the cull is off by default** -- it overestimates and the
    // relaxed step cannot take that (`cull_wanted`). This file is the instrument that found that,
    // so it has to compile the feature in whatever the environment says; a sweep that inherited
    // the default would quietly measure nothing and report a clean bill.
    let program = generate_for_measurement(&doc, &graph, true, true).expect("generates");
    assert!(program.baked.is_empty(), "this sweep has no atlas; the document must be closed-form");
    let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
            return;
        }
    };
    // Resolved through the document, not read off the slots: a named parameter is a value the
    // document holds, and reading the literal would silently draw a different model.
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let base = program
        .cull_base
        .expect("the document needs two or more bodies for there to be anything to cull");

    // **The skip grid, because the application always has one and `render` never did.** That is
    // the largest difference between this sweep and the viewport, and the artefact appears in the
    // viewport and not here. The grid is filled from the *unmasked* field -- `skip_build` is its
    // own entry point and never loads a mask -- and the march then reads the grid beside a masked
    // fold, which is the one place the two arrangements meet.
    //
    // 64 cells and copy masks on: `SKIP_CELLS_SETTLED` and the `true` in `traced/draw.rs`, so
    // this is the grid a settled still marches rather than a coarser one chosen for the sweep's
    // own sake.
    let centre_of = Point3::new(
        (bounds.0.x + bounds.1.x) * 0.5,
        (bounds.0.y + bounds.1.y) * 0.5,
        (bounds.0.z + bounds.1.z) * 0.5,
    );
    let scene_radius = (bounds.1 - bounds.0).norm() * 0.5;
    // **The over-relaxed step, as an arm.** The march steps `RELAX` times its reading, so it
    // overshoots on purpose. The cull makes the reading *larger* in the gaps between bodies --
    // only the near body is folded there -- so the same multiplier overshoots further. If that is
    // what loses the hit, relaxation one takes the runs away and nothing else needs to change.
    let plain = std::env::args().any(|a| a == "--plain");
    renderer.set_relaxation(if plain { 1.0 } else { 1.4 });

    if std::env::args().all(|a| a != "--no-grid") {
        let reach = nalgebra::Vector3::repeat(scene_radius);
        renderer.build_skip_grid(centre_of - reach, scene_radius * 2.0, 64, true);
        renderer
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the grid fills");
    }

    // A ring of eye positions at two elevations and two distances. Nothing clever: a rare thing
    // wants many tries, and each frame is a few milliseconds.
    let mut worst = (0usize, String::new(), 0u32, 0u32, 0u32);
    let mut frames = 0usize;
    let mut total_runs = 0usize;
    let mut deleted = 0usize;
    let centre = Point3::new(
        (bounds.0.x + bounds.1.x) * 0.5,
        (bounds.0.y + bounds.1.y) * 0.5,
        (bounds.0.z + bounds.1.z) * 0.5,
    );

    for turn in 0..48 {
        for &lift in &[-0.35_f64, 0.0, 0.3, 0.75] {
            // **Framed tight enough that the outer bodies hang off the edges.** Every reported
            // artefact is on a body the frame cuts, and the middle body -- fully inside -- is
            // clean in all three screenshots. A tile at the frame's edge covers pixels past the
            // image, so its frustum is built from coordinates outside [-1, 1]; this is what puts
            // those tiles under the sweep rather than leaving them to a screenshot.
            for &away in &[0.95_f64, 1.25, 2.2, 3.6] {
                let a = turn as f64 * std::f64::consts::TAU / 48.0;
                let radius = 64.0 * away;
                let eye = Point3::new(
                    centre.x + radius * a.cos() * lift.cos(),
                    centre.y + radius * a.sin() * lift.cos(),
                    centre.z + radius * lift.sin(),
                );
                let mut camera = Camera::looking_at(
                    eye,
                    centre,
                    nalgebra::Vector3::z(),
                    (centre, (bounds.1 - bounds.0).norm() * 0.5),
                    35.0,
                    WIDTH as f64 / HEIGHT as f64,
                );
                camera.start_at_scene = true;
                // Half the sweep orthographic: the second screenshot is an axis view with a
                // grid, and the frustum's corner rays are built differently under each
                // projection -- shared origin and fanning directions against shared direction
                // and spread origins.
                camera.orthographic = turn % 2 == 1;
                if camera.orthographic {
                    camera.half_height = (bounds.1 - bounds.0).norm() * 0.5;
                }

                let mut arms = Vec::new();
                for cull in [false, true] {
                    camera.cull = cull;
                    arms.push(renderer.render(&camera, WIDTH, HEIGHT, 512).expect("renders"));
                }
                frames += 1;

                // The longest run of differing pixels down one column. Down, not across: the
                // reported artefact was a vertical bar, and a run across would be the same
                // measurement rotated -- both are taken, and the longer is kept.
                // **The normal counts, and the first draft of this left it out.** The reported
                // artefact is a thin dark line lying *on* a lit surface, not background showing
                // through -- so the pixel is hit, by the right body, at the right depth, and
                // shaded from the wrong normal. A test on depth and owner alone cannot see that,
                // and duly reported runs of one pixel while the screenshots showed runs of thirty.
                let differs = |i: usize| {
                    let (a, b) = (&arms[0][i], &arms[1][i]);
                    if (a.depth >= 0.0) != (b.depth >= 0.0) || a.body != b.body {
                        return true;
                    }
                    if a.depth < 0.0 {
                        return false;
                    }
                    // A degree is far more than the carried normal's own 0.03 and far less than
                    // anything that would read as a mark.
                    a.normal.dot(&b.normal) < 0.9998
                };
                let mut longest = 0usize;
                let mut at = (0u32, 0u32);
                for x in 0..WIDTH {
                    let mut run = 0usize;
                    for y in 0..HEIGHT {
                        if differs((y * WIDTH + x) as usize) {
                            run += 1;
                            if run > longest {
                                longest = run;
                                at = (x, y + 1 - run as u32);
                            }
                        } else {
                            run = 0;
                        }
                    }
                }
                for y in 0..HEIGHT {
                    let mut run = 0usize;
                    for x in 0..WIDTH {
                        if differs((y * WIDTH + x) as usize) {
                            run += 1;
                            if run > longest {
                                longest = run;
                                at = (x + 1 - run as u32, y);
                            }
                        } else {
                            run = 0;
                        }
                    }
                }
                if longest > 0 {
                    total_runs += 1;
                }
                // Anything that could be a tile rather than scatter gets the CPU's verdict.
                if longest >= 4 {
                    let i = (at.1 * WIDTH + at.0) as usize;
                    let body = if arms[0][i].depth >= 0.0 { arms[0][i].body } else { arms[1][i].body };
                    let slot = |k: usize| match &program.params[base + body as usize * 6 + k] {
                        ParamRef::Literal(v) => *v,
                        other => unreachable!("a cull corner is a number, got {other:?}"),
                    };
                    let (out, gaps) = ruled_out_on_the_cpu(
                        &camera,
                        WIDTH,
                        HEIGHT,
                        (at.0 / 8, at.1 / 8),
                        Point3::new(slot(0), slot(1), slot(2)),
                        Point3::new(slot(3), slot(4), slot(5)),
                    );
                    if out {
                        deleted += 1;
                    }
                    // **What kind of difference, not just that there is one.** Hit against miss is
                    // material gone; a changed owner is the wrong body in front; a changed normal
                    // alone is the right surface shaded wrongly. They point at different
                    // subsystems, and a run length cannot tell them apart. The projection is
                    // printed for the same reason: the frustum's corner rays are built one way
                    // under perspective and another under orthographic.
                    let (u, c) = (&arms[0][i], &arms[1][i]);
                    let kind = if (u.depth >= 0.0) != (c.depth >= 0.0) {
                        if u.depth >= 0.0 { "LOST a hit" } else { "gained a hit" }
                    } else if u.body != c.body {
                        "changed owner"
                    } else {
                        "changed normal only"
                    };
                    // **The gaps are printed, not dropped.** They are the clearance from the box
                    // to each of the tile's five planes, and they are what settled this: 8 to
                    // 31 mm, nowhere near a boundary, so the frustum was keeping the body and the
                    // lost pixel had to be coming from somewhere else. A verdict of "kept" on its
                    // own would have left that as an assertion rather than a measurement.
                    println!(
                        "  turn {turn:2} lift {lift:+.2} away {away:.1} {}: run {longest} at ({},{}), \
                         {kind}, body {body}, depth {:.2} -> {:.2}, normal dot {:.4}, \
                         CPU says {} -- plane gaps {:.1} {:.1} {:.1} {:.1} {:.1} mm",
                        if camera.orthographic { "ortho" } else { "persp" },
                        at.0,
                        at.1,
                        u.depth,
                        c.depth,
                        u.normal.dot(&c.normal),
                        if out { "RULED OUT" } else { "kept" },
                        gaps[0], gaps[1], gaps[2], gaps[3], gaps[4],
                    );
                }
                if longest > worst.0 {
                    worst = (
                        longest,
                        format!("turn {turn} lift {lift:+.2} away {away:.1}"),
                        at.0,
                        at.1,
                        0,
                    );
                }
            }
        }
    }

    println!();
    println!();
    println!(
        "relaxation {}, skip grid {}",
        if plain { "1.0 (plain)" } else { "1.4 (shipped)" },
        if std::env::args().any(|a| a == "--no-grid") { "off" } else { "on, 64 cells" },
    );
    println!("{frames} frames; {total_runs} had any difference at all.");
    println!(
        "longest run {} pixels, at ({},{}) on {}",
        worst.0, worst.2, worst.3, worst.1
    );
    println!("{deleted} of the runs of 4 or more were a body the frustum ruled out.");
    if worst.0 >= 4 {
        println!("** a run of 4 or more is a sliver, not the look-ahead's scatter **");
        std::process::exit(1);
    }
}
