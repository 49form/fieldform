// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where the driver's compile time goes when a union grows, and what it is superlinear *in*.
//!
//! **Because `compile_scaling` measured 120 seconds for 766 generated lines and 1.0 second for
//! 696** — seventy lines more, a hundred and twenty times the compile — and a count of generated
//! lines plainly is not the quantity the driver charges for. Two readings of that, wanting
//! opposite fixes: the union's fold is being emitted as one enormous function, so the fix is to
//! split it; or the driver inlines everything into the marcher anyway, so splitting cannot help
//! and the fix is to stop handing it `n` copies of the same walk.
//!
//! It is the second, and the multiplier is six. The whole of it is in
//! the union-compile-8fa06b7 design note; what follows is what this example measures and
//! what it read.
//!
//! **Every figure below is from before the loop was built, and the pipe rows no longer reproduce.**
//! `pipe_run_distance` now walks a union of four or more plain pipes instead of writing out a call
//! each, so this harness reads a *constant* 2 845 inlined lines and 45 loops at 4, 8, 16 and 32
//! pipes where the tables say 8 335 / 144 up to 59 631 / 1 124 — and `pipe_distance` ×6 rather than
//! ×96, which is the six copies of the composition with one walk in each instead of sixteen. Below
//! `RUN_FLOOR` nothing changed: two pipes still read 4 683 and 74.
//!
//! The tables are kept as they were measured, because they are what the diagnosis rests on.
//! `FIELDFORM_NO_UNION_LOOP=1` puts the old composition back and reproduces them **to within
//! 0.3%** — 8 359 against 8 335 at four pipes, 30 415 against 30 319 at sixteen. The prelude gained
//! two functions when the loop was built, and a figure counted over the whole source is not
//! expected to be bit-identical after that; the loops column, which the new functions do not touch,
//! is unchanged at every count.
//!
//! **The free half — what the driver is handed.** WGSL has no recursion, so the call graph is a
//! DAG and full inlining is well defined: the size of `march` once everything it calls is folded
//! in is the number to hold against the compile, and the count of source lines is not. `march`
//! reads the field at five places — the primary loop through `scene`, the occlusion's five
//! samples, the shadow's walk, the relaxed step's proof and the hit — and `field_normal` once. So
//! **every operand of the union is written into one function six times over**, and nothing in the
//! generated text says so.
//!
//! | arm | n | generated | biggest generated fn | inlined into `march` | loops | per operand |
//! |---|---|---|---|---|---|---|
//! | spheres | 1 → 32 | 686 → 847 | 173, and it is `march` | 1 026 → 2 528 | 4 → 4 | **48** |
//! | extrudes | 1 → 32 | 686 → 1 138 | 173, and it is `march` | 3 238 → 75 407 | 16 → 388 | **2 328** |
//! | pipes | 1 → 32 | 686 → 1 138 | 173, and it is `march` | 2 745 → 59 631 | 39 → 1 124 | **1 835** |
//! | one pipe patterned | 2 → 32 | 731 | 173, and it is `march` | 4 945 | 75 | **0** |
//!
//! Two things that settles. The fold is **not** one enormous function: `body_0` is 90 lines at
//! sixteen operands and the largest generated function at every size is the fixed `march`. And the
//! operands are **already** functions of their own — sixteen pipes generate 33 functions against
//! one pipe's 17 — which is why `OperandMasks`, whose whole method is a function per operand,
//! cannot and does not help.
//!
//! **The cold half — which stage grows.** `BuildCost` splits a build into the module (naga parsing
//! and validating) and the pipelines (naga's backend, then the driver), so this needs no new
//! stopwatch; naga's front end is timed here beside it, so `module` is accounted for rather than
//! assumed. Every row is a source this machine had never compiled. Measured on an **RTX A500
//! Laptop GPU on Vulkan, 2026-09-26**, milliseconds:
//!
//! | arm | n | inlined into `march` | loops | module | pipelines | again |
//! |---|---|---|---|---|---|---|
//! | spheres | 1 | 1 026 | 4 | 7 | 48 | 8 |
//! | spheres | 16 | 1 760 | 4 | 9 | 139 | 9 |
//! | spheres | 32 | 2 528 | 4 | 8 | **272** | 10 |
//! | one pipe patterned | any count | 4 945 | 75 | 7 | **680** | 24 |
//! | extrudes | 1 | 3 238 | 16 | 17 | 583 | 14 |
//! | extrudes | 4 | 10 307 | 52 | 18 | 4 365 | 25 |
//! | extrudes | 8 | 19 607 | 100 | 19 | 11 733 | 24 |
//! | extrudes | 16 | 38 207 | 196 | 16 | **87 036** | 19 |
//! | pipes | 1 | 2 745 | 39 | 8 | 308 | 11 |
//! | pipes | 2 | 4 671 | 74 | 9 | 645 | 11 |
//! | pipes | 4 | 8 335 | 144 | 8 | 4 920 | 10 |
//! | pipes | 8 | 15 663 | 284 | 7 | 18 900 | 11 |
//! | pipes | 16 | 30 319 | 564 | 9 | **122 931** | 22 |
//! | pipes as 16 bodies | 16 | 32 206 | 564 | 17 | **119 588** | 20 |
//!
//! **Where the time is, to the stage.** `module` is 7–19 ms at every size, and naga's own parse and
//! validation add up to the same 8–15 ms, of which validation is 1 — so the front end is flat and
//! is not it. Whatever is left is in `create_compute_pipeline`, and the `again` column splits
//! *that* without a stopwatch: wgpu is passed `cache: None`, so the second reading runs naga's
//! backend again from scratch and only the driver has a cache of its own. `again` is 10–32 ms on
//! every row, sixteen pipes included. So naga's backend is under about 30 ms for a shader that
//! inlines to 30 000 lines, and **the 123 seconds is the driver's own compile.**
//!
//! **What it is superlinear in.** Not the operand count: thirty-two spheres are 2 528 inlined lines
//! and 280 ms, while *one* pipe is 2 745 inlined lines and 316 ms — two unrelated arms, the same
//! inlined size, the same cost. It is the inlined size, at a power of about 2 to 2.5 (pipes 2 → 16
//! is 189x on 8x the operands; a replication at 3 → 14 is 21.7x on 4.7x), and each kind of leaf has
//! its own price per copy — a spine walk being by far the dearest, which `wing_compile_cost` had
//! already priced at about 720 ms a walk against 55 for a profile reading. **Treat anything under
//! about 2x as noise**: the replication's n = 6 row reads 34 s where n = 8 above reads 18.9, and
//! `wing_compile_cost` records a 4.5x swing on an unchanged row.
//!
//! **And the way out, measured.** A radial pattern of one pipe draws the sixteen spokes the union
//! draws, takes the same sixteen walks a reading, and compiles in **687 ms against 122 931** —
//! because its count is a buffer value and its subtree is emitted once, behind a loop. That is
//! backlog item 34's fix, one level up: it was made for a pattern's window, and a boolean's operand
//! list never got it.
//!
//! Five arms, so that "n operands", "n curve walks" and "n copies of a walk" cannot be confused:
//!
//! - **spheres** — `n` operands that walk nothing. The control: if this blows up, it is the count.
//! - **extrudes** — `n` operands that each walk a closed profile.
//! - **pipes** — `n` operands that each walk a spine. The shape that costs 120 s.
//! - **pipes as n bodies** — the same pipes with the boolean removed, so the fold is split across
//!   `n` functions of one operand each. The direct answer to "would splitting it help": no, and it
//!   is 753 / 5 914 / 29 575 / 119 588 ms against the union's 645 / 4 920 / 18 900 / 122 931.
//! - **one pipe patterned** — the *same* geometry as the spokes and the same `n` walks a reading,
//!   emitted as a loop over one walk instead of `n` calls.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example compile_where_the_time_goes`, and
//! `-- 3 6 10 14` for a replication at sizes this machine has not compiled before. It takes several
//! minutes on its first run, because of the very cost it is measuring — and about twenty seconds on
//! every run after that, which is the driver's cache and means nothing.
use fieldform_scene::{BooleanOp, Curve, CurveId, Document, Graph, Node, NodeId, ParamRef, Place};
use nalgebra::Point3;

fn main() {
    let kinds: Vec<(&str, fn(usize) -> (Document, Graph))> = vec![
        ("spheres", spheres),
        ("extrudes", extrudes),
        ("pipes", spokes),
        ("pipes as n bodies", spokes_as_bodies),
        ("one pipe patterned", patterned),
    ];
    for (label, build) in kinds {
        println!("=== {label} ===");
        println!(
            "  {:>3} {:>7} {:>8} {:>5} {:>8} {:>10} {:>10} {:>9}",
            "n", "lines", "biggest", "fns", "in body", "march in", "skip in", "loops in"
        );
        for n in [1usize, 2, 4, 8, 16, 32] {
            let (doc, graph) = build(n);
            let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
            let generated = program.field_function();
            let fns = functions(generated);
            let biggest = fns.iter().map(|(_, n)| *n).max().unwrap_or(0);
            let whole = parse(&program.source);
            let (march_lines, march_loops) = inlined(&whole, "march");
            let (skip_lines, _) = inlined(&whole, "skip_build");
            println!(
                "  {n:>3} {:>7} {:>8} {:>5} {:>8} {:>10} {:>10} {:>9}",
                generated.lines().count(),
                biggest,
                fns.len(),
                // Through `functions` rather than a counter of its own, so this and the
                // largest-function line below cannot report two sizes for one function.
                fns.iter()
                    .find(|(name, _)| name == "body_0")
                    .map_or(0, |(_, n)| *n),
                march_lines,
                skip_lines,
                march_loops,
            );
        }
        // The shape of the largest case, function by function, since the driver charges per
        // function and a table of totals cannot say which one it is charging for.
        let (doc, graph) = build(16);
        let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
        let mut fns = functions(program.field_function());
        fns.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        print!("  at 16, the four largest generated functions:");
        for (name, n) in fns.iter().take(4) {
            print!(" {name} {n},");
        }
        println!();
        // And the multiplier: how many copies of the composition, and of one operand's own
        // walk, the driver is handed once it has inlined `march`.
        let whole = parse(&program.source);
        let reach = reached(&whole, "march");
        print!("  at 16, copies inside march after inlining:");
        for name in [
            "field",
            "field_here",
            "field_normal",
            "body_0",
            "body_normal_0",
        ] {
            if let Some(n) = reach.get(name) {
                print!(" {name} x{n},");
            }
        }
        for name in [
            "pipe_distance",
            "extrude_distance",
            "sphere_distance",
            "nearest_segment",
            "profile_distance",
        ] {
            if let Some(n) = reach.get(name) {
                print!(" {name} x{n},");
            }
        }
        println!();
        println!();
    }

    // **The control the pattern arm needs.** Its rows above are flat, and there are two reasons
    // that could be: the count is a buffer value and the source never changes, or the source
    // changes and the measurement is blind to it. Said out loud, since a flat row is exactly what
    // a broken arm looks like.
    let two = fieldform_gpu::generate(&patterned(2).0, &patterned(2).1).expect("generates");
    let many = fieldform_gpu::generate(&patterned(32).0, &patterned(32).1).expect("generates");
    println!(
        "control: a pattern of 2 and a pattern of 32 generate {} source, and {} params\n",
        if two.source == many.source {
            "byte-identical"
        } else {
            "DIFFERENT"
        },
        if two.params.len() == many.params.len() {
            "the same"
        } else {
            "different"
        },
    );

    // ------------------------------------------------------------------------------------------
    // The cold half. One device, every row a source the machine has never compiled, and each row
    // read twice: the first reading is the cost, the second is the control.
    // ------------------------------------------------------------------------------------------
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
    let info = device.adapter_info();
    println!("{} on {:?}\n", info.name, info.backend);
    println!("which stage grows, cold, and the same source again straight after");
    println!(
        "  {:<20} {:>3} {:>9} {:>6} {:>6} {:>6} {:>7} {:>8} {:>8} {:>6}",
        "arm", "n", "march in", "loops", "parse", "valid", "module", "pipes", "total", "again"
    );
    // **Counts from the command line**, because a reading is only cold once: the driver serves a
    // source it has compiled before from its own cache in about twenty milliseconds. So a
    // replication is taken at sizes this machine has never seen -- `... -- 3 6 10 14` -- rather
    // than by editing the example and losing the series that was already measured.
    let asked: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let counts = if asked.is_empty() {
        vec![1, 2, 4, 8, 16]
    } else {
        asked
    };
    let mut spheres_counts = counts.clone();
    if counts == vec![1, 2, 4, 8, 16] {
        spheres_counts.push(32);
    }
    let kinds: Vec<(&str, fn(usize) -> (Document, Graph), Vec<usize>)> = vec![
        ("spheres", spheres, spheres_counts),
        ("one pipe patterned", patterned, vec![counts[0].max(2)]),
        ("extrudes", extrudes, counts.clone()),
        ("pipes", spokes, counts.clone()),
        ("pipes as n bodies", spokes_as_bodies, counts.clone()),
    ];
    for (label, build, counts) in kinds {
        for n in counts {
            let (doc, graph) = build(n);
            let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
            let whole = parse(&program.source);
            let (march_lines, march_loops) = inlined(&whole, "march");
            // **naga's front end on its own.** `module` below is wgpu's `create_shader_module`,
            // which parses and validates the WGSL with exactly these two calls -- so timing them
            // here says whether `module` is accounted for, rather than leaving it a figure with
            // no parts. Whatever `pipes` then holds is naga's *backend* and the driver.
            let t = std::time::Instant::now();
            let parsed = naga::front::wgsl::parse_str(&program.source).expect("parses");
            let parse_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let _ = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&parsed)
            .expect("validates");
            let valid_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut readings = Vec::new();
            for _ in 0..2 {
                let t = std::time::Instant::now();
                let built = fieldform_gpu::Renderer::with_device(
                    device.clone(),
                    queue.clone(),
                    &program,
                    &[],
                    &[],
                    &[],
                )
                .expect("the driver builds it");
                let total = t.elapsed().as_secs_f64() * 1e3;
                readings.push((built.build_cost(), total));
            }
            let (cost, total) = readings[0];
            println!(
                "  {label:<20} {n:>3} {march_lines:>9} {march_loops:>6} {parse_ms:>6.0} {valid_ms:>6.0} {:>7.0} {:>8.0} {:>8.0} {:>6.0}",
                cost.module_ms, cost.pipelines_ms, total, readings[1].1,
            );
        }
    }
    println!(
        "\nEvery figure is milliseconds. `parse` and `valid` are naga's own front end, timed here; \
         `module` is wgpu's `create_shader_module`, which is those same two calls, so the two \
         columns should agree and `module` is then accounted for rather than assumed. `pipes` is \
         naga's backend and then the driver turning that module into the three compute pipelines.\n\
         \n\
         `again` is the same source compiled a second time on the same device, straight after, and \
         it is the control that splits `pipes` without a stopwatch: wgpu is passed `cache: None`, \
         so the second reading runs naga's backend again from scratch and only the driver has a \
         cache of its own. Whatever `again` does not charge for is the driver's; whatever it \
         charges for both times is work no cache can avoid."
    );
}

/// Every generated function and how many lines it spans.
fn functions(generated: &str) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    let mut current: Option<(String, usize)> = None;
    for line in generated.lines() {
        if let Some(rest) = line.strip_prefix("fn ") {
            if let Some(done) = current.take() {
                out.push(done);
            }
            current = Some((rest.split('(').next().unwrap_or("?").to_string(), 0));
        }
        if let Some((_, n)) = current.as_mut() {
            *n += 1;
        }
    }
    out.extend(current);
    out
}

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// `n` spheres in one union — operands that do no walking at all, which is the control for
/// "is it the count of operands or the count of curve walks".
fn spheres(n: usize) -> (Document, Graph) {
    let mut g = Graph::empty();
    let inputs: Vec<NodeId> = (0..n)
        .map(|i| {
            g.add(Node::Sphere {
                centre: Place::At([lit(i as f64 * 30.0), lit(0.0), lit(0.0)]),
                radius: lit(20.0),
            })
        })
        .collect();
    if inputs.len() > 1 {
        g.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
    }
    (Document::new(), g)
}

/// `n` extrudes in one union — each walks a closed profile, so this is a curve walk that is not
/// a pipe's.
fn extrudes(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut g = Graph::empty();
    let mut inputs = Vec::new();
    for i in 0..n {
        let id = CurveId::new();
        let x = i as f64 * 80.0;
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "section",
                vec![
                    Point3::new(x, -15.0, 0.0),
                    Point3::new(x + 30.0, -15.0, 0.0),
                    Point3::new(x + 30.0, 15.0, 0.0),
                    Point3::new(x, 15.0, 0.0),
                ],
                true,
            )
            .expect("a rectangle"),
        );
        inputs.push(g.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(40.0),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        }));
    }
    if inputs.len() > 1 {
        g.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
    }
    (doc, g)
}

/// The same shape `compile_scaling` measures: `n` bent pipes on their own spokes, one union.
fn spokes(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let mut pipes = Vec::new();
    for i in 0..n {
        let turn = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (turn.cos(), turn.sin());
        let id = CurveId::new();
        let at = |r: f64, z: f64| Point3::new(r * c, r * s, z);
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "spine",
                vec![
                    at(20.0, 0.0),
                    at(80.0, 25.0),
                    at(140.0, -10.0),
                    at(200.0, 15.0),
                ],
                false,
            )
            .expect("a bent line"),
        );
        pipes.push(graph.add(Node::Pipe {
            curve: id,
            radius: lit(9.0),
        }));
    }
    if pipes.len() > 1 {
        graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: pipes.into_iter().map(Into::into).collect(),
        });
    }
    (doc, graph)
}

// ---------------------------------------------------------------------------------------------
// What the driver actually sees, which is not what the text says.
//
// A count of generated lines is a count of *source*, and the driver compiles entry points, not
// source. WGSL has no recursion, so the call graph is a DAG and full inlining is well defined:
// the size of `march` after inlining is the sum, over every path from `march` down, of the lines
// each callee contributes once per call site. That is the number to hold against the compile.
// ---------------------------------------------------------------------------------------------

/// One WGSL function: its name, its own lines, the calls it makes (with repeats), and how many
/// loop headers it contains.
struct Fun {
    lines: usize,
    calls: Vec<String>,
    loops: usize,
}

/// Every function in a whole WGSL source, by name.
fn parse(source: &str) -> std::collections::HashMap<String, Fun> {
    let mut out = std::collections::HashMap::new();
    let mut name: Option<String> = None;
    let mut fun = Fun {
        lines: 0,
        calls: Vec::new(),
        loops: 0,
    };
    let known: Vec<String> = source
        .lines()
        .filter_map(|l| l.strip_prefix("fn "))
        .map(|r| r.split('(').next().unwrap_or("?").to_string())
        .collect();
    for line in source.lines() {
        if let Some(rest) = line.strip_prefix("fn ") {
            if let Some(done) = name.take() {
                out.insert(
                    done,
                    std::mem::replace(
                        &mut fun,
                        Fun {
                            lines: 0,
                            calls: Vec::new(),
                            loops: 0,
                        },
                    ),
                );
            }
            name = Some(rest.split('(').next().unwrap_or("?").to_string());
        }
        if name.is_some() {
            fun.lines += 1;
            // **Comments first, or the multiplier is wrong.** `march.wgsl` explains the skip
            // grid's bound with `field(c) - h` in prose, and a counter that reads that as a call
            // site inflates exactly the number this example exists to report. Whole-line comments
            // are dropped and a trailing one is cut; the prelude has no block comments.
            let line = match line.find("//") {
                Some(at) => &line[..at],
                None => line,
            };
            let t = line.trim_start();
            if t.starts_with("for ")
                || t.starts_with("for(")
                || t.starts_with("loop ")
                || t.starts_with("loop{")
                || t.starts_with("while ")
            {
                fun.loops += 1;
            }
            for callee in &known {
                // Whole-identifier match followed by an open bracket, so `field` never counts
                // `field_here` and a declaration never counts as a call.
                let mut from = 0usize;
                while let Some(at) = line[from..].find(callee.as_str()) {
                    let at = from + at;
                    from = at + callee.len();
                    let before_ok = at == 0
                        || !line[..at]
                            .chars()
                            .next_back()
                            .is_some_and(|c| c.is_alphanumeric() || c == '_');
                    let after_ok = line[from..].starts_with('(');
                    if before_ok && after_ok && !line.trim_start().starts_with("fn ") {
                        fun.calls.push(callee.clone());
                    }
                }
            }
        }
    }
    if let Some(done) = name {
        out.insert(done, fun);
    }
    out
}

/// Lines and loop headers an entry point contains once everything it calls is inlined.
fn inlined(funs: &std::collections::HashMap<String, Fun>, entry: &str) -> (usize, usize) {
    fn walk(
        funs: &std::collections::HashMap<String, Fun>,
        name: &str,
        memo: &mut std::collections::HashMap<String, (usize, usize)>,
        depth: usize,
    ) -> (usize, usize) {
        if depth > 64 {
            return (0, 0);
        }
        if let Some(hit) = memo.get(name) {
            return *hit;
        }
        let Some(f) = funs.get(name) else {
            return (0, 0);
        };
        let mut lines = f.lines;
        let mut loops = f.loops;
        for callee in &f.calls {
            if callee == name {
                continue;
            }
            let (l, o) = walk(funs, callee, memo, depth + 1);
            lines += l;
            loops += o;
        }
        memo.insert(name.to_string(), (lines, loops));
        (lines, loops)
    }
    walk(funs, entry, &mut std::collections::HashMap::new(), 0)
}

/// How many times each named function ends up in `entry` once everything is inlined -- the
/// multiplier a count of source lines cannot show.
fn reached(
    funs: &std::collections::HashMap<String, Fun>,
    entry: &str,
) -> std::collections::HashMap<String, usize> {
    fn walk(
        funs: &std::collections::HashMap<String, Fun>,
        name: &str,
        times: usize,
        out: &mut std::collections::HashMap<String, usize>,
        depth: usize,
    ) {
        if depth > 64 {
            return;
        }
        let Some(f) = funs.get(name) else { return };
        for callee in &f.calls {
            if callee == name {
                continue;
            }
            *out.entry(callee.clone()).or_insert(0) += times;
            walk(funs, callee, times, out, depth + 1);
        }
    }
    let mut out = std::collections::HashMap::new();
    walk(funs, entry, 1, &mut out, 0);
    out
}

/// One pipe turned into `n` by a radial pattern -- the same geometry the spokes make, and the
/// same `n` walks a reading costs, but emitted as a **loop** rather than as `n` calls. The arm
/// that separates "the driver is handed n walks" from "the driver is handed n *copies* of a
/// walk".
fn patterned(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut g = Graph::empty();
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::polyline(
            id,
            "spine",
            vec![
                Point3::new(20.0, 0.0, 0.0),
                Point3::new(80.0, 0.0, 25.0),
                Point3::new(140.0, 0.0, -10.0),
                Point3::new(200.0, 0.0, 15.0),
            ],
            false,
        )
        .expect("a bent line"),
    );
    let pipe = g.add(Node::Pipe {
        curve: id,
        radius: lit(9.0),
    });
    if n > 1 {
        g.add(Node::Pattern {
            options: Default::default(),
            input: pipe.into(),
            count: lit(n as f64),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::default(),
            },
            join: true,
            fillet: fieldform_scene::Fillet {
                radius: lit(0.0),
                continuity: fieldform_scene::Continuity::G1,
                second: None,
            },
        });
    }
    (doc, g)
}

/// The same `n` pipes, left as `n` **bodies** rather than gathered into one boolean -- so the
/// composition is folded across `n` functions of one operand each instead of one function that
/// names all `n`.
///
/// This is the arm that answers "would splitting the fold into several smaller functions flatten
/// it" directly, rather than by argument: `split_into_bodies` already emits a function per root,
/// so asking for `n` roots asks for exactly that split. The geometry is the same union either
/// way -- a part's field is the union of its bodies.
fn spokes_as_bodies(n: usize) -> (Document, Graph) {
    let (doc, graph) = spokes(n);
    // Every pipe is already there; dropping the boolean leaves each of them a root.
    let mut out = Graph::empty();
    let mut kept = Vec::new();
    for (_, node) in graph.nodes() {
        if let Node::Pipe { curve, radius } = node {
            kept.push(out.add(Node::Pipe {
                curve: *curve,
                radius: radius.clone(),
            }));
        }
    }
    assert_eq!(kept.len(), n, "every pipe survived");
    (doc, out)
}
