// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Drives every path the lighting pass can take, offscreen, and says whether the GPU survived.
//!
//! The viewport's overlays — the selection outline, the draft read, the flake, each of the three
//! environments — are all shader paths that only run when something is picked in the interface.
//! A fault in one of them shows up as a device that stops answering, which in the application
//! looks like a window that vanished. This presents a frame down each path and reads it back, so
//! a fault is a message here rather than a disappearance there.
//!
//! Run with:
//! `cargo run -p fieldform-gpu --example overlay_smoke`
//!
//! **And `cargo test` runs it too**, through
//! `tests/the_shaded_pass_survives_every_lighting_path.rs`, which includes this file and calls
//! [`run`]. An example is something a person remembers to run, and the 9-against-12 crash (loop
//! spec §1 row 1) shipped past every one of them, because they march plainly and only the
//! application shades (loop spec S5). The test is this file rather than a copy of it so that the
//! table printed here and the gate cannot drift apart.

use fieldform_gpu::{
    generate, Camera, Draft, EnvironmentChoice, Look, Occlusion, RenderError, Renderer,
    ShadingResources,
};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Frame, Graph, LatticeCell, LatticePhase, Material, Node, ParamRef,
};
use nalgebra::{Point3, Vector3};

const SIZE_MM: f64 = 100.0;

fn main() {
    match run() {
        Ok(frames) => {
            println!("{frames} frames presented and read back; the device is still answering.")
        }
        Err(e) => eprintln!("no GPU available ({e}); nothing else in the workspace needs one."),
    }
}

/// Presents a frame down every path the lighting pass can take and reads each one back,
/// panicking at the first that is wrong. Answers how many frames it presented, or the
/// renderer's own reason it could not start — [`RenderError::NoAdapter`] on a machine with no
/// GPU, which is the one answer the test reads as a skip rather than a fault.
pub fn run() -> Result<u32, RenderError> {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let cube = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
        size: [ParamRef::Literal(SIZE_MM * 0.7); 3],
        turn: fieldform_scene::no_turn(),
    });
    let sphere = graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([ParamRef::Literal(SIZE_MM * 0.3), ParamRef::Literal(0.0), ParamRef::Literal(0.2 * SIZE_MM)]),
        radius: ParamRef::Literal(SIZE_MM * 0.3),
    });
    // Two bodies, so the outline has an edge between bodies to find as well as a silhouette.
    let bitten = graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![cube.into(), sphere.into()] });
    let beside = graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([ParamRef::Literal(-SIZE_MM * 0.5), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
        radius: ParamRef::Literal(SIZE_MM * 0.15),
    });
    // **Stamped in the order made, as the application stamps every block, so body 0 is the
    // bitten cube.** Bodies are numbered as `Document::folded_bodies` orders them: by stamp, then
    // by raw id. Ids are random, so in a document nothing stamped body 0 was the small sphere
    // about half the time -- behind the cube from this camera, where an outline draws nothing,
    // and the two checks that select body 0 passed with no outline drawn anywhere.
    for (made, id) in [cube, sphere, bitten, beside].into_iter().enumerate() {
        doc.created.insert(id, made as u64 + 1);
    }
    let program = generate(&doc, &graph).expect("the graph generates");

    let renderer = pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0))?;
    // Which card answered, so a failure read from the test's captured output says where it
    // happened: a number without the adapter that produced it is not a measurement.
    let adapter = renderer.device().adapter_info();
    println!("adapter: {} ({:?}, {:?})", adapter.name, adapter.backend, adapter.device_type);
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => unreachable!("literals only"),
        })
        .collect();
    renderer.set_params(&values);

    let half = SIZE_MM * 0.5;
    let bounds = (Point3::new(-half, -half, -half), Point3::new(half, half, half));
    let (width, height) = (640u32, 400u32);
    let mut camera = Camera::framing(bounds, width as f64 / height as f64);

    let mut environments = EnvironmentChoice::ALL.iter().map(|c| (*c, c.bake()));
    let (first_choice, first_maps) = environments.next().expect("at least one environment");
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &first_maps)
        .expect("the lighting pipeline builds");
    let baked: Vec<_> = std::iter::once((first_choice, first_maps)).chain(environments).collect();

    let mut frames = 0u32;
    let mut check = |renderer: &Renderer, shading: &mut ShadingResources, camera: &Camera, look: &Look, accumulated: u32, what: &str| {
        renderer
            .present(shading, camera, width, height, 512, look, accumulated)
            .expect("the frame presents");
        let image = renderer.read_display(shading).expect("the frame reads back");
        let sum: u64 = image.rgba.iter().map(|&b| u64::from(b)).sum();
        let hits = renderer.read_hdr(shading).expect("the hdr reads back");
        let nan = hits.iter().filter(|p| p.iter().any(|v| !v.is_finite())).count();
        frames += 1;
        println!("  {what:<44} checksum {sum:>10}  non-finite {nan}");
        assert_eq!(nan, 0, "{what}: the HDR buffer holds non-finite values");
        sum
    };

    for (choice, maps) in &baked {
        println!("environment: {}", choice.label());
        shading.set_environment(renderer.device(), renderer.queue(), maps);
        // Each material's drag frame, for the check after the loop: whether the material reaches
        // the picture at all. Under normals it must not, which is the whole claim of that view.
        let mut drags: Vec<(&str, u64)> = Vec::new();
        for material in [Material::default(), Material::preset("Car paint").unwrap(), Material::preset("Chrome").unwrap(), Material::preset("Brushed aluminium").unwrap()] {
            let name = material.preset_name().unwrap_or("custom");
            let mut look = Look::for_environment(*choice, material, camera.forward, camera.right, camera.up);
            camera.occlusion = None;
            camera.jitter = [0.0, 0.0];
            let dragged =
                check(&renderer, &mut shading, &camera, &look, 0, &format!("{name}: drag frame"));
            drags.push((name, dragged));

            look.selected = Some(0);
            look.outline_radius = 2;
            let outlined =
                check(&renderer, &mut shading, &camera, &look, 0, &format!("{name}: outline on body 0"));
            // **An outline that draws nothing checks nothing.** The outline is drawn outside the
            // selected body's own pixels, so a selected body the camera cannot see leaves the
            // frame the drag frame byte for byte -- and this check, and the bands' below, would
            // pass with no outline drawn anywhere.
            assert_ne!(
                outlined, dragged,
                "{name}: selecting body 0 changed no pixel, so the selected body is not one the \
                 camera sees and no outline was drawn to check"
            );

            look.draft = Some(Draft::new(Vector3::z(), 3.0, 1.0));
            check(&renderer, &mut shading, &camera, &look, 0, &format!("{name}: draft along +Z"));
            look.draft = Some(Draft::new(-Vector3::z(), 3.0, 0.01));
            let drafted =
                check(&renderer, &mut shading, &camera, &look, 0, &format!("{name}: draft flipped, sharp"));

            // The still tier: occlusion, shadow and eight accumulated jittered frames.
            camera.occlusion = Some(Occlusion::for_scene(look.lights[0].direction, half * 3f64.sqrt()));
            let mut stills = Vec::new();
            for frame in 0..8 {
                camera.jitter = fieldform_gpu::halton_jitter(frame);
                stills.push(check(&renderer, &mut shading, &camera, &look, frame, &format!("{name}: still frame {frame}")));
            }
            // **A still is not a draft** -- the gate the draft-tier split needs and nothing had.
            //
            // The still's occlusion and shadow are a pipeline of their own
            // (`Renderer::occlude_pipeline`): the draft's march does not mention `occlusion` or
            // `shadow` so the driver need not compile them, and the still dispatches `occlude` after
            // it. If that dispatch were ever left out of a still, the picture would simply lose its
            // ambient term and its shadow -- which reads as a *lighting* change, not as a broken
            // march, and no existing check would have said a word. So: the occlusion has to move
            // the image.
            //
            // Held over the eight accumulated frames rather than on frame 0, because a material can
            // legitimately match. `Brushed aluminium` does: it is near enough a mirror that the
            // ambient term it multiplies is nought, and its still frame 0 is byte-identical to its
            // draft. Clay and Car paint differ by tens of thousands. One material matching is a
            // material; all of them matching is a wrong pipeline.
            //
            // **Not asked of a view that paints normals** (item 116), where the occlusion is not
            // read at all -- `wants_occlusion` is false and the lighting pass returns before it
            // would look. Such a still still differs from its draft, because the eight frames are
            // jittered and the edges antialias, so this would *pass* there and say nothing about
            // the dispatch it exists to catch. What holds instead is below: the material cannot
            // reach the picture.
            if !choice.paints_normals() {
                assert!(
                    stills.iter().any(|s| *s != drafted),
                    "{name}: eight still frames all match the draft byte for byte, so the occlusion                  and the shadow are doing nothing -- see Renderer::occlude_pipeline"
                );
            }
        }
        // **A normals view shows the march alone, so the material cannot reach it** (item 116) --
        // and every other environment is the control, where it must. This is the claim that view
        // is for: a mark that survives is in the geometry, and the way to know the lighting is
        // really out of the picture is that the four materials draw one frame. Clay against
        // chrome is the whole of the lighting pass; if any of it still ran, these would differ.
        let (_, first) = drags[0];
        if choice.paints_normals() {
            let odd: Vec<&str> =
                drags.iter().filter(|(_, sum)| *sum != first).map(|(n, _)| *n).collect();
            assert!(
                odd.is_empty(),
                "normals: {odd:?} drew a different frame from {}, so the material still reaches \
                 the picture and the lighting pass is not being skipped",
                drags[0].0,
            );
        } else {
            assert!(
                drags.iter().any(|(_, sum)| *sum != first),
                "{}: all four materials drew one frame, so the material is reaching nothing and \
                 the check above it cannot fail",
                choice.label(),
            );
        }
    }
    // **A frame in bands is the same frame.** The application traces a still as several
    // submissions of a band of rows each (loop spec S2), and nothing about that may change a
    // pixel: the band is a uniform value the shaders offset their rows by, and the bind groups
    // are made once and shared. Compared byte for byte against the whole frame, with the
    // outline on, since the outline reads the owners of neighbouring rows across the band edge.
    let (choice, _) = &baked[0];
    let mut look = Look::for_environment(*choice, Material::default(), camera.forward, camera.right, camera.up);
    look.selected = Some(0);
    // Every pixel, as the viewport draws a still; and one in four (loop spec P5), whose second
    // dispatch answers for the row above each band's first measured one -- a row the band before
    // it marched -- so its band edges are the ones that could go wrong.
    let every_pixel = Occlusion::for_scene(look.lights[0].direction, half * 3f64.sqrt());
    let one_in_four = Occlusion { half_resolution: true, ..every_pixel };
    camera.jitter = [0.0, 0.0];
    shading.set_environment(renderer.device(), renderer.queue(), &baked[0].1);
    for (occlusion, what) in [(every_pixel, "still in three bands"), (one_in_four, "still at one in four, in three bands")] {
        camera.occlusion = Some(occlusion);
        renderer
            .present(&mut shading, &camera, width, height, 512, &look, 0)
            .expect("the whole frame presents");
        let whole = renderer.read_display(&shading).expect("the whole frame reads back");
        // **And with another view's frame lying underneath**, which is how the application meets
        // it: zoom into the selected body, let it settle, zoom out. The rows a band has not
        // reached still hold the last view -- the body in every pixel -- and an outline that reads
        // across the band's edge into them finds "selected" beside "not selected" all along it.
        // Reported by the owner as horizontal lines across a settled frame, spaced as the bands
        // are. With the same view underneath, as this check used to have it, the stale rows are
        // right by luck and nothing shows.
        let mut close = camera;
        close.eye = camera.eye + (camera.forward * ((camera.eye - Point3::origin()).norm() * 0.8));
        renderer
            .present(&mut shading, &close, width, height, 512, &look, 0)
            .expect("the close frame presents");
        let bands = [0..height / 3, height / 3..2 * height / 3, 2 * height / 3..height];
        for rows in bands {
            renderer
                .present_rows(&mut shading, &camera, width, height, 512, &look, 0, rows)
                .expect("the band presents");
        }
        let banded = renderer.read_display(&shading).expect("the banded frame reads back");
        let differing = whole.rgba.iter().zip(&banded.rgba).filter(|(a, b)| a != b).count();
        frames += 5;
        println!("  {what:<44} {differing} bytes differ from the whole frame");
        assert_eq!(differing, 0, "{what}: a frame traced in bands must be the frame traced whole");
    }

    // **At one in four a still marches its occlusion and its shadow for a quarter of the pixels**
    // (loop spec P5), and for those the lighting pass cannot fill in from them -- here, the rims
    // of large faces and the edges of their shadows. Every other pixel that hit holds `NO_READING`
    // for the lighting pass to fill, so the share of them holding a reading is the share the
    // march measured: 0.32 here, where every pixel's would be 1. A fill that judged every pixel
    // unfillable would give the same picture at every pixel's cost, and nothing but this would
    // notice.
    camera.occlusion = Some(one_in_four);
    renderer
        .present(&mut shading, &camera, width, height, 512, &look, 0)
        .expect("the still presents");
    let extras = renderer.read_extras(&shading).expect("the extras read back");
    let hits = renderer.render(&camera, width, height, 512).expect("the hits read back");
    let hit = hits.iter().filter(|h| h.is_hit()).count();
    let marched = hits
        .iter()
        .zip(&extras)
        .filter(|(h, [open, _])| h.is_hit() && *open >= 0.0)
        .count();
    let share = marched as f64 / hit.max(1) as f64;
    frames += 1;
    println!("  {:<44} {:.3} of {hit} pixels that hit", "one in four, occlusion and shadow marched at", share);
    assert!(
        (0.2..0.4).contains(&share),
        "a still at one in four measured its occlusion and shadow at {share:.3} of the pixels that \
         hit; it marches them for a quarter, and for the few the lighting pass cannot fill in"
    );

    // **And the still it finishes is the still measured at every pixel.** Eight jittered frames
    // each way, compared as they are shown. A pixel is filled only between two measured pixels of
    // its own surface that agree, so what may differ is a reading that varies within the two
    // pixels between them; a fill that read the wrong pixels, or a band of rows nothing measured,
    // moves whole faces and fails this by a wide margin.
    let quarter = finished_still(&renderer, &mut shading, &Camera { occlusion: Some(one_in_four), ..camera }, &look, (width, height), 512);
    let every = finished_still(&renderer, &mut shading, &Camera { occlusion: Some(every_pixel), ..camera }, &look, (width, height), 512);
    let (over_four, worst) = apart(&every, &quarter);
    frames += 16;
    println!(
        "  {:<44} {:.3} % more than four levels, worst {worst}",
        "still at one pixel in four, against every",
        100.0 * over_four
    );
    assert!(
        over_four < 0.01,
        "{:.2} % of the still moved by more than four levels when its occlusion and shadow were \
         measured at one pixel in four -- the fill is reading something other than its neighbours",
        100.0 * over_four
    );

    // **And where a strut is a pixel or two across**, which is the scene above's opposite: a
    // pixel on a strut that narrow has no measured pixel of its own surface beside it -- its
    // neighbours are the background, or another strut further back -- or has a strut's shadow
    // crossing it a pixel wide. A fill there can only give it a reading of something else, and
    // one did: this block finished 8.4 % of its still more than four levels from every pixel's
    // when every unmeasured pixel was filled. Its still is held to the same tolerance.
    let (lattice_doc, lattice_graph) = strut_block();
    let lattice = generate(&lattice_doc, &lattice_graph).expect("the lattice generates");
    let struts = Renderer::with_device(
        renderer.device().clone(),
        renderer.queue().clone(),
        &lattice,
        &[],
        &[],
        &[],
    )?;
    let literals: Vec<f32> = lattice
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => unreachable!("literals only"),
        })
        .collect();
    struts.set_params(&literals);
    let scale = fieldform_field::step_multiplier(&lattice_doc, &lattice_graph);
    let budget = (1024.0 * scale).ceil() as u32;
    let mut framed = Camera::framing(bounds, width as f64 / height as f64);
    framed.start_at_scene = true;
    framed.value_scale = scale;
    let plain = Look::for_environment(*choice, Material::default(), framed.forward, framed.right, framed.up);
    let every_pixel = Occlusion::for_scene(plain.lights[0].direction, half * 3f64.sqrt());
    framed.occlusion = Some(Occlusion { half_resolution: true, ..every_pixel });
    let quarter = finished_still(&struts, &mut shading, &framed, &plain, (width, height), budget);
    framed.occlusion = Some(every_pixel);
    let every = finished_still(&struts, &mut shading, &framed, &plain, (width, height), budget);
    let (over_four, worst) = apart(&every, &quarter);
    frames += 16;
    println!(
        "  {:<44} {:.3} % more than four levels, worst {worst}",
        "struts at one pixel in four, against every",
        100.0 * over_four
    );
    assert!(
        over_four < 0.01,
        "{:.2} % of a still of struts a pixel or two across moved by more than four levels when \
         its occlusion and shadow were measured at one pixel in four -- a pixel was filled from \
         neighbours on another surface, or from neighbours that disagree",
        100.0 * over_four
    );

    Ok(frames)
}

/// A block filled with face-centred struts a pixel or two across at the size this file draws:
/// 0.8 mm thick at about 0.45 mm a pixel, on a 14 mm cell, so a strut's pixels mostly have the
/// background or another strut beside them rather than more of themselves.
fn strut_block() -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    let block = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(SIZE_MM * 0.7); 3],
        turn: fieldform_scene::no_turn(),
    });
    let struts = graph.add(Node::Lattice {
        cell: LatticeCell::Fcc,
        frame: Frame::default(),
        map: CellMap::Rectangular { cell: lit(SIZE_MM * 0.14) },
        phase: LatticePhase::Strut { radius: lit(SIZE_MM * 0.004) },
    });
    graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![block.into(), struts.into()] });
    (Document::new(), graph)
}

/// Eight jittered frames of `camera`, accumulated as the viewport accumulates a still, and the
/// picture they finish as: display-encoded bytes.
fn finished_still(
    renderer: &Renderer,
    shading: &mut ShadingResources,
    camera: &Camera,
    look: &Look,
    (width, height): (u32, u32),
    budget: u32,
) -> Vec<u8> {
    let mut jittered = *camera;
    for frame in 0..8 {
        jittered.jitter = fieldform_gpu::halton_jitter(frame);
        renderer
            .present(shading, &jittered, width, height, budget, look, frame)
            .expect("the still presents");
    }
    renderer.read_display(shading).expect("the still reads back").rgba
}

/// How far apart two pictures of one view are, as the share of pixels whose furthest channel is
/// more than four levels of eight bits away, and the furthest of all.
fn apart(a: &[u8], b: &[u8]) -> (f64, u8) {
    let apart: Vec<u8> = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .map(|(p, q)| (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap_or(0))
        .collect();
    let over_four = apart.iter().filter(|d| **d > 4).count() as f64 / apart.len().max(1) as f64;
    (over_four, apart.iter().copied().max().unwrap_or(0))
}
