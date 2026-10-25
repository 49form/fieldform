// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a rebuild pays for the atlas when its bands have not moved: kept, and packed afresh.
//!
//! Refactor plan step 3.6, the atlas third of the performance plan's V2. A rebuild hands the
//! renderer every baked band again, and the renderer used to copy every sample of every band
//! twice on the interface's thread -- once into texels, once into the padded atlas -- and upload
//! the lot, though a rebuild that changed a number, a curve or the shader leaves every band where
//! it was and the view hands back the very bands it kept. `Renderer::rebuilt_from` now keeps the
//! atlas when its bands are the same ones (`atlas_holds`).
//!
//! `cargo run --release -p fieldform-gpu --example kept_atlas_cost`
//!
//! Two arms on one renderer, alternated within each round so neither always runs on a warmer
//! card: **the same bands**, which keeps the atlas, and **the same samples in bands made afresh**,
//! which packs it -- what every rebuild paid before, and what one still pays when a band moves.
//! Both keep the pipeline, since the source is the same, so the rest of the rebuild is equal and
//! the difference is the atlas alone. Over the starting cage at the draft's 32 cells and the
//! settled rungs' 64, 128 and 192 -- and the kept renderer is held to draw what the fresh one
//! draws, to the bit, since an atlas that was kept wrongly is a wrong picture.

use std::sync::Arc;
use std::time::Instant;

use fieldform_cage::Cage;
use fieldform_field::{CageField, levels_for_tolerance};
use fieldform_gpu::{BakedField, Camera, Renderer};
use fieldform_scene::{CageId, Document, Graph, Node};
use nalgebra::Point3;

const ROUNDS: usize = 15;

fn median(mut ms: Vec<f64>) -> f64 {
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

fn main() {
    // The application's starting cage, over the box the view bakes it in: its hull with a
    // twentieth of the diagonal to spare, and a band of fifteen per cent of that box.
    let cage = Cage::cube_of(1000.0);
    let (mut lo, mut hi) = (Point3::from([f64::MAX; 3]), Point3::from([f64::MIN; 3]));
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = lo.inf(&p);
        hi = hi.sup(&p);
    }
    let margin = (hi - lo).norm() * 0.05;
    let bounds = (
        lo - nalgebra::Vector3::repeat(margin),
        hi + nalgebra::Vector3::repeat(margin),
    );
    let band = (bounds.1 - bounds.0).norm() * 0.15;
    let levels = levels_for_tolerance((hi - lo).norm(), ((hi - lo).norm() / 64.0 * 0.5).max(1e-4));
    let field = CageField::from_cage(&cage, levels).expect("a cube subdivides");

    let id = CageId::new();
    let mut doc = Document::new();
    doc.cages.insert(id, cage.into());
    let mut graph = Graph::empty();
    graph.add(Node::Cage(id));
    let program = fieldform_gpu::generate(&doc, &graph).expect("a cage generates");

    // A device, from a first renderer at the draft's resolution.
    let first = pollster::block_on(Renderer::new(
        &program,
        &[(&field, bounds)],
        &[],
        &[],
        32,
        band,
    ))
    .expect("a GPU, which is where a renderer is built");
    let (device, queue) = (first.device().clone(), first.queue().clone());
    println!("{}\n", device.adapter_info().name);
    println!("Medians of {ROUNDS}, milliseconds of the interface's thread.\n");
    println!(
        "| cells | band | atlas, packed afresh | atlas, kept | rebuild, packed afresh | rebuild, kept |"
    );
    println!("|---|---|---|---|---|---|");

    let camera = Camera::framing(bounds, 1.5);
    for cells in [32usize, 64, 128, 192] {
        let bands = vec![Arc::new(BakedField::new(&field, bounds, cells, band))];
        let base = Renderer::with_device(device.clone(), queue.clone(), &program, &bands, &[], &[])
            .expect("the renderer builds");
        let (mut afresh_atlas, mut kept_atlas) = (Vec::new(), Vec::new());
        let (mut afresh_whole, mut kept_whole) = (Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            // The copy is made before the clock starts: it stands for a band the view baked
            // anew, whose cost is the bake's and not the atlas's.
            let fresh = vec![Arc::new((*bands[0]).clone())];
            let order = if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for arm in order {
                let handed = if arm { &bands } else { &fresh };
                let began = Instant::now();
                let built = Renderer::rebuilt_from(&base, &program, handed, &[], &[])
                    .expect("the renderer builds");
                let whole = began.elapsed().as_secs_f64() * 1e3;
                let atlas = f64::from(built.build_cost().atlas_ms);
                if arm {
                    kept_atlas.push(atlas);
                    kept_whole.push(whole);
                } else {
                    afresh_atlas.push(atlas);
                    afresh_whole.push(whole);
                }
                drop(built);
                // An upload waits for the next submission, which a frame would make and this
                // loop does not; without one every packed atlas stays queued, and fifteen of the
                // largest ran the card out of memory. Submitted and waited for, so each round
                // starts with nothing of the last one's outstanding.
                queue.submit(std::iter::empty());
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
        }

        // The kept atlas draws what a packed one draws, to the bit -- with the renderer it was
        // kept from gone, as the view lets the previous one go straight after a rebuild.
        let fresh = vec![Arc::new((*bands[0]).clone())];
        let kept = Renderer::rebuilt_from(&base, &program, &bands, &[], &[]).expect("builds");
        let packed = Renderer::rebuilt_from(&base, &program, &fresh, &[], &[]).expect("builds");
        drop(base);
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
            .collect();
        kept.set_params(&values);
        packed.set_params(&values);
        let a = kept.render(&camera, 240, 160, 1024).expect("renders");
        let b = packed.render(&camera, 240, 160, 1024).expect("renders");
        assert!(
            a == b,
            "the kept atlas drew a different frame at {cells} cells"
        );
        assert!(
            a.iter().any(|hit| hit.is_hit()),
            "the frame shows the cage at {cells} cells"
        );
        // **The control: a band that has moved is drawn moved.** Equal samples cannot tell a
        // kept atlas from one kept wrongly, so the same rebuild is handed a band whose surface
        // sits fifty millimetres further in -- and from the kept renderer, whose atlas is the
        // old one, it must not come back drawing the old shape.
        let mut shrunk = (*bands[0]).clone();
        for d in &mut shrunk.data {
            *d += 50.0;
        }
        let moved =
            Renderer::rebuilt_from(&kept, &program, &[Arc::new(shrunk)], &[], &[]).expect("builds");
        moved.set_params(&values);
        let c = moved.render(&camera, 240, 160, 1024).expect("renders");
        assert!(a != c, "a moved band drew the kept atlas at {cells} cells");

        println!(
            "| {cells} | {} MB | {:.2} | {:.3} | {:.2} | {:.2} |",
            bands[0].size_bytes() / 1_000_000,
            median(afresh_atlas),
            median(kept_atlas),
            median(afresh_whole),
            median(kept_whole),
        );
    }
    println!(
        "\nEvery row's kept renderer drew the frame the packed one drew, bit for bit, and a moved band drew a different one."
    );
}
