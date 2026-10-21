// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where the bands on a revolved or extruded surface come from, and how wide they are.
//!
//! Reported from the application, 2026-10-02: "anytime it's curve based -- extrudes, revolves
//! that are from a profile -- there's uneven banding on the surface", with pictures of a vase
//! striped horizontally at an even spacing, straight parts included. The suspicion was the
//! occlusion and shadow passes. Three candidates, each measured here against the same vase at
//! the viewport's own settings -- a tolerance of the extent over 128 and an occlusion radius of
//! six percent of the scene's radius, both read off the application rather than chosen:
//!
//! 1. **The profile's facets.** The polyline turns a degree at each joint, so every segment is
//!    a facet with a normal of its own. Printed as segment lengths: if they match the bands,
//!    this is it.
//! 2. **The bake's gradient.** A grid read bilinearly is continuous but its gradient jumps at
//!    every cell edge, and the shading normal is that gradient. Printed as the angle between
//!    the baked field's normal and the walked field's, down the surface.
//! 3. **The bake's band edge.** Fine blocks are kept only near the surface; beyond them the
//!    coarse level answers, with a margin eight times wider -- a step down in the reading, out
//!    where the occlusion's five samples and the shadow's march read. Printed as the shader's
//!    own occlusion formula evaluated on the baked field and on the walked one, down the
//!    surface, beside the block size -- and on a third field, baked with its band widened by
//!    the occlusion's radius, which is the repair `traced/rebuild.rs` makes through
//!    `reads_widened`.
//!
//! Measured 2026-10-02, and the third candidate is the one: the facets are 0.13 to 1.10 mm,
//! the baked normal is within 1.36 degrees of the walked one, and the occlusion down a surface
//! that is open everywhere reads 1.00 on the walked field, 0.75 to 1.00 in three flat levels
//! on the baked one -- 1.00, 0.90, 0.75, by how many of the five samples fall past the band --
//! stepping about twice a block, and 1.00 again on the widened one, for a bake of 15 ms
//! against 5.
//!
//! Run with: `cargo run --release -p fieldform-field --example bake_normal_bands`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    AngleSides, Curve, CurveId, Document, Frame, Graph, Node, Outline, ParamRef,
};
use nalgebra::{Point3, Vector3};

/// The vase's box, which is what the application sizes everything from.
const BOUNDS: (Point3<f64>, Point3<f64>) =
    (Point3::new(-40.0, -40.0, -30.0), Point3::new(40.0, 40.0, 30.0));

fn main() {
    let mut doc = Document::new();
    // A vase: a cubic silhouette that swells and narrows, closed by the axis side at r = 6.
    let id = CurveId::new();
    let control = vec![
        Point3::new(6.0, 0.0, -30.0),
        Point3::new(34.0, 0.0, -30.0),
        Point3::new(40.0, 0.0, -10.0),
        Point3::new(18.0, 0.0, 10.0),
        Point3::new(26.0, 0.0, 30.0),
        Point3::new(6.0, 0.0, 30.0),
    ];
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    doc.curves.insert(
        id,
        Curve::new(id, "vase", 3, control, vec![1.0; 6], knots, true).expect("a closed cubic"),
    );
    let mut graph = Graph::empty();
    graph.add(Node::Revolve { rims: Default::default(),
        outline: Outline::closed(id),
        frame: Frame::default(),
        angle: ParamRef::Literal(360.0),
        sides: AngleSides::One,
    });

    // The application's numbers: `tolerance = extent / mesh_resolution * 0.5` with 64 cells,
    // and `Occlusion::for_scene` reaching six percent of the scene's radius.
    let extent = BOUNDS.1 - BOUNDS.0;
    let tolerance = (extent.x.max(extent.y).max(extent.z) / 64.0) * 0.5;
    let scene_radius = extent.norm() * 0.5;
    let ao_radius = scene_radius * 0.06;
    println!(
        "viewport: tolerance {tolerance:.3} mm, scene radius {scene_radius:.1} mm, occlusion \
         radius {ao_radius:.2} mm"
    );

    // 1. The facets.
    let points = fieldform_field::profile_loop(&doc, id, None).expect("a loop").points;
    let mut lengths: Vec<f64> =
        points.windows(2).map(|w| (w[1] - w[0]).norm()).filter(|l| *l > 1e-6).collect();
    lengths.sort_by(|a, b| a.total_cmp(b));
    println!(
        "1. profile polyline: {} points; segment lengths min {:.2}, median {:.2}, max {:.2} mm",
        points.len(),
        lengths[0],
        lengths[lengths.len() / 2],
        lengths[lengths.len() - 1]
    );

    let baked = GraphField::compile_graph(&doc, &graph, tolerance).expect("compiles");
    let walked = GraphField::compile_graph(&doc, &graph, 1e-4).expect("compiles");
    println!(
        "   baked field says {:?}, walked says {:?}",
        baked.normalization(),
        walked.normalization()
    );
    // The repair: the same bake with its band widened by what the display reads, as
    // `traced/rebuild.rs` now asks for through `reads_widened`. Built by hand because the compile
    // takes the field's own reads and nothing else, which is the point.
    let bake_with = |reads: (f64, f64), what: &str| {
        let world = fieldform_field::profile_loop(&doc, id, None).expect("a loop").points;
        let profile =
            fieldform_field::Profile::about_axis(&world, |p| p).expect("a section about z");
        let (lo, hi) = profile.bounds();
        let started = std::time::Instant::now();
        let cells = fieldform_field::bake_cells((hi - lo).max(), tolerance, reads).expect("bakes");
        let profile = profile.baked(cells, reads);
        let took = started.elapsed();
        let grid = profile.bake().expect("a grid");
        let (stored, of) = grid.stored_blocks();
        let layout = grid.for_shader();
        println!(
            "   {what}: band {:.2} to {:.2} mm, {stored} of {of} blocks stored, baked in {:.0} ms",
            reads.0,
            reads.1,
            took.as_secs_f64() * 1e3
        );
        let block = layout.cell * layout.block as f64;
        println!(
            "      cells of {:.3} mm, a block of {} is {block:.2} mm and the fine band reaches a \
             block's diagonal, {:.2} mm, past the band asked for; margins: fine {:.3} mm, coarse \
             {:.3} mm",
            layout.cell,
            layout.block,
            block * std::f64::consts::SQRT_2,
            layout.margin,
            layout.coarse_margin
        );
        (fieldform_field::Revolution { footprint: fieldform_field::Footprint::Closed(profile) }, block)
    };
    let (_, block) = bake_with((0.0, 0.0), "as the field reads");
    let (widened, _) = bake_with(
        fieldform_field::reads_widened((0.0, 0.0), ao_radius),
        "widened by the occlusion radius",
    );

    let normal = |field: &dyn ScalarField, p: Point3<f64>, h: f64| {
        let d = |o: Vector3<f64>| field.distance(p + o);
        Vector3::new(
            d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
            d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
            d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
        )
        .normalize()
    };
    // The surface, found by marching inward along the radius at each height **on the field
    // being asked**, as the shader finds it: a baked field's march stops on the baked field's
    // own zero set, a margin outside the true boundary, and reads outward from there. Reading
    // the baked field from the exact surface instead would charge it the margin at every
    // sample, which is a shade the viewport never shows.
    let surface_on = |field: &dyn ScalarField, z: f64| {
        let mut r = 60.0;
        for _ in 0..200 {
            let d = field.distance(Point3::new(r, 0.0, z));
            if d.abs() < 1e-7 {
                break;
            }
            r -= d;
        }
        Point3::new(r, 0.0, z)
    };
    // `march.wgsl`'s `occlusion`, line for line, with the field unscaled as a revolve's is.
    let occlusion = |field: &dyn ScalarField, p: Point3<f64>, n: Vector3<f64>| {
        let mut occluded = 0.0;
        let mut falloff = 1.0;
        for i in 1..=5 {
            let h = ao_radius * i as f64 / 5.0;
            let d = field.distance(p + n * h);
            occluded += (h - d) * falloff;
            falloff *= 0.7;
        }
        (1.0 - 1.5 * occluded / ao_radius).clamp(0.0, 1.0)
    };

    println!("\n2 and 3, down the surface:");
    println!(
        "z (mm)   normal error (deg)   occlusion: walked   baked   widened   (1 is open, 0 is \
         black)"
    );
    let mut worst_normal: f64 = 0.0;
    let mut range = [(f64::INFINITY, f64::NEG_INFINITY); 2];
    let mut edges = [Vec::new(), Vec::new()];
    let mut last = [f64::NAN; 2];
    let steps = 1200;
    // Each field is asked from its own surface with its own normal, as the shader asks it.
    let ao_on = |field: &dyn ScalarField, z: f64| {
        let p = surface_on(field, z);
        occlusion(field, p, normal(field, p, 1e-3))
    };
    for i in 0..=steps {
        let z = -29.0 + 58.0 * i as f64 / steps as f64;
        let p = surface_on(&walked, z);
        let exact = normal(&walked, p, 1e-3);
        let fine = normal(&baked, p, 1e-3);
        let error = exact.dot(&fine).clamp(-1.0, 1.0).acos().to_degrees();
        worst_normal = worst_normal.max(error);
        let ao_walked = ao_on(&walked, z);
        let ao = [ao_on(&baked, z), ao_on(&widened, z)];
        for k in 0..2 {
            range[k].0 = range[k].0.min(ao[k]);
            range[k].1 = range[k].1.max(ao[k]);
            // A band edge is where the occlusion steps between one height and the next.
            if last[k].is_finite() && (ao[k] - last[k]).abs() > 0.02 {
                edges[k].push(z);
            }
            last[k] = ao[k];
        }
        if i % 40 == 0 {
            println!(
                "{z:>7.2}   {error:>8.3}             {ao_walked:>8.3}   {:>8.3}   {:>8.3}",
                ao[0], ao[1]
            );
        }
    }
    println!("\n2. worst normal error between baked and walked: {worst_normal:.2} deg");
    for (k, what) in ["baked as the field reads", "baked widened by the occlusion radius"]
        .iter()
        .enumerate()
    {
        println!(
            "3. {what}: occlusion ranges {:.3} to {:.3} down a surface that is open everywhere, \
             {} steps of more than 0.02 between neighbouring heights",
            range[k].0,
            range[k].1,
            edges[k].len()
        );
        let gaps: Vec<f64> =
            edges[k].windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.2).collect();
        if !gaps.is_empty() {
            let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
            println!("   {:.2} mm apart on average, against a block of {block:.2} mm", mean);
        }
    }
}
