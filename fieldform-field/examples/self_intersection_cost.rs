// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How long the self-intersection check takes, on the tessellation the field is built from.
//!
//! The check runs after every committed cage edit, off the interface thread, and the cage shows
//! "checking" until it answers; over two seconds on the floor machine is a performance fault, not
//! a reason to run it coarser. This prints the table the decision rests on: the starting cage,
//! that cage refined once, twice and three times, each at the tessellation level the viewport
//! would use for it (4, the floor), with the tessellation and the check timed apart.
//!
//!     cargo run --release -p fieldform-field --example self_intersection_cost

use fieldform_cage::validate::epsilon_length;
use fieldform_cage::Cage;
use fieldform_field::self_intersection;
use fieldform_subd::LimitSurface;
use std::time::Instant;

fn main() {
    println!("cage faces | level | triangles | tessellate | check | crossings");
    let mut cage = Cage::cube_of(1000.0);
    for refinements in 0..=4 {
        if refinements > 0 {
            cage = fieldform_subd::subdivide(&cage).expect("a refined cube is a cage");
        }
        for levels in [3usize, 4] {
            let started = Instant::now();
            let t = LimitSurface::new(&cage).tessellate(levels);
            let tessellated = started.elapsed();
            let started = Instant::now();
            let found = self_intersection::find_in(&cage, &t, epsilon_length(&cage));
            let checked = started.elapsed();
            println!(
                "{:>10} | {:>5} | {:>9} | {:>8.1} ms | {:>7.1} ms | {} crossings",
                cage.face_count(),
                levels,
                t.triangles.len(),
                tessellated.as_secs_f64() * 1e3,
                checked.as_secs_f64() * 1e3,
                found.crossings,
            );
        }
    }
    // And the fault the check exists for: a corner dragged straight down through the bottom face,
    // at the viewport's level. (Dragged diagonally through the opposite corner the limit surface
    // stays embedded -- measured, see the check's own tests.)
    let cube = Cage::cube_of(1000.0);
    let mut positions: Vec<_> = cube.vertex_ids().map(|v| cube.vertex_position(v)).collect();
    positions[6] = nalgebra::Point3::new(0.0, 0.0, -1200.0);
    let faces: Vec<Vec<u32>> =
        cube.face_ids().map(|f| cube.face_vertices(f).iter().map(|v| v.index() as u32).collect()).collect();
    let folded = Cage::from_faces(positions, &faces).unwrap();
    let started = Instant::now();
    let found = self_intersection::find(&folded, 4, epsilon_length(&folded));
    println!(
        "\na corner dragged through the cube, level 4: {} faces red, {} crossings, {:.1} ms",
        found.faces.len(),
        found.crossings,
        started.elapsed().as_secs_f64() * 1e3
    );
}
