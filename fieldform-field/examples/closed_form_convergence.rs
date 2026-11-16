// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How fast a measured volume and area approach their closed forms as the grid is halved.
//!
//! `expect` measures at a spacing and again at half of it, takes the finer reading, and calls the
//! difference its error estimate. That estimate is sound only if the method converges at order one
//! or better and is already in its asymptotic range, so this prints the signed error at each level
//! of the same anchored grid and the order the error falls at between two of them, for shapes whose
//! answer is known on paper. A single reading that is close and then stops improving is the fault
//! this is here to show, and a sign that never changes is a bias worth knowing the size of.
//!
//! `cargo run --release -p fieldform-field --example closed_form_convergence`

use std::f64::consts::PI;

use fieldform_field::{Anchored, MassProperties, ScalarField};
use nalgebra::{Point3, Vector3};

struct Sdf<F: Fn(Point3<f64>) -> f64 + Send + Sync>(F);

impl<F: Fn(Point3<f64>) -> f64 + Send + Sync> ScalarField for Sdf<F> {
    fn distance(&self, p: Point3<f64>) -> f64 {
        (self.0)(p)
    }
}

struct Case {
    name: &'static str,
    field: Box<dyn ScalarField + Sync>,
    half: f64,
    volume: f64,
    area: f64,
}

fn cases() -> Vec<Case> {
    let sphere = |c: Point3<f64>, r: f64| move |p: Point3<f64>| (p - c).norm() - r;
    let (ra, rb) = (sphere(Point3::new(-5.0, 0.0, 0.0), 10.0), sphere(Point3::new(5.0, 0.0, 0.0), 10.0));
    let half_box = Vector3::new(10.0, 15.0, 20.0);
    let (a, b, c, d) = (20.0, 30.0, 40.0, 3.0);
    vec![
        Case {
            name: "ball r 10",
            field: Box::new(Sdf(sphere(Point3::origin(), 10.0))),
            half: 12.0,
            volume: 4.0 / 3.0 * PI * 1000.0,
            area: 4.0 * PI * 100.0,
        },
        Case {
            name: "lens of two balls",
            field: Box::new(Sdf(move |p| ra(p).max(rb(p)))),
            half: 12.0,
            volume: PI * 50.0 * 100.0 / 12.0,
            area: 2.0 * 2.0 * PI * 10.0 * 5.0,
        },
        Case {
            name: "bicylinder r 10",
            field: Box::new(Sdf(|p: Point3<f64>| p.y.hypot(p.z).max(p.x.hypot(p.z)) - 10.0)),
            half: 12.0,
            volume: 16.0 * 1000.0 / 3.0,
            area: 16.0 * 100.0,
        },
        Case {
            name: "torus 30 / 5",
            field: Box::new(Sdf(|p: Point3<f64>| (p.x.hypot(p.y) - 30.0).hypot(p.z) - 5.0)),
            half: 37.0,
            volume: 2.0 * PI * PI * 30.0 * 25.0,
            area: 4.0 * PI * PI * 30.0 * 5.0,
        },
        Case {
            name: "box 20x30x40 grown 3",
            field: Box::new(Sdf(move |p: Point3<f64>| {
                let q = p.coords.abs() - half_box;
                q.sup(&Vector3::zeros()).norm() + q.max().min(0.0) - d
            })),
            half: 25.0,
            volume: a * b * c + 2.0 * d * (a * b + b * c + c * a) + PI * d * d * (a + b + c) + 4.0 / 3.0 * PI * d.powi(3),
            area: 2.0 * (a * b + b * c + c * a) + 2.0 * PI * d * (a + b + c) + 4.0 * PI * d * d,
        },
    ]
}

fn main() {
    let spacings = [1.0, 0.5, 0.25, 0.125, 0.0625];
    println!("Signed error of the anchored grid's reading against the closed form, and the order");
    println!("the error falls at from the coarser level to this one (2 is a square law).\n");
    println!("{:<22} {:>7}  {:>11} {:>6}  {:>11} {:>6}  {:>8}", "shape", "h mm", "volume err", "order", "area err", "order", "time");
    for case in cases() {
        let bounds = (Point3::from(Vector3::repeat(-case.half)), Point3::from(Vector3::repeat(case.half)));
        let mut previous: Option<(f64, f64)> = None;
        for &h in &spacings {
            let started = std::time::Instant::now();
            let m = match MassProperties::anchored(case.field.as_ref(), bounds, h) {
                Anchored::Measured(m) => m,
                Anchored::TooFine { corners } => {
                    println!("{:<22} {h:>7}  past the corner budget ({corners} corners)", case.name);
                    break;
                }
                Anchored::Empty => panic!("{} has no surface in its box", case.name),
            };
            let secs = started.elapsed().as_secs_f64();
            let ev = (m.props.volume_mm3 - case.volume) / case.volume;
            let ea = (m.props.surface_area_mm2 - case.area) / case.area;
            let order = |now: f64, before: Option<f64>| match before {
                Some(b) if now != 0.0 => format!("{:>6.2}", (b.abs() / now.abs()).log2()),
                _ => format!("{:>6}", ""),
            };
            println!(
                "{:<22} {h:>7}  {:>+10.5}% {}  {:>+10.5}% {}  {secs:>7.2}s",
                case.name,
                ev * 100.0,
                order(ev, previous.map(|p| p.0)),
                ea * 100.0,
                order(ea, previous.map(|p| p.1)),
            );
            previous = Some((ev, ea));
        }
        println!();
    }
}
