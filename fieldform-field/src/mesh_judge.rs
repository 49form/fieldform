// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a mesh is the part it was made from: closed, the right way out, and where the field is.
//!
//! An exported STL is the one thing this program hands to a machine that cannot ask what was meant.
//! A slicer reads a torn edge as a hole and a reversed shell as a cavity, and either prints the
//! wrong part without complaint. So a tessellation is judged against the field it came from rather
//! than trusted for having come out of the mesher: every question here is answered from the
//! triangles alone or from the field alone, never from what the mesher believed it did.
//!
//! # Closed
//!
//! Every edge belongs to exactly two triangles, and every vertex's triangles form one fan. The
//! second is not implied by the first: two cones touching at their tips have no open edge and a
//! vertex that is two surfaces at once, which a slicer reads however it likes.
//!
//! # The right way out
//!
//! Three readings, because each misses something the others catch. Two triangles sharing an edge
//! must cross it in opposite directions, or the winding flips between them. Each connected shell
//! must enclose positive volume, or the whole shell is inside out with every edge consistent. And
//! each triangle's normal must agree with the field's gradient at its middle, which is the only
//! one of the three that knows which side is the material.
//!
//! # Where the field is: a distance each way
//!
//! The Hausdorff distance is the larger of two one-sided distances, and both are measured.
//! **Off the surface** is how far a point of the mesh is from the field's zero set, sampled at the
//! vertices, the edge middles and the triangle middles -- the middle being where a flat triangle
//! sags furthest from a curved face. **Uncovered** is how far a point of the zero set is from the
//! mesh: it is what finds a hole the mesher stepped over or a strut it dropped, which no reading of
//! the mesh's own points can, since a missing piece has no points.
//!
//! Each is an *upper* bound wherever it is reported as settled. The field's value is a lower bound
//! on the distance, by its contract, so a reading of it alone would pass a mesh that was further
//! off than it said; the distance used is instead to a point that was walked onto the surface, and
//! the surface is at least that near. A walk that does not settle is counted rather than read, and
//! a judgement with unsettled samples has not shown what it would otherwise claim.
//!
//! The zero set is sampled where the field says it may be -- cells of a grid whose centres read
//! within a half-diagonal of nought, which by the same contract holds every cell the surface
//! crosses -- so the sampling does not depend on where the mesh happens to be.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::bvh::Bvh;
use crate::sdf::closest_point_on_triangle;
use crate::surface_query::{surface_normal, walk_to_surface};
use crate::ScalarField;

/// The most points either distance is sampled at. Past this the samples are strided, and the
/// zero set's grid coarsened, so a judgement costs about the same for any mesh.
const MOST_SAMPLES: usize = 400_000;

/// How far one surface is from another, sampled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reach {
    /// The furthest any settled sample was, in millimetres.
    pub worst: f64,
    /// Where that sample was.
    pub at: Option<Point3<f64>>,
    /// The distance 99 in 100 settled samples were within, so one bad place reads apart from a
    /// mesh that is off everywhere.
    pub p99: f64,
    /// How many samples were measured.
    pub samples: usize,
    /// Samples whose walk onto the surface did not settle, so whose distance is not known. Not in
    /// `worst`: a judgement with any of these has not measured everywhere it sampled.
    pub unsettled: usize,
}

/// What a mesh is, judged against the field it was made from. See the module notes.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshJudgement {
    pub triangles: usize,
    /// Edges only one triangle has: a tear.
    pub open_edges: usize,
    /// Edges more than two triangles have: sheets meeting along a line.
    pub overloaded_edges: usize,
    /// Vertices whose triangles fall into more than one fan: surfaces meeting at a point.
    pub pinched_vertices: usize,
    /// Triangles with no area: a repeated corner, or three corners on one line.
    pub degenerate: usize,
    /// Edges whose two triangles cross it the same way, so their windings disagree.
    pub misoriented_edges: usize,
    /// Connected pieces of the mesh.
    pub shells: usize,
    /// Pieces whose signed volume is negative: inside out as a whole.
    pub inward_shells: usize,
    /// Triangles whose normal points against the field's gradient at their middle.
    pub facing_in: usize,
    /// How many triangles that reading was taken at.
    pub facing_checked: usize,
    /// The middle of one triangle that faces in, to look at.
    pub facing_in_at: Option<Point3<f64>>,
    /// The mesh's enclosed volume, signed by its winding.
    pub volume_mm3: f64,
    pub area_mm2: f64,
    /// Mesh to field: how far points of the mesh are from the zero set.
    pub off_surface: Reach,
    /// Field to mesh: how far points of the zero set are from the mesh.
    pub uncovered: Reach,
    /// The grid the zero set was sampled on, in millimetres. A missing piece smaller than this
    /// can fall between samples.
    pub spacing: f64,
}

impl MeshJudgement {
    /// Every edge has two triangles, every vertex one fan, and no triangle is flat.
    pub fn is_closed(&self) -> bool {
        self.triangles > 0
            && self.open_edges == 0
            && self.overloaded_edges == 0
            && self.pinched_vertices == 0
            && self.degenerate == 0
    }

    /// Every winding agrees, every shell encloses positive volume, and every triangle faces away
    /// from the material.
    pub fn is_outward(&self) -> bool {
        self.misoriented_edges == 0 && self.inward_shells == 0 && self.facing_in == 0
    }

    /// The larger of the two one-sided distances: how far apart the mesh and the field's surface
    /// are, as far as the samples saw.
    pub fn hausdorff(&self) -> f64 {
        self.off_surface.worst.max(self.uncovered.worst)
    }

    /// Whether every sample was settled, so [`Self::hausdorff`] is a measurement and not a floor.
    pub fn is_settled(&self) -> bool {
        self.off_surface.unsettled == 0 && self.uncovered.unsettled == 0
    }

    /// The most the mesh's volume can differ from the field's when no point of either surface is
    /// further than [`Self::hausdorff`] from the other: the area swept that far. First order, which
    /// is what a distance a small fraction of the part allows.
    pub fn volume_slack(&self) -> f64 {
        self.hausdorff() * self.area_mm2
    }
}

/// Judges `triangles` over `positions` against `field`, whose surface lies inside `bounds`.
///
/// `within` is the tolerance the mesh was asked to keep: the walk onto the surface settles to a
/// thousandth of it, and the zero set is sampled no finer than half of it. It sets how carefully
/// the distance is read, not what passes -- that is the caller's to compare.
pub fn judge_mesh(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
    within: f64,
) -> MeshJudgement {
    let within = within.max(1e-9);
    let topology = topology(positions, triangles);
    let settle = within * 1e-3;
    let step = within * 1e-2;

    let facing = stride(triangles.len());
    let checked: Vec<(bool, Point3<f64>)> = (0..triangles.len())
        .into_par_iter()
        .step_by(facing)
        .filter_map(|i| {
            let [a, b, c] = corners(positions, &triangles[i]);
            let n = (b - a).cross(&(c - a));
            let middle = Point3::from((a.coords + b.coords + c.coords) / 3.0);
            let gradient = surface_normal(field, middle, step)?;
            (n.norm() > 0.0).then(|| (n.dot(&gradient) < 0.0, middle))
        })
        .collect();
    let facing_in = checked.iter().filter(|x| x.0).count();
    let facing_in_at = checked.iter().find(|x| x.0).map(|x| x.1);

    let off_surface = off_surface(field, positions, triangles, settle, step);

    let spacing = (1.75 * topology.area / MOST_SAMPLES as f64).sqrt().max(within * 0.5);
    let uncovered = uncovered(field, bounds, positions, triangles, spacing, settle, step);

    MeshJudgement {
        triangles: triangles.len(),
        open_edges: topology.open,
        overloaded_edges: topology.overloaded,
        pinched_vertices: topology.pinched,
        degenerate: topology.degenerate,
        misoriented_edges: topology.misoriented,
        shells: topology.shells,
        inward_shells: topology.inward_shells,
        facing_in,
        facing_checked: checked.len(),
        facing_in_at,
        volume_mm3: topology.volume,
        area_mm2: topology.area,
        off_surface,
        uncovered,
        spacing,
    }
}

fn corners(positions: &[Point3<f64>], t: &[u32; 3]) -> [Point3<f64>; 3] {
    [positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]]
}

/// Every `n`th item, so that `count` items give about [`MOST_SAMPLES`].
fn stride(count: usize) -> usize {
    count.div_ceil(MOST_SAMPLES).max(1)
}

struct Topology {
    open: usize,
    overloaded: usize,
    pinched: usize,
    degenerate: usize,
    misoriented: usize,
    shells: usize,
    inward_shells: usize,
    volume: f64,
    area: f64,
}

fn topology(positions: &[Point3<f64>], triangles: &[[u32; 3]]) -> Topology {
    // Every directed edge, as (low, high, forward), sorted so an edge's uses sit together.
    let mut edges: Vec<(u32, u32, bool)> = triangles
        .iter()
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .filter(|(a, b)| a != b)
        .map(|(a, b)| if a < b { (a, b, true) } else { (b, a, false) })
        .collect();
    edges.par_sort_unstable();
    let (mut open, mut overloaded, mut misoriented) = (0, 0, 0);
    for run in edges.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
        match run.len() {
            1 => open += 1,
            2 if run[0].2 == run[1].2 => misoriented += 1,
            2 => {}
            _ => overloaded += 1,
        }
    }

    let degenerate = triangles
        .par_iter()
        .filter(|t| {
            let [a, b, c] = corners(positions, t);
            t[0] == t[1] || t[1] == t[2] || t[2] == t[0] || (b - a).cross(&(c - a)).norm() == 0.0
        })
        .count();

    // A vertex's triangles, grouped into fans by the edges at the vertex they share.
    let mut first = vec![0u32; positions.len() + 1];
    for t in triangles {
        for &v in t {
            first[v as usize + 1] += 1;
        }
    }
    for i in 0..positions.len() {
        first[i + 1] += first[i];
    }
    let mut fill = first.clone();
    let mut around = vec![0u32; first[positions.len()] as usize];
    for (i, t) in triangles.iter().enumerate() {
        for &v in t {
            around[fill[v as usize] as usize] = i as u32;
            fill[v as usize] += 1;
        }
    }
    let pinched = (0..positions.len())
        .into_par_iter()
        .filter(|&v| {
            let fan = &around[first[v] as usize..first[v + 1] as usize];
            fans_at(v as u32, fan, triangles) > 1
        })
        .count();

    // Shells: vertices joined by the triangles they share.
    let mut parent: Vec<u32> = (0..positions.len() as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for t in triangles {
        let a = root(&mut parent, t[0]);
        for &v in &t[1..] {
            let b = root(&mut parent, v);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut shell_volume = std::collections::HashMap::<u32, f64>::new();
    let (mut volume, mut area) = (0.0, 0.0);
    for t in triangles {
        let [a, b, c] = corners(positions, t);
        let v = a.coords.dot(&b.coords.cross(&c.coords)) / 6.0;
        volume += v;
        area += (b - a).cross(&(c - a)).norm() / 2.0;
        *shell_volume.entry(root(&mut parent, t[0])).or_default() += v;
    }
    Topology {
        open,
        overloaded,
        pinched,
        degenerate,
        misoriented,
        shells: shell_volume.len(),
        inward_shells: shell_volume.values().filter(|v| **v < 0.0).count(),
        volume,
        area,
    }
}

/// How many fans the triangles around `v` make: triangles are in one fan when a chain of them,
/// each sharing an edge at `v` with the next, joins them.
fn fans_at(v: u32, fan: &[u32], triangles: &[[u32; 3]]) -> usize {
    if fan.is_empty() {
        return 0;
    }
    // (the vertex across an edge at v, which of the fan's triangles has it)
    let mut spokes: Vec<(u32, usize)> = fan
        .iter()
        .enumerate()
        .flat_map(|(k, &t)| {
            triangles[t as usize].iter().filter(move |&&w| w != v).map(move |&w| (w, k))
        })
        .collect();
    spokes.sort_unstable();
    let mut parent: Vec<usize> = (0..fan.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for pair in spokes.windows(2) {
        if pair[0].0 == pair[1].0 {
            let (a, b) = (root(&mut parent, pair[0].1), root(&mut parent, pair[1].1));
            parent[b] = a;
        }
    }
    (0..fan.len()).filter(|&k| root(&mut parent, k) == k).count()
}

/// The distance from `p` to the field's surface, as a bound that cannot read short: to a point
/// walked onto the surface, and never less than the field's own reading. `None` when the walk
/// does not settle.
fn to_surface(field: &dyn ScalarField, p: Point3<f64>, settle: f64, step: f64) -> Option<f64> {
    let reading = field.distance(p).abs();
    if reading <= settle {
        return Some(reading);
    }
    let q = walk_to_surface(field, p, settle, step)?;
    Some((p - q).norm().max(reading))
}

fn off_surface(
    field: &(dyn ScalarField + Sync),
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
    settle: f64,
    step: f64,
) -> Reach {
    // A vertex, the three edge middles and the middle of each triangle: seven points a triangle,
    // with every vertex and edge counted from each triangle that has it, which costs repeats and
    // saves building an edge list for a sampling that is strided anyway.
    let every = stride(triangles.len() * 7);
    let samples: Vec<Point3<f64>> = triangles
        .iter()
        .flat_map(|t| {
            let [a, b, c] = corners(positions, t);
            let mid = |x: Point3<f64>, y: Point3<f64>| Point3::from((x.coords + y.coords) / 2.0);
            [a, b, c, mid(a, b), mid(b, c), mid(c, a), Point3::from((a.coords + b.coords + c.coords) / 3.0)]
        })
        .step_by(every)
        .collect();
    reach(samples.par_iter().map(|&p| (p, to_surface(field, p, settle, step))).collect())
}

fn uncovered(
    field: &(dyn ScalarField + Sync),
    (lo, hi): (Point3<f64>, Point3<f64>),
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
    spacing: f64,
    settle: f64,
    step: f64,
) -> Reach {
    // Cells halved from one cube over the bounds, kept while the field allows the surface in
    // them: a lower bound larger than the half-diagonal says the surface is further than any
    // point of the cell.
    let extent = hi - lo;
    let mut size = extent.max().max(spacing);
    let mut cells = vec![Point3::from((lo.coords + hi.coords) / 2.0)];
    loop {
        let reach = size * 3f64.sqrt() / 2.0;
        cells.retain(|c| field.distance(*c).abs() <= reach);
        if size <= spacing || cells.is_empty() {
            break;
        }
        size /= 2.0;
        let q = size / 2.0;
        cells = cells
            .par_iter()
            .flat_map_iter(|c| {
                (0..8).map(move |k| {
                    c + Vector3::new(
                        if k & 1 == 0 { -q } else { q },
                        if k & 2 == 0 { -q } else { q },
                        if k & 4 == 0 { -q } else { q },
                    )
                })
            })
            .collect();
    }
    let every = stride(cells.len());
    let bvh = Bvh::build(positions, triangles);
    let nearest = |p: Point3<f64>| {
        let mut best = f64::INFINITY;
        bvh.nearest(p, |t| {
            let tri = &triangles[t as usize];
            let [a, b, c] = corners(positions, tri);
            let (q, _) = closest_point_on_triangle(p, a, b, c, t as usize, tri);
            let d = (p - q).norm_squared();
            best = best.min(d);
            d
        });
        best.sqrt()
    };
    reach(
        cells
            .par_iter()
            .step_by(every)
            .map(|&c| match walk_to_surface(field, c, settle, step) {
                Some(q) => (q, Some(nearest(q))),
                None => (c, None),
            })
            .collect(),
    )
}

fn reach(measured: Vec<(Point3<f64>, Option<f64>)>) -> Reach {
    let samples = measured.len();
    let unsettled = measured.iter().filter(|(_, d)| d.is_none()).count();
    let mut settled: Vec<(f64, Point3<f64>)> =
        measured.into_iter().filter_map(|(p, d)| d.map(|d| (d, p))).collect();
    if settled.is_empty() {
        return Reach { samples, unsettled, ..Reach::default() };
    }
    let (worst, at) = settled.iter().copied().fold((f64::NEG_INFINITY, None), |best, (d, p)| {
        if d > best.0 { (d, Some(p)) } else { best }
    });
    let k = ((settled.len() - 1) as f64 * 0.99).round() as usize;
    let (_, p99, _) = settled.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
    Reach { worst, at, p99: p99.0, samples, unsettled }
}

#[cfg(test)]
#[path = "mesh_judge_tests.rs"]
mod tests;
