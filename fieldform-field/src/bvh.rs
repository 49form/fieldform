// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A bounding volume hierarchy over triangles, for nearest-surface queries.
//!
//! # Why this is an accuracy feature, not just a speed one
//!
//! [`CageField`](crate::CageField) answers distance queries against a triangulation of the
//! limit surface, and how closely that triangulation follows the real surface is what sets the
//! *shading* of anything composed from it — the gradient of a distance-to-triangles field is
//! piecewise constant across each triangle and jumps at the seams, so too coarse a
//! triangulation reads as banding on a surface that should be smooth.
//!
//! Measured on a 1000 mm cube (`normal_accuracy`), the worst shading-normal error runs 8.99
//! degrees at subdivision level 2, 4.90 at level 3, 1.48 at level 4 and 0.18 at level 5. The
//! accuracy needed is therefore known; what made it unaffordable was that every query scanned
//! every triangle, so each level cost 4x the last. This is what buys those levels back.
//!
//! # Shape
//!
//! Binary, split at the median centroid along the widest axis, leaves of at most four
//! triangles. Queries descend nearest-child-first and prune any node whose box is already
//! farther than the best hit so far, which is what turns a linear scan into a logarithmic one.

use nalgebra::Point3;

/// Triangles per leaf. Small enough that a leaf scan is cheap, large enough that the tree
/// does not become mostly pointer-chasing.
const LEAF_SIZE: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Aabb {
    lo: Point3<f64>,
    hi: Point3<f64>,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            lo: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            hi: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    fn add(&mut self, p: Point3<f64>) {
        self.lo = Point3::new(self.lo.x.min(p.x), self.lo.y.min(p.y), self.lo.z.min(p.z));
        self.hi = Point3::new(self.hi.x.max(p.x), self.hi.y.max(p.y), self.hi.z.max(p.z));
    }

    /// Squared distance from `p` to the box; zero inside it.
    ///
    /// This is the pruning test, so it must never *over*-estimate — a box reported as farther
    /// than it is would let the search skip the triangle that actually wins.
    fn distance_squared(&self, p: Point3<f64>) -> f64 {
        let dx = (self.lo.x - p.x).max(0.0).max(p.x - self.hi.x);
        let dy = (self.lo.y - p.y).max(0.0).max(p.y - self.hi.y);
        let dz = (self.lo.z - p.z).max(0.0).max(p.z - self.hi.z);
        dx * dx + dy * dy + dz * dz
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Aabb,
    /// For a leaf, where its triangles start in `order`. For an interior node, the index of
    /// its left child; the right child is the next node after that subtree.
    first: u32,
    /// Triangle count for a leaf; zero marks an interior node.
    count: u32,
    /// Interior nodes only: index of the right child.
    right: u32,
}

/// A bounding volume hierarchy over a fixed set of triangles.
pub(crate) struct Bvh {
    nodes: Vec<Node>,
    /// Triangle indices, permuted so each leaf owns a contiguous run.
    order: Vec<u32>,
}

impl Bvh {
    /// Builds a hierarchy over the given triangles.
    pub(crate) fn build(positions: &[Point3<f64>], triangles: &[[u32; 3]]) -> Self {
        let mut order: Vec<u32> = (0..triangles.len() as u32).collect();
        let centroids: Vec<Point3<f64>> = triangles
            .iter()
            .map(|t| {
                let a = positions[t[0] as usize].coords;
                let b = positions[t[1] as usize].coords;
                let c = positions[t[2] as usize].coords;
                Point3::from((a + b + c) / 3.0)
            })
            .collect();

        let mut nodes = Vec::new();
        if !order.is_empty() {
            build_node(
                positions,
                triangles,
                &centroids,
                &mut order,
                0,
                triangles.len(),
                &mut nodes,
            );
        }
        Self { nodes, order }
    }

    /// The hierarchy as two flat arrays, for a shader that has to traverse it itself.
    ///
    /// **Already flat, which is the only reason this is a method and not a rewrite.** The nodes
    /// are a `Vec` indexed by position and the leaf triangles are a permutation, so there is no
    /// pointer chasing to unpick — the same layout a GPU wants, arrived at for the CPU's own
    /// cache reasons.
    ///
    /// Bounds and links come apart because a buffer holds one scalar type: six floats per node,
    /// then three integers per node. `count == 0` marks an interior node, exactly as it does
    /// here.
    pub(crate) fn packed(&self) -> (Vec<f32>, Vec<u32>) {
        let mut bounds = Vec::with_capacity(self.nodes.len() * 6);
        let mut links = Vec::with_capacity(self.nodes.len() * 3);
        for node in &self.nodes {
            for v in [
                node.bounds.lo.x,
                node.bounds.lo.y,
                node.bounds.lo.z,
                node.bounds.hi.x,
                node.bounds.hi.y,
                node.bounds.hi.z,
            ] {
                // An empty node's bounds are infinite, which is a number a shader cannot use.
                // Clamped to the largest finite float, which bounds nothing differently: a ray
                // is never further away than that.
                bounds.push(if v.is_finite() { v as f32 } else { v.signum() as f32 * f32::MAX });
            }
            links.extend_from_slice(&[node.first, node.count, node.right]);
        }
        (bounds, links)
    }

    /// The leaf triangle order, which the packed nodes index into.
    pub(crate) fn order(&self) -> &[u32] {
        &self.order
    }

    /// Visits the triangles that could hold the nearest point to `p`, nearest first.
    ///
    /// `eval` is called with a triangle index and must return the squared distance from `p` to
    /// that triangle; the search uses the smallest value seen so far to prune. Keeping the
    /// triangle maths in the caller is deliberate — the caller also needs to know *which*
    /// feature of the triangle won, which matters for the inside/outside test.
    pub(crate) fn nearest<F>(&self, p: Point3<f64>, mut eval: F)
    where
        F: FnMut(u32) -> f64,
    {
        if self.nodes.is_empty() {
            return;
        }
        let mut best = f64::INFINITY;
        // An explicit stack rather than recursion: this runs once per sample point when
        // meshing, which is millions of times.
        let mut stack: Vec<(u32, f64)> = Vec::with_capacity(64);
        stack.push((0, self.nodes[0].bounds.distance_squared(p)));

        while let Some((index, bound)) = stack.pop() {
            // The best hit may have improved since this node was pushed, so re-check.
            if bound >= best {
                continue;
            }
            let node = self.nodes[index as usize];

            if node.count > 0 {
                let start = node.first as usize;
                for &tri in &self.order[start..start + node.count as usize] {
                    let d = eval(tri);
                    if d < best {
                        best = d;
                    }
                }
                continue;
            }

            let left = node.first;
            let right = node.right;
            let dl = self.nodes[left as usize].bounds.distance_squared(p);
            let dr = self.nodes[right as usize].bounds.distance_squared(p);

            // Push the farther child first so the nearer one is popped first: descending
            // toward the likely winner tightens `best` early, which is what makes the pruning
            // above actually prune.
            if dl <= dr {
                stack.push((right, dr));
                stack.push((left, dl));
            } else {
                stack.push((left, dl));
                stack.push((right, dr));
            }
        }
    }

    /// Number of nodes, for diagnostics and tests.
    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Builds one node over `order[start..end]`, appending it and its children to `nodes`.
/// Returns the new node's index.
fn build_node(
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
    centroids: &[Point3<f64>],
    order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> u32 {
    let mut bounds = Aabb::empty();
    for &tri in &order[start..end] {
        for &v in &triangles[tri as usize] {
            bounds.add(positions[v as usize]);
        }
    }

    let index = nodes.len() as u32;
    let count = end - start;

    if count <= LEAF_SIZE {
        nodes.push(Node {
            bounds,
            first: start as u32,
            count: count as u32,
            right: 0,
        });
        return index;
    }

    // Split along whichever axis the centroids spread over most: splitting a flat direction
    // leaves both children with nearly the parent's box and prunes nothing.
    let mut centroid_bounds = Aabb::empty();
    for &tri in &order[start..end] {
        centroid_bounds.add(centroids[tri as usize]);
    }
    let extent = centroid_bounds.hi - centroid_bounds.lo;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    let mid = start + count / 2;
    order[start..end].select_nth_unstable_by(count / 2, |a, b| {
        let ca = centroids[*a as usize][axis];
        let cb = centroids[*b as usize][axis];
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reserve this node's slot before building children, so the index is stable.
    nodes.push(Node {
        bounds,
        first: 0,
        count: 0,
        right: 0,
    });
    let left = build_node(positions, triangles, centroids, order, start, mid, nodes);
    let right = build_node(positions, triangles, centroids, order, mid, end, nodes);
    nodes[index as usize].first = left;
    nodes[index as usize].right = right;
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of triangles in the z = 0 plane, so the nearest one to any point is knowable.
    fn grid(n: usize) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                positions.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        for j in 0..n {
            for i in 0..n {
                triangles.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                triangles.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        (positions, triangles)
    }

    /// Squared distance from a point to a triangle, brute force, for comparison.
    fn tri_distance_squared(p: Point3<f64>, t: &[u32; 3], positions: &[Point3<f64>]) -> f64 {
        // Good enough for a planar test grid: nearest of the three vertices and the centroid.
        let a = positions[t[0] as usize];
        let b = positions[t[1] as usize];
        let c = positions[t[2] as usize];
        let centroid = Point3::from((a.coords + b.coords + c.coords) / 3.0);
        [a, b, c, centroid]
            .iter()
            .map(|q| (q - p).norm_squared())
            .fold(f64::INFINITY, f64::min)
    }

    /// The property that matters: pruning must never change the answer. A hierarchy that is
    /// merely fast and occasionally wrong would show up as isolated wrong-facing pixels, which
    /// is exactly the kind of bug that survives casual inspection.
    #[test]
    fn the_hierarchy_finds_the_same_nearest_triangle_as_a_full_scan() {
        let (positions, triangles) = grid(12);
        let bvh = Bvh::build(&positions, &triangles);

        for k in 0..200 {
            let p = Point3::new(
                (k as f64 * 0.37) % 14.0 - 1.0,
                (k as f64 * 0.71) % 14.0 - 1.0,
                (k as f64 * 0.13) % 5.0 - 2.5,
            );

            let brute = triangles
                .iter()
                .map(|t| tri_distance_squared(p, t, &positions))
                .fold(f64::INFINITY, f64::min);

            let mut found = f64::INFINITY;
            bvh.nearest(p, |i| {
                let d = tri_distance_squared(p, &triangles[i as usize], &positions);
                found = found.min(d);
                d
            });

            assert!(
                (found - brute).abs() < 1e-12,
                "at {p:?} the hierarchy found {found} but a full scan found {brute}"
            );
        }
    }

    /// The point of the thing: it must visit far fewer than every triangle.
    #[test]
    fn the_hierarchy_visits_a_small_fraction_of_the_triangles() {
        let (positions, triangles) = grid(24);
        let bvh = Bvh::build(&positions, &triangles);

        let p = Point3::new(12.0, 12.0, 3.0);
        let mut visited = 0;
        bvh.nearest(p, |i| {
            visited += 1;
            tri_distance_squared(p, &triangles[i as usize], &positions)
        });

        assert!(
            visited * 4 < triangles.len(),
            "visited {visited} of {} triangles, which is not much of a saving",
            triangles.len()
        );
    }

    #[test]
    fn every_triangle_appears_exactly_once() {
        let (positions, triangles) = grid(7);
        let bvh = Bvh::build(&positions, &triangles);
        let mut seen = vec![0usize; triangles.len()];
        // Force a full traversal by never letting the search prune.
        bvh.nearest(Point3::origin(), |i| {
            seen[i as usize] += 1;
            f64::INFINITY
        });
        assert!(seen.iter().all(|&c| c == 1), "{seen:?}");
    }

    #[test]
    fn an_empty_set_is_not_a_panic() {
        let bvh = Bvh::build(&[], &[]);
        assert_eq!(bvh.node_count(), 0);
        bvh.nearest(Point3::origin(), |_| panic!("nothing to visit"));
    }
}
