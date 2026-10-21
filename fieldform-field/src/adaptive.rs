// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Adaptive meshing: an octree that spends cells where the surface curves and saves them where
//! it does not.
//!
//! # Why the uniform mesher is not enough
//!
//! [`crate::mesh_field`] samples one grid over the whole box, so every part of the surface gets
//! the same cell whether it needs it or not. Measured on a 40 mm cube
//! (`examples/edge_fidelity.rs`), that lands **all** of the error on creases and **none** on
//! flats: a face meshes exactly at any resolution, while a crease is wrong by 0.43 of a cell and
//! only improves by making every cell smaller. So raising the resolution to sharpen an edge also
//! pays for flats that were already exact.
//!
//! Memory says the same thing louder. The uniform mesher allocates one sample per grid corner,
//! so 512 cells is 513³ × 8 bytes ≈ 1.1 GB and 1024 cells is 8.6 GB. An octree never holds the
//! cells it did not descend into.
//!
//! # And it is where the non-manifold edges get fixed
//!
//! One vertex per cell cannot represent two surface sheets crossing that cell, so an edge there
//! ends up used by more triangles than a surface allows. Measured on a real document, *every*
//! irregular edge in a uniform export was of this kind — 1319 of them at 0.4 mm, and not one
//! hole. A uniform grid has nowhere to put the fix; a tree can simply subdivide until the cell
//! holds one sheet, which is what [`is_manifold`] decides.
//!
//! # What decides a subdivision
//!
//! Four tests, and a cell subdivides if **any** says so: it is not manifold, the surface turns
//! too far inside it, it sags too far from flat, or it is longer than a cap. They are separate
//! because they fail in different places — see [`MeshTolerance`].
//!
//! # What this does not do
//!
//! It does not guarantee that a feature smaller than a cell is found. The flatness tests are
//! local: they read the surface where it crosses a cell's edges, so a bump entirely inside a
//! cell, touching none of them, is invisible to them. [`MeshTolerance::min_cell_mm`] and
//! [`FORCED_DEPTH`] bound that, and they are the honest controls for it rather than an
//! afterthought — an adaptive mesher claiming otherwise would be claiming to have solved
//! sampling.
//!
//! **What that cost, and the one case now covered.** A sheet lattice is *nothing but* features
//! thinner than a cell: a 2 mm wall passes between two corners of a 21 mm cell with both of them
//! outside it, so whether any wall is found at all comes down to where the grid falls. On the
//! owner's conformal Schwarz P part, meshed at Coarse, that was the difference between 130 576
//! triangles and **nothing**, decided by growing the box by 28 mm
//! (`examples/surface_edit_tessellation.rs`). So a field may now declare its thinnest feature
//! through [`ScalarField::feature_size`], and a cell that might hold surface keeps dividing
//! until it is no larger than that — at which point no wall fits between the ends of an edge.
//! It is not a general answer to sampling; it is the field saying what it knows and the mesher
//! believing it. Measured on the same part, at Coarse, over six boxes: 130 420 to 131 732
//! triangles where before it ranged 0 to 130 760, for 4% more time. On a gyroid cube that was
//! never missing walls (`examples/capped_lattice.rs`) it changes neither count nor time.

use nalgebra::{Matrix3, Point3, Vector3};

use crate::{FieldMesh, ScalarField};

/// How closely a mesh must follow the field — defined in `fieldform-scene`, so that a block
/// saved in a document can hold one, and re-exported here where the mesher reads it.
pub use fieldform_scene::MeshTolerance;

/// The depth every cell is taken to before flatness is consulted.
///
/// The flatness tests read the surface only where it crosses a cell's *edges*, so they are blind
/// to anything fitting inside a cell without touching one. At the root that is the whole part.
/// Descending unconditionally first stops a single early "this looks flat" from discarding the
/// model — cheap, since these levels hold at most 8³ cells.
pub(crate) const FORCED_DEPTH: u32 = 3;

/// One cell of the tree.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub lo: Point3<f64>,
    /// Cubic, so one number. Keeping cells cubic lets an octant index describe a child
    /// completely.
    pub size: f64,
    pub depth: u32,
}

impl Cell {
    fn hi(&self) -> Point3<f64> {
        Point3::new(
            self.lo.x + self.size,
            self.lo.y + self.size,
            self.lo.z + self.size,
        )
    }

    /// Corner `c`, where bit 0 is +x, bit 1 is +y, bit 2 is +z.
    fn corner(&self, c: usize) -> Point3<f64> {
        Point3::new(
            self.lo.x + if c & 1 != 0 { self.size } else { 0.0 },
            self.lo.y + if c & 2 != 0 { self.size } else { 0.0 },
            self.lo.z + if c & 4 != 0 { self.size } else { 0.0 },
        )
    }

    /// The eight children, in the same bit order as [`Cell::corner`].
    fn children(&self) -> [Cell; 8] {
        let h = self.size * 0.5;
        std::array::from_fn(|c| Cell {
            lo: Point3::new(
                self.lo.x + if c & 1 != 0 { h } else { 0.0 },
                self.lo.y + if c & 2 != 0 { h } else { 0.0 },
                self.lo.z + if c & 4 != 0 { h } else { 0.0 },
            ),
            size: h,
            depth: self.depth + 1,
        })
    }
}

/// What one leaf knows about the surface inside it.
#[derive(Debug, Clone)]
pub struct Leaf {
    /// The cell the leaf fills. The traversal carries each cell beside the tree rather than
    /// reading it here, so only the tests' depth count reads it, and only the tests store it.
    #[cfg(test)]
    pub cell: Cell,
    /// One vertex per surface sheet in this cell, each clamped into it.
    ///
    /// Almost always one. Two where the cell holds two sheets — the case a single vertex
    /// misrepresents, and the reason an export reported edges belonging to more triangles than
    /// a surface allows.
    pub vertices: Vec<Point3<f64>>,
    pub normals: Vec<Vector3<f64>>,
    /// Which sheet each of the twelve edges belongs to, so an edge reaching into this cell is
    /// given the vertex for *its* sheet rather than the cell's only one.
    pub of_edge: [u8; 12],
    /// Signs at the eight corners: `true` where the field is inside.
    pub inside: [bool; 8],
}

/// The tree.
#[derive(Debug)]
pub enum Tree {
    /// No surface here, and which side of it this region is on.
    ///
    /// The side is carried rather than inferred because the traversal asks empty regions for
    /// their sign: an edge with an empty cell on one side still needs to know whether that side
    /// is solid.
    Empty { inside: bool },
    Leaf(Box<Leaf>),
    Branch(Box<[Tree; 8]>),
}

impl Tree {
    /// The depth of every leaf that carries surface, appended to `out`.
    ///
    /// Nothing outside the tests asks how deep the leaves went or how many there are — the
    /// mesher walks the tree for itself — so this and [`Self::surface_leaves`] are the tests'
    /// view of how far a tolerance refined, and only the tests compile them.
    #[cfg(test)]
    fn leaf_depths(&self, out: &mut Vec<u32>) {
        match self {
            Tree::Empty { .. } => {}
            Tree::Leaf(l) => out.push(l.cell.depth),
            Tree::Branch(kids) => kids.iter().for_each(|k| k.leaf_depths(out)),
        }
    }

    /// How many leaves carry surface. Test-only for the same reason as `leaf_depths`.
    #[cfg(test)]
    fn surface_leaves(&self) -> usize {
        let mut d = Vec::new();
        self.leaf_depths(&mut d);
        d.len()
    }

    fn is_branch(&self) -> bool {
        matches!(self, Tree::Branch(_))
    }
}

/// Whether a cell's sign pattern can be represented by a single vertex.
///
/// **The test that fixes non-manifold edges.** Dual contouring puts one vertex in a cell and
/// joins vertices across sign-changing edges. That is only sound while the surface inside the
/// cell is *one sheet*: where two sheets pass through, one vertex has to stand for both, and the
/// edges around it end up shared by more triangles than a surface permits.
///
/// One sheet, in terms of the eight corner signs, means the inside corners are connected to each
/// other and the outside corners are connected to each other — **through faces**, not through
/// diagonals. Two inside corners touching only at a cube diagonal are the ambiguous case: the
/// field may join them or may not, and the cell cannot tell. Subdividing asks the question again
/// on a smaller cell, where it is usually no longer ambiguous.
///
/// A cell with no sign change at all is trivially manifold; it has no surface to misrepresent.
pub(crate) fn is_manifold(inside: [bool; 8]) -> bool {
    connected(&inside, true) && connected(&inside, false)
}

/// Whether the corners on one side of the surface form one face-connected group.
fn connected(inside: &[bool; 8], want: bool) -> bool {
    let members: Vec<usize> = (0..8).filter(|c| inside[*c] == want).collect();
    if members.len() <= 1 {
        return true;
    }
    // Flood fill from the first member. Two corners share a face when their indices differ in
    // exactly one bit — which is what makes this the *face* connectivity rather than the
    // diagonal one, and the whole point of the test.
    let mut seen = [false; 8];
    let mut stack = vec![members[0]];
    seen[members[0]] = true;
    let mut count = 1;
    while let Some(c) = stack.pop() {
        for bit in 0..3 {
            let n = c ^ (1 << bit);
            if inside[n] == want && !seen[n] {
                seen[n] = true;
                count += 1;
                stack.push(n);
            }
        }
    }
    count == members.len()
}

/// The 12 edges of a cell as corner-index pairs, in the same bit order as [`Cell::corner`].
const CELL_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Which surface sheet each of a cell's twelve edges belongs to.
///
/// **This is what a single vertex per cell cannot express.** Dual contouring puts one vertex in a
/// cell and joins vertices across sign-changing edges, which is sound while the surface inside
/// that cell is one sheet. Where two pass through, one vertex has to stand for both, and every
/// edge around it ends up shared by more triangles than a surface permits — the non-manifold
/// edges an export reports.
///
/// Two sign-changing edges belong to the same sheet when their *inside* ends are connected
/// through faces of the cell. That is the same connectivity [`is_manifold`] tests, used to
/// separate rather than to refuse: a cell with two components of inside corners has two sheets,
/// and giving each its own vertex is what Ju and Schaefer's manifold dual contouring does.
///
/// Returns one label per edge, or `NO_SHEET` where the edge does not cross, and the number of
/// sheets found.
const NO_SHEET: u8 = u8::MAX;

fn sheets_of(inside: &[bool; 8]) -> ([u8; 12], usize) {
    // Label the inside corners by connected component, through faces only. Corners meeting at a
    // cube diagonal are *not* joined, which is exactly the ambiguity being separated.
    let mut label = [NO_SHEET; 8];
    let mut sheets = 0u8;
    for start in 0..8 {
        if !inside[start] || label[start] != NO_SHEET {
            continue;
        }
        let mut stack = vec![start];
        label[start] = sheets;
        while let Some(c) = stack.pop() {
            for bit in 0..3 {
                let n = c ^ (1 << bit);
                if inside[n] && label[n] == NO_SHEET {
                    label[n] = sheets;
                    stack.push(n);
                }
            }
        }
        sheets += 1;
    }

    // An edge crosses when its ends differ, and it belongs to the sheet wrapping its inside end.
    let mut of_edge = [NO_SHEET; 12];
    for (e, &(a, b)) in CELL_EDGES.iter().enumerate() {
        if inside[a] == inside[b] {
            continue;
        }
        of_edge[e] = if inside[a] { label[a] } else { label[b] };
    }
    (of_edge, sheets as usize)
}

/// Which of a cell's twelve edges runs along `axis` at the corner `ring_child` describes.
///
/// The traversal knows a minimal edge by its axis and which corner of a surrounding cell it sits
/// on. A leaf coarser than that edge still contains it, along the edge at the same corner — so
/// this is what lets a coarse neighbour say *which of its sheets* the edge belongs to, rather
/// than offering the one vertex it would have had.
fn edge_at_corner(axis: usize, want_u: usize, want_v: usize) -> usize {
    // Derived from `CELL_EDGES`'s own order, which groups the twelve by direction: four along
    // x, then y, then z, each varying by the two perpendicular bits in that order.
    match axis {
        0 => want_u + 2 * want_v,
        1 => 4 + want_v + 2 * want_u,
        _ => 8 + want_u + 2 * want_v,
    }
}

/// What a cell's sign-changing edges say about the surface in it.
struct Hermite {
    /// One vertex per surface sheet, and the normal there.
    ///
    /// Usually one. Two where the cell holds two sheets, which is the case a single vertex
    /// misrepresents — see [`sheets_of`].
    vertices: Vec<Point3<f64>>,
    normals: Vec<Vector3<f64>>,
    /// Which sheet each of the twelve edges belongs to, or [`NO_SHEET`].
    of_edge: [u8; 12],
    inside: [bool; 8],
    /// The widest angle between any two crossing normals, in radians.
    spread: f64,
    /// The furthest any crossing sits from the plane through their mean, in millimetres.
    sag: f64,
}

fn hermite(field: &(dyn ScalarField + Sync), cell: &Cell) -> Option<Hermite> {
    let values: [f64; 8] = std::array::from_fn(|c| field.distance(cell.corner(c)));
    let inside: [bool; 8] = std::array::from_fn(|c| values[c] <= 0.0);
    let (of_edge, sheets) = sheets_of(&inside);
    if sheets == 0 {
        return None;
    }

    // Solved per sheet. The crossings of one sheet say nothing about where another's vertex
    // belongs, and averaging them together is precisely the mistake that puts one vertex
    // between two surfaces.
    let mut a = vec![Matrix3::zeros(); sheets];
    let mut b = vec![Vector3::zeros(); sheets];
    let mut mass = vec![Vector3::zeros(); sheets];
    let mut count = vec![0usize; sheets];
    let mut normals_of: Vec<Vec<Vector3<f64>>> = vec![Vec::new(); sheets];
    let mut points_of: Vec<Vec<Point3<f64>>> = vec![Vec::new(); sheets];

    for (e, &(ea, eb)) in CELL_EDGES.iter().enumerate() {
        let sheet = of_edge[e];
        if sheet == NO_SHEET {
            continue;
        }
        let s = sheet as usize;
        let (pa, pb) = (cell.corner(ea), cell.corner(eb));
        let p = crate::mesh::refine_crossing(field, pa, pb, values[ea], values[eb]);
        let n = crate::mesh::surface_normal(field, p, cell.size * 1e-3);
        a[s] += n * n.transpose();
        b[s] += n * n.dot(&p.coords);
        mass[s] += p.coords;
        count[s] += 1;
        normals_of[s].push(n);
        points_of[s].push(p);
    }

    let mut vertices = Vec::with_capacity(sheets);
    let mut normals = Vec::with_capacity(sheets);
    let mut spread: f64 = 0.0;
    let mut sag: f64 = 0.0;
    for s in 0..sheets {
        if count[s] == 0 {
            // A sheet whose corners are inside but whose edges never cross is one the cell only
            // touches. It still needs a vertex, or an edge reaching for it finds nothing — the
            // centre is the honest placeholder, and nothing will ask unless an edge does.
            vertices.push(Point3::from(
                (cell.lo.coords + cell.hi().coords) * 0.5,
            ));
            normals.push(Vector3::z());
            continue;
        }
        let mean = Point3::from(mass[s] / count[s] as f64);
        let solved = crate::mesh::solve_qef(a[s], b[s], mean);
        vertices.push(Point3::new(
            solved.x.clamp(cell.lo.x, cell.lo.x + cell.size),
            solved.y.clamp(cell.lo.y, cell.lo.y + cell.size),
            solved.z.clamp(cell.lo.z, cell.lo.z + cell.size),
        ));

        // The widest disagreement between any two normals, not the spread about their mean. A
        // crease shows as two tight clusters pointing different ways, whose mean is a direction
        // neither takes — averaging first would report a small deviation for exactly the
        // feature this test exists to find.
        let ns = &normals_of[s];
        for i in 0..ns.len() {
            for j in (i + 1)..ns.len() {
                spread = spread.max(ns[i].dot(&ns[j]).clamp(-1.0, 1.0).acos());
            }
        }
        let mean_normal = ns
            .iter()
            .fold(Vector3::zeros(), |acc: Vector3<f64>, n| acc + n)
            .try_normalize(1e-12)
            .unwrap_or_else(Vector3::z);
        sag = sag.max(
            points_of[s]
                .iter()
                .map(|p| (p - mean).dot(&mean_normal).abs())
                .fold(0.0f64, f64::max),
        );
        normals.push(mean_normal);
    }

    Some(Hermite {
        vertices,
        normals,
        of_edge,
        inside,
        spread,
        sag,
    })
}

/// [`build_cancellable`] with nothing to stop it.
///
/// The mesher builds its own tree, through `build_cancellable`, so only the tests here ask for
/// one directly, and only the tests compile this (decision 1.4). It was the one thing that
/// handed a `Tree`, `Cell` or `Leaf` to another crate, so the root no longer names them.
#[cfg(test)]
fn build(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tol: &MeshTolerance,
) -> (Tree, Cell) {
    build_cancellable(field, bounds, tol, &crate::Cancel::default())
}

/// Builds the tree over `bounds`, and returns it with the cubic root it was built on, giving up
/// as soon as `cancel` is set. What comes back from a cancelled build is a tree with holes in it
/// and is only fit to be thrown away, which is what every caller does.
pub(crate) fn build_cancellable(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tol: &MeshTolerance,
    cancel: &crate::Cancel,
) -> (Tree, Cell) {
    let (lo, hi) = bounds;
    let extent = hi - lo;
    let size = extent.x.max(extent.y).max(extent.z);
    // Centred, so the room a non-cubic box gains is shared rather than added to one side.
    let centre = Point3::from((lo.coords + hi.coords) * 0.5);
    let half = size * 0.5;
    let root = Cell {
        lo: Point3::new(centre.x - half, centre.y - half, centre.z - half),
        size,
        depth: 0,
    };
    // Infinite when the field declares nothing, which leaves the rule below switched off.
    let feature = field.feature_size().unwrap_or(f64::INFINITY);
    let mut tree = descend(field, &root, tol, feature, cancel);
    balance(&mut tree, root, field, tol, feature, cancel);
    (tree, root)
}

fn descend(
    field: &(dyn ScalarField + Sync),
    cell: &Cell,
    tol: &MeshTolerance,
    feature: f64,
    cancel: &crate::Cancel,
) -> Tree {
    // **Checked at every cell**, which is where the mesher spends its time — a Fine mesh is
    // most of a minute of this recursion and nothing else. An atomic load against a cell's worth
    // of interval and Hermite work is lost in the noise, and it is what lets a stale run stop
    // within a cell of being told to rather than at the end.
    if cancel.is_cancelled() {
        return Tree::Empty { inside: false };
    }
    // Whether the surface can be in here at all, from the field's own interval bound. This is
    // the step that makes an octree cheaper than a grid rather than merely differently shaped.
    let (lo_v, hi_v) = field.interval(cell.lo, cell.hi());
    if lo_v > 0.0 {
        return Tree::Empty { inside: false };
    }
    if hi_v < 0.0 {
        return Tree::Empty { inside: true };
    }

    let too_small = cell.size * 0.5 < tol.min_cell_mm;
    if cell.depth < FORCED_DEPTH && !too_small {
        return branch(field, cell, tol, feature, cancel);
    }

    match hermite(field, cell) {
        // **Larger than the field's thinnest feature, and the surface may be here: look
        // closer.** No corner changed sign, but a wall thinner than this cell could lie between
        // two corners with both outside it — and a sheet lattice is nothing but such walls, so
        // taking the guess below would lose them all at once, by where the grid fell. Only the
        // field's own declaration switches this on: a solid with no thin features keeps the
        // guess, and with it the cost the note below measured.
        None if !too_small && cell.size > feature => branch(field, cell, tol, feature, cancel),
        // The interval said the surface might be here and no corner changed sign, so either it
        // misses this cell or it passes through too thinly to touch an edge.
        //
        // **Given a vertex anyway, rather than declared empty.** A cell with no vertex still
        // borders its neighbours' edges, and where a finer neighbour finds a crossing on a
        // sub-segment of an edge this cell only sampled at its ends, the quad has nowhere to
        // anchor and the surface opens — a crack exactly along a level change, which cost 9312
        // holes on a lattice.
        //
        // Subdividing instead was tried and is what a first reading suggests, but the interval
        // bound is loose: for a plane it reports "maybe" for a whole shell of cells that do not
        // contain the surface at all, so every one of them recurses to the floor. It took the
        // test suite from 0.01 s to 40 s and produced nothing.
        //
        // This costs nothing when it is wrong. A vertex is written to the mesh only when an
        // edge actually reaches for it, so a cell the surface really misses is never indexed
        // and the guess is discarded unread. Placed by one Newton step from the centre, which
        // is where the surface would be if it is here at all.
        None => {
            let centre = Point3::from((cell.lo.coords + cell.hi().coords) * 0.5);
            let value = field.distance(centre);
            let normal = crate::mesh::surface_normal(field, centre, cell.size * 1e-3);
            let guess = centre - normal * value;
            Tree::Leaf(Box::new(Leaf {
                #[cfg(test)]
                cell: *cell,
                vertices: vec![Point3::new(
                    guess.x.clamp(cell.lo.x, cell.lo.x + cell.size),
                    guess.y.clamp(cell.lo.y, cell.lo.y + cell.size),
                    guess.z.clamp(cell.lo.z, cell.lo.z + cell.size),
                )],
                normals: vec![normal],
                // No edge crosses, so no edge belongs to a sheet. An edge reaching in from a
                // finer neighbour falls back to the one vertex, which is the guess above.
                of_edge: [NO_SHEET; 12],
                // No corner changed sign, so every corner is on the same side and the cell
                // reports that honestly. Nothing here invents a crossing.
                inside: [value <= 0.0; 8],
            }))
        }
        Some(h) => {
            let ambiguous = !is_manifold(h.inside);
            let turns = h.spread > tol.angle_deg.to_radians();
            let sags = h.sag > tol.chord_mm;
            let long = tol.max_edge_mm > 0.0 && cell.size > tol.max_edge_mm;
            if !too_small && (ambiguous || turns || sags || long) {
                return branch(field, cell, tol, feature, cancel);
            }
            Tree::Leaf(Box::new(Leaf {
                #[cfg(test)]
                cell: *cell,
                vertices: h.vertices,
                normals: h.normals,
                of_edge: h.of_edge,
                inside: h.inside,
            }))
        }
    }
}

fn branch(
    field: &(dyn ScalarField + Sync),
    cell: &Cell,
    tol: &MeshTolerance,
    feature: f64,
    cancel: &crate::Cancel,
) -> Tree {
    let kids = cell.children();
    Tree::Branch(Box::new(std::array::from_fn(|c| {
        descend(field, &kids[c], tol, feature, cancel)
    })))
}


/// How deep this subtree goes.
fn max_depth(tree: &Tree, here: u32) -> u32 {
    match tree {
        Tree::Empty { .. } | Tree::Leaf(_) => here,
        Tree::Branch(kids) => kids
            .iter()
            .map(|k| max_depth(k, here + 1))
            .max()
            .unwrap_or(here),
    }
}

/// The subtree covering `p`, descended no further than `limit`.
fn subtree_at<'a>(tree: &'a Tree, cell: Cell, p: Point3<f64>, limit: u32) -> (&'a Tree, u32) {
    match tree {
        Tree::Branch(kids) if cell.depth < limit => {
            let mid = cell.size * 0.5;
            let c = usize::from(p.x >= cell.lo.x + mid)
                | usize::from(p.y >= cell.lo.y + mid) << 1
                | usize::from(p.z >= cell.lo.z + mid) << 2;
            subtree_at(&kids[c], cell.children()[c], p, limit)
        }
        _ => (tree, cell.depth),
    }
}

/// Leaves whose face-neighbour is more than one level finer than they are.
fn unbalanced(tree: &Tree, cell: Cell, root: &Tree, root_cell: Cell, out: &mut Vec<Cell>) {
    match tree {
        Tree::Branch(kids) => {
            let kid_cells = cell.children();
            for c in 0..8 {
                unbalanced(&kids[c], kid_cells[c], root, root_cell, out);
            }
        }
        // Empty regions are left alone. They carry no vertex and join nothing, so how finely a
        // neighbour is cut cannot make them wrong -- and refining them would cost the whole
        // saving, since most of a box is empty.
        Tree::Empty { .. } => {}
        Tree::Leaf(_) => {
            let half = cell.size * 0.5;
            let centre = Point3::from(cell.lo.coords.add_scalar(half));
            for axis in 0..3 {
                for dir in [-1.0f64, 1.0] {
                    let mut p = centre;
                    p[axis] += dir * cell.size;
                    // Outside the root has no neighbour to be out of step with.
                    if p[axis] < root_cell.lo[axis]
                        || p[axis] > root_cell.lo[axis] + root_cell.size
                    {
                        continue;
                    }
                    let (sub, at) = subtree_at(root, root_cell, p, cell.depth);
                    if at == cell.depth && max_depth(sub, at) > cell.depth + 1 {
                        out.push(cell);
                        return;
                    }
                }
            }
        }
    }
}

/// Replaces the leaf occupying `target` with a subdivision of it.
fn subdivide_at(
    tree: &mut Tree,
    cell: Cell,
    target: &Cell,
    field: &(dyn ScalarField + Sync),
    tol: &MeshTolerance,
    feature: f64,
    cancel: &crate::Cancel,
) {
    if cell.depth == target.depth {
        *tree = branch(field, &cell, tol, feature, cancel);
        return;
    }
    let Tree::Branch(kids) = tree else { return };
    let mid = cell.size * 0.5;
    let centre = Point3::from(target.lo.coords.add_scalar(target.size * 0.5));
    let c = usize::from(centre.x >= cell.lo.x + mid)
        | usize::from(centre.y >= cell.lo.y + mid) << 1
        | usize::from(centre.z >= cell.lo.z + mid) << 2;
    subdivide_at(&mut kids[c], cell.children()[c], target, field, tol, feature, cancel);
}

/// Subdivides until no leaf is more than one level coarser than a face-neighbour.
///
/// **What stops the fans.** A leaf stands in for each of its own children during the traversal,
/// so every edge along its face reaches for the same vertex. Where a coarse leaf borders a much
/// finer region that is dozens of edges converging on one point — visible as a spray of long
/// triangles radiating from it, spanning geometry they should not.
///
/// Balancing bounds it: a face can then be met by at most four finer cells, so a vertex is
/// shared by a handful of quads rather than by however many the neighbour happened to cut. This
/// is the standard restriction an octree mesher carries, and it is not an optimisation — an
/// unbalanced tree produces a mesh that is wrong to look at.
///
/// Repeated to a fixed point, because subdividing one leaf can put it out of step with another.
/// The bound is a safety net rather than an expected limit: each round strictly refines, and the
/// floor already stops that.
fn balance(
    tree: &mut Tree,
    root_cell: Cell,
    field: &(dyn ScalarField + Sync),
    tol: &MeshTolerance,
    feature: f64,
    cancel: &crate::Cancel,
) {
    for _ in 0..32 {
        if cancel.is_cancelled() {
            return;
        }
        let mut todo = Vec::new();
        unbalanced(tree, root_cell, tree, root_cell, &mut todo);
        if todo.is_empty() {
            return;
        }
        for cell in todo {
            // A cell already at the floor cannot be refined, and asking would loop forever.
            if cell.size * 0.5 < tol.min_cell_mm {
                continue;
            }
            subdivide_at(tree, root_cell, &cell, field, tol, feature, cancel);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Dual contouring over the tree
//
// The recursion is the one from Ju et al.: a cell's own children are visited, then the faces
// between them, then the edges between those. A quad is emitted only at a **minimal edge** — one
// no smaller cell subdivides — which is what makes neighbouring cells at different depths join
// up without cracks, and is why the whole traversal exists rather than a loop over leaves.
//
// The tables below are derived rather than transcribed, from one convention: a child index has
// bit 0 for +x, bit 1 for +y and bit 2 for +z, the same as [`Cell::corner`]. Every table is a
// consequence of that, and each is stated with the reasoning so it can be rechecked.
// ---------------------------------------------------------------------------------------------

/// The four pairs of children adjacent across the internal face perpendicular to each axis.
///
/// Two children share such a face when their indices differ in exactly that axis' bit, so the
/// minus-side child is the one with the bit clear.
fn face_pairs(axis: usize) -> [(usize, usize); 4] {
    let bit = 1 << axis;
    let mut out = [(0, 0); 4];
    let mut n = 0;
    for c in 0..8 {
        if c & bit == 0 {
            out[n] = (c, c | bit);
            n += 1;
        }
    }
    out
}

/// The two axes perpendicular to `axis`, in the order that makes a right-handed ring.
fn perpendicular(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    }
}

/// The four children around each internal edge running along `axis`, in cyclic order.
///
/// An edge along `axis` is surrounded by four cells differing in the two perpendicular
/// directions. Taking them as (−,−), (+,−), (+,+), (−,+) in that plane gives a consistent
/// rotation, which is what lets the emitted quad be wound from the sign alone.
fn edge_groups(axis: usize) -> [[usize; 4]; 2] {
    let (u, v) = perpendicular(axis);
    let (bu, bv) = (1 << u, 1 << v);
    let bit = 1 << axis;
    std::array::from_fn(|half| {
        let base = if half == 1 { bit } else { 0 };
        [base, base | bu, base | bu | bv, base | bv]
    })
}

/// Which child of the cell at ring position `i` touches an edge along `axis`.
///
/// A cell at (−,−) relative to the edge meets it at its (+u,+v) corner, so its children on that
/// edge are the ones with both perpendicular bits set. Going round the ring flips them in turn.
fn ring_child(i: usize, axis: usize, half: usize) -> usize {
    let (u, v) = perpendicular(axis);
    let (want_u, want_v) = match i {
        0 => (1, 1),
        1 => (0, 1),
        2 => (0, 0),
        _ => (1, 0),
    };
    (half << axis) | (want_u << u) | (want_v << v)
}

/// One node of the traversal, resolved: a subtree together with the cell it occupies.
struct Node<'a> {
    tree: &'a Tree,
    cell: Cell,
}

impl<'a> Node<'a> {
    /// A child, or this node again when it is a leaf.
    ///
    /// **The step that makes mixed depths work.** A leaf standing in for its own children lets
    /// the recursion keep descending on the neighbours that *are* subdivided, until the edge is
    /// minimal, while the coarse side keeps offering the same vertex. That is exactly the
    /// behaviour a crack-free join needs, and it is why nothing here has to reason about
    /// levels.
    fn child(&self, c: usize) -> Node<'a> {
        match self.tree {
            Tree::Branch(kids) => Node {
                tree: &kids[c],
                cell: self.cell.children()[c],
            },
            _ => Node {
                tree: self.tree,
                cell: self.cell,
            },
        }
    }

    fn is_branch(&self) -> bool {
        self.tree.is_branch()
    }

    /// The dual vertex, when this region has one.
    fn vertex(&self) -> Option<&Leaf> {
        match self.tree {
            Tree::Leaf(l) => Some(l),
            _ => None,
        }
    }

    /// Whether the point at corner `c` of this node's cell is inside.
    fn sign_at_corner(&self, c: usize) -> bool {
        match self.tree {
            Tree::Empty { inside } => *inside,
            Tree::Leaf(l) => l.inside[c],
            // Only reached through `child`, which never returns a branch as a leaf stand-in.
            Tree::Branch(_) => false,
        }
    }
}

/// What the traversal builds.
struct Contour {
    positions: Vec<Point3<f64>>,
    normals: Vec<Vector3<f64>>,
    triangles: Vec<[u32; 3]>,
    /// One index per leaf, assigned on first use, so a leaf's vertex is written once however
    /// many edges reach it.
    index_of: std::collections::HashMap<(usize, usize), u32>,
}

impl Contour {
    fn index(&mut self, leaf: &Leaf, sheet: usize) -> u32 {
        let sheet = sheet.min(leaf.vertices.len().saturating_sub(1));
        // Keyed by the leaf's address, which is stable for the life of the tree and unique per
        // leaf — cheaper than hashing a position, and exact where a position could collide.
        let key = (leaf as *const Leaf as usize, sheet);
        if let Some(i) = self.index_of.get(&key) {
            return *i;
        }
        let i = self.positions.len() as u32;
        self.positions.push(leaf.vertices[sheet]);
        self.normals.push(leaf.normals[sheet]);
        self.index_of.insert(key, i);
        i
    }
}

fn cell_proc(n: &Node<'_>, out: &mut Contour) {
    if !n.is_branch() {
        return;
    }
    for c in 0..8 {
        cell_proc(&n.child(c), out);
    }
    for axis in 0..3 {
        for (a, b) in face_pairs(axis) {
            face_proc(&n.child(a), &n.child(b), axis, out);
        }
    }
    for axis in 0..3 {
        for group in edge_groups(axis) {
            let q = [
                n.child(group[0]),
                n.child(group[1]),
                n.child(group[2]),
                n.child(group[3]),
            ];
            edge_proc(&q, axis, out);
        }
    }
}

/// The face between two neighbours, perpendicular to `axis`, with `lo` on the minus side.
fn face_proc(lo: &Node<'_>, hi: &Node<'_>, axis: usize, out: &mut Contour) {
    if !lo.is_branch() && !hi.is_branch() {
        return;
    }
    let bit = 1 << axis;
    let (u, v) = perpendicular(axis);
    let (bu, bv) = (1 << u, 1 << v);

    // The four quadrants of the shared face. Each pairs a child of `lo` on its plus side with
    // the child of `hi` directly opposite.
    for &(su, sv) in &[(0, 0), (1, 0), (1, 1), (0, 1)] {
        let offset = (su * bu) | (sv * bv);
        face_proc(
            &lo.child(bit | offset),
            &hi.child(offset),
            axis,
            out,
        );
    }

    // The four edges lying in that face: two along each perpendicular direction, one for each
    // half of the face along the edge's own direction.
    //
    // Two rather than one per direction, because a child of `lo` spans only half the face: an
    // edge running the full width would not be contained by any single one of them, and the
    // four cells handed to `edge_proc` must each contain the whole edge.
    for (dir, other) in [(u, v), (v, u)] {
        for half in 0..2 {
            let q = ring_around_face(lo, hi, axis, dir, other, half);
            edge_proc(&q, dir, out);
        }
    }
}

/// The four cells around an edge lying in the face between `lo` and `hi`.
///
/// The edge runs along `dir`, sits at the middle of the face in `other`, and takes the `half`
/// portion of the face along `dir`. Its four neighbours therefore differ in `axis` — which is
/// what decides whether each comes from `lo` or from `hi` — and in `other`.
fn ring_around_face<'a>(
    lo: &Node<'a>,
    hi: &Node<'a>,
    axis: usize,
    dir: usize,
    other: usize,
    half: usize,
) -> [Node<'a>; 4] {
    // Only the first of the pair is compared against, since knowing which of `axis` and
    // `other` is `u` settles the other by elimination.
    let (u, _) = perpendicular(dir);
    std::array::from_fn(|i| {
        // Where this ring position sits relative to the edge, in the same convention
        // `ring_child` uses: a 1 means the cell is on the minus side of that direction, so the
        // child of it that touches the edge is the one with that bit set.
        let (want_u, want_v) = match i {
            0 => (1, 1),
            1 => (0, 1),
            2 => (0, 0),
            _ => (1, 0),
        };
        let want_axis = if u == axis { want_u } else { want_v };
        let want_other = if u == other { want_u } else { want_v };

        // On the minus side of `axis` is `lo`, and the child of it that reaches the shared face
        // is the one on its own plus side.
        let parent = if want_axis == 1 { lo } else { hi };
        let bit_axis = want_axis;
        // In `other` the edge is at the parent's midplane, so a cell on the minus side of the
        // edge is the child with that bit clear — the opposite sense to `axis`.
        let bit_other = 1 - want_other;
        parent.child((half << dir) | (bit_axis << axis) | (bit_other << other))
    })
}

/// Four cells around an edge running along `axis`, in cyclic order.
fn edge_proc(q: &[Node<'_>; 4], axis: usize, out: &mut Contour) {
    if q.iter().any(|n| n.is_branch()) {
        for half in 0..2 {
            let sub = std::array::from_fn(|i| q[i].child(ring_child(i, axis, half)));
            edge_proc(&sub, axis, out);
        }
        return;
    }
    emit(q, axis, out);
}

/// A minimal edge: every surrounding cell is a leaf, so this is where a quad belongs.
fn emit(q: &[Node<'_>; 4], axis: usize, out: &mut Contour) {
    // The finest of the four owns the edge. A coarser neighbour does not have these corners, so
    // its own signs would answer a different question.
    let Some(finest) = (0..4).max_by_key(|i| q[*i].cell.depth) else {
        return;
    };
    let (u, v) = perpendicular(axis);
    let (want_u, want_v) = match finest {
        0 => (1, 1),
        1 => (0, 1),
        2 => (0, 0),
        _ => (1, 0),
    };
    let base = (want_u << u) | (want_v << v);
    let (c0, c1) = (base, base | (1 << axis));
    let (s0, s1) = (
        q[finest].sign_at_corner(c0),
        q[finest].sign_at_corner(c1),
    );
    if s0 == s1 {
        return;
    }

    // Every corner of the quad must exist, and each cell offers the vertex for *its own sheet*.
    //
    // **This is the whole of the manifold fix.** A cell holding two sheets used to have one
    // vertex, so both sheets joined through it and the edges around it were shared by more
    // triangles than a surface permits. Each ring cell is now asked which sheet the edge in
    // question belongs to — by the edge at the same corner of its own twelve, which is the edge
    // containing this one whether the cell is the finest or coarser than it.
    let mut idx = [0u32; 4];
    for i in 0..4 {
        let Some(leaf) = q[i].vertex() else { return };
        let (wu, wv) = match i {
            0 => (1, 1),
            1 => (0, 1),
            2 => (0, 0),
            _ => (1, 0),
        };
        let sheet = leaf.of_edge[edge_at_corner(axis, wu, wv)];
        // A cell whose own corners show no crossing on that edge has one vertex and no sheets
        // to choose between — the guessed vertex a thin feature gets. Zero is that one.
        idx[i] = out.index(leaf, if sheet == NO_SHEET { 0 } else { sheet as usize });
    }

    // Wound so the face points from inside to outside. The ring is right-handed about +axis, so
    // it faces along +axis when the far corner is the outside one.
    let ring = if s1 { [idx[0], idx[3], idx[2], idx[1]] } else { idx };

    // **A ring can name the same cell twice.** Where a coarse leaf meets finer neighbours, two
    // ring positions resolve to that one leaf — `Node::child` hands a leaf back for every child
    // asked of it — so the quad is really a triangle with one corner written twice. Emitting it
    // as a quad puts a zero-area half in the mesh and, worse, leaves the shared diagonal used
    // once: a tear exactly along the level change, which is where a tree is most likely to be
    // wrong and hardest to see.
    //
    // Collapsed cyclically, so a repeat anywhere in the ring is caught rather than only an
    // adjacent one.
    let mut ring: Vec<u32> = ring.to_vec();
    ring.dedup();
    if ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }

    match ring.len() {
        // A quad. Which diagonal is cut decides whether its two halves face the same way, so it
        // is decided by the field's own normals rather than by shape -- the same correction the
        // uniform mesher needed, where scoring the splits took inward-wound triangles from 4350
        // to 2105. The shorter diagonal breaks a tie, which is what keeps a long thin quad
        // joining cells of different sizes from folding.
        4 => {
            let agrees = |t: [u32; 3]| {
                let (pa, pb, pc) = (
                    out.positions[t[0] as usize],
                    out.positions[t[1] as usize],
                    out.positions[t[2] as usize],
                );
                let facing = out.normals[t[0] as usize]
                    + out.normals[t[1] as usize]
                    + out.normals[t[2] as usize];
                ((pb - pa).cross(&(pc - pa)).dot(&facing) > 0.0) as usize
            };
            let split_02 = [[ring[0], ring[1], ring[2]], [ring[0], ring[2], ring[3]]];
            let split_13 = [[ring[0], ring[1], ring[3]], [ring[1], ring[2], ring[3]]];
            let s02 = agrees(split_02[0]) + agrees(split_02[1]);
            let s13 = agrees(split_13[0]) + agrees(split_13[1]);

            let p = |i: u32| out.positions[i as usize];
            let chosen = if s02 != s13 {
                if s02 > s13 { split_02 } else { split_13 }
            } else if (p(ring[2]) - p(ring[0])).norm_squared()
                <= (p(ring[3]) - p(ring[1])).norm_squared()
            {
                split_02
            } else {
                split_13
            };
            out.triangles.extend_from_slice(&chosen);
        }
        3 => out.triangles.push([ring[0], ring[1], ring[2]]),
        // Two or fewer distinct cells around a sign-changing edge describes no surface at all.
        _ => {}
    }
}

/// Meshes the zero level set of `field` inside `bounds`, adaptively.
///
/// The counterpart to [`crate::mesh_field`]: same contract, same output, but the cells are
/// chosen rather than fixed. Returns an empty mesh when the surface does not cross the box,
/// which is a legitimate answer and not an error.
pub fn mesh_field_adaptive(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tol: &MeshTolerance,
) -> FieldMesh {
    mesh_field_adaptive_cancellable(field, bounds, tol, &crate::Cancel::default())
        .expect("a token nobody holds is never set")
}

/// [`mesh_field_adaptive`], or `None` if `cancel` was set before the mesh was finished.
pub(crate) fn mesh_field_adaptive_cancellable(
    field: &(dyn ScalarField + Sync),
    bounds: (Point3<f64>, Point3<f64>),
    tol: &MeshTolerance,
    cancel: &crate::Cancel,
) -> Option<FieldMesh> {
    let (tree, root) = build_cancellable(field, bounds, tol, cancel);
    // Asked after the build rather than trusted to it: a build that was cancelled returns a tree
    // with empty cells where it stopped, and contouring that would produce a mesh with holes
    // that looks like an answer.
    if cancel.is_cancelled() {
        return None;
    }
    let mut out = Contour {
        positions: Vec::new(),
        normals: Vec::new(),
        triangles: Vec::new(),
        index_of: std::collections::HashMap::new(),
    };
    cell_proc(&Node { tree: &tree, cell: root }, &mut out);
    Some(FieldMesh {
        positions: out.positions,
        normals: out.normals,
        triangles: out.triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wall of `thickness` millimetres across Z, centred at `at`, which declares how thin it is
    /// unless `silent`.
    struct Wall {
        at: f64,
        thickness: f64,
        silent: bool,
    }

    impl ScalarField for Wall {
        fn distance(&self, p: Point3<f64>) -> f64 {
            (p.z - self.at).abs() - self.thickness * 0.5
        }

        fn feature_size(&self) -> Option<f64> {
            if self.silent { None } else { Some(self.thickness) }
        }
    }

    /// **A wall thinner than a cell is found when the field says how thin it is, and missed when
    /// it does not.** The mesher reads the field at corners, so a 1 mm wall lying between two of
    /// them leaves every corner outside it and every edge uncrossed — here the wall sits at
    /// z = 4 in a 64 mm box, where the forced levels put corners at multiples of 8. That is how
    /// a 2 mm sheet lattice came out as nothing at all from a box a few millimetres taller than
    /// one that meshed to 130 000 triangles.
    ///
    /// The silent half is not a wish for that behaviour; it is what every field that declares
    /// nothing still gets, and it is why the declaration has to come from the field.
    #[test]
    fn a_wall_thinner_than_a_cell_is_meshed_when_the_field_declares_it() {
        let tol = MeshTolerance {
            chord_mm: 0.05,
            angle_deg: 10.0,
            max_edge_mm: 0.0,
            min_cell_mm: 0.25,
        };
        let box_of = (Point3::new(0.0, 0.0, 0.0), Point3::new(64.0, 64.0, 64.0));
        let spoken =
            mesh_field_adaptive(&Wall { at: 4.0, thickness: 1.0, silent: false }, box_of, &tol);
        assert!(
            spoken.triangles.len() > 100,
            "the wall should be meshed, got {} triangles",
            spoken.triangles.len()
        );
        // Both faces of it, where they actually are.
        let (lo, hi) = spoken
            .positions
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.z), hi.max(p.z))
            });
        assert!(
            (lo - 3.5).abs() < 0.3 && (hi - 4.5).abs() < 0.3,
            "the meshed wall spans {lo:.2} to {hi:.2}, not 3.5 to 4.5"
        );

        let silent =
            mesh_field_adaptive(&Wall { at: 4.0, thickness: 1.0, silent: true }, box_of, &tol);
        assert!(
            silent.triangles.is_empty(),
            "without the declaration the corners never straddle the wall, so nothing is found: {} triangles says this test no longer measures what it was written for",
            silent.triangles.len()
        );
    }

    struct Flat;
    impl ScalarField for Flat {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.z
        }
        fn normal(&self, _p: Point3<f64>) -> Option<Vector3<f64>> {
            Some(Vector3::z())
        }
    }

    /// Two half-spaces at a right angle: a crease along the whole of Y, and no sag anywhere.
    struct Crease;
    impl ScalarField for Crease {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.x.max(p.z)
        }
        fn normalization(&self) -> crate::Normalization {
            crate::Normalization::LowerBound
        }
    }

    struct Sphere {
        radius: f64,
    }
    impl ScalarField for Sphere {
        fn distance(&self, p: Point3<f64>) -> f64 {
            p.coords.norm() - self.radius
        }
        fn normal(&self, p: Point3<f64>) -> Option<Vector3<f64>> {
            p.coords.try_normalize(1e-12)
        }
    }

    fn box_of(half: f64) -> (Point3<f64>, Point3<f64>) {
        (
            Point3::new(-half, -half, -half),
            Point3::new(half, half, half),
        )
    }

    fn edge_uses(mesh: &FieldMesh) -> std::collections::HashMap<(u32, u32), usize> {
        let mut uses = std::collections::HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        uses
    }

    /// A flat surface stops at the forced depth, because nothing about it asks for more.
    #[test]
    fn a_flat_surface_costs_only_the_cells_it_is_forced_to_take() {
        let tol = MeshTolerance {
            chord_mm: 0.01,
            angle_deg: 15.0,
            max_edge_mm: 0.0,
            min_cell_mm: 0.001,
        };
        let (tree, _) = build(&Flat, box_of(10.0), &tol);
        let mut depths = Vec::new();
        tree.leaf_depths(&mut depths);
        assert!(!depths.is_empty(), "the plane should produce leaves");
        assert!(
            depths.iter().all(|d| *d == FORCED_DEPTH),
            "a plane should stop at the forced depth, deepest was {:?}",
            depths.iter().max()
        );
    }

    /// A crease is refined although it has no sag, which is why the angular test is its own test.
    #[test]
    fn a_crease_is_refined_although_it_has_no_sag() {
        let tol = MeshTolerance {
            chord_mm: 1.0,
            angle_deg: 15.0,
            max_edge_mm: 0.0,
            min_cell_mm: 0.2,
        };
        let (tree, root) = build(&Crease, box_of(10.0), &tol);
        let mut depths = Vec::new();
        tree.leaf_depths(&mut depths);
        let deepest = *depths.iter().max().expect("leaves");
        assert!(
            deepest > FORCED_DEPTH,
            "the crease should drive refinement past the forced depth, stopped at {deepest}"
        );
        let finest = root.size / (1u32 << deepest) as f64;
        assert!(
            finest >= tol.min_cell_mm * 0.5,
            "refined past the floor: {finest} mm against a {} mm floor",
            tol.min_cell_mm
        );
    }

    /// Tightening the angular tolerance buys more cells, and loosening it buys fewer.
    #[test]
    fn a_tighter_angle_costs_more_cells_and_a_looser_one_costs_fewer() {
        let at = |angle_deg: f64| {
            let tol = MeshTolerance {
                chord_mm: 1.0,
                angle_deg,
                max_edge_mm: 0.0,
                min_cell_mm: 0.05,
            };
            build(&Crease, box_of(10.0), &tol).0.surface_leaves()
        };
        let (loose, tight) = (at(60.0), at(5.0));
        assert!(
            tight > loose,
            "5 degrees gave {tight} leaves and 60 degrees gave {loose}"
        );
    }

    /// A capped edge length subdivides a plane that both other tests are happy with.
    #[test]
    fn a_maximum_edge_length_subdivides_a_surface_nothing_else_would() {
        let base = MeshTolerance {
            chord_mm: 1.0,
            angle_deg: 45.0,
            max_edge_mm: 0.0,
            min_cell_mm: 0.05,
        };
        let uncapped = build(&Flat, box_of(10.0), &base).0.surface_leaves();
        let capped = build(
            &Flat,
            box_of(10.0),
            &MeshTolerance {
                max_edge_mm: 0.5,
                ..base
            },
        )
        .0
        .surface_leaves();
        assert!(
            capped > uncapped,
            "capping the edge should cost cells: {capped} against {uncapped}"
        );
    }

    /// A cell holding two sheets reports two, and one holding one reports one.
    ///
    /// The partition the manifold fix rests on. A single vertex per cell is sound while the
    /// surface inside it is one sheet; where two pass through, one vertex makes every edge
    /// around it join both, and an export reports edges belonging to more triangles than a
    /// surface allows. Splitting by the connectivity of the inside corners is what separates
    /// them — and it is the *same* connectivity `is_manifold` tests, used to divide rather than
    /// to refuse.
    #[test]
    fn a_cell_with_two_sheets_gives_each_its_own_vertex() {
        // One corner in: one sheet, and the three edges leaving that corner belong to it.
        let mut one = [false; 8];
        one[0] = true;
        let (of_edge, sheets) = sheets_of(&one);
        assert_eq!(sheets, 1);
        assert_eq!(of_edge.iter().filter(|s| **s != NO_SHEET).count(), 3);

        // Two corners sharing a face: still one sheet, now with four crossing edges.
        let mut face = [false; 8];
        face[0] = true;
        face[1] = true;
        let (of_edge, sheets) = sheets_of(&face);
        assert_eq!(sheets, 1);
        assert!(of_edge.iter().all(|s| *s == NO_SHEET || *s == 0));

        // Opposite corners, touching only at the cube's diagonal: two sheets, and the edges
        // divide between them rather than all reaching one vertex.
        let mut diagonal = [false; 8];
        diagonal[0] = true;
        diagonal[7] = true;
        let (of_edge, sheets) = sheets_of(&diagonal);
        assert_eq!(sheets, 2, "corners meeting at a diagonal are two sheets");
        assert_eq!(of_edge.iter().filter(|s| **s == 0).count(), 3);
        assert_eq!(of_edge.iter().filter(|s| **s == 1).count(), 3);

        // And nothing at all: no sheets, so `hermite` has nothing to solve.
        assert_eq!(sheets_of(&[false; 8]).1, 0);
        assert_eq!(sheets_of(&[true; 8]).1, 1);
    }

    /// Which of the twelve edges a corner names, checked against the table's own order.
    #[test]
    fn an_edge_at_a_corner_is_the_edge_that_corner_names() {
        for axis in 0..3 {
            let (u, v) = perpendicular(axis);
            for want_u in 0..2 {
                for want_v in 0..2 {
                    let e = edge_at_corner(axis, want_u, want_v);
                    let (a, b) = CELL_EDGES[e];
                    // It runs along `axis`: its ends differ in that bit and no other.
                    assert_eq!(a ^ b, 1 << axis, "edge {e} does not run along {axis}");
                    // And sits at the corner asked for, in the perpendicular directions.
                    assert_eq!((a >> u) & 1, want_u, "edge {e} is not at u={want_u}");
                    assert_eq!((a >> v) & 1, want_v, "edge {e} is not at v={want_v}");
                }
            }
        }
    }

    /// The ambiguous sign patterns are the ones with a diagonal, and only those.
    #[test]
    fn a_cell_is_manifold_exactly_when_each_side_is_face_connected() {
        // Nothing, everything, and one corner: all trivially one sheet.
        assert!(is_manifold([false; 8]));
        assert!(is_manifold([true; 8]));
        let mut one = [false; 8];
        one[0] = true;
        assert!(is_manifold(one));

        // Two corners sharing a face: still one sheet.
        let mut face = [false; 8];
        face[0] = true;
        face[1] = true;
        assert!(is_manifold(face));

        // Two corners meeting only at a cube diagonal: the ambiguous case, and the whole reason
        // this test exists. One vertex cannot stand for both sheets.
        let mut diagonal = [false; 8];
        diagonal[0] = true;
        diagonal[7] = true;
        assert!(!is_manifold(diagonal));

        // Two corners meeting only along a face diagonal: ambiguous for the same reason.
        let mut face_diagonal = [false; 8];
        face_diagonal[0] = true;
        face_diagonal[3] = true;
        assert!(!is_manifold(face_diagonal));
    }

    /// A sphere meshes to a closed surface: every edge belongs to exactly two triangles.
    ///
    /// The property the whole traversal exists for. Cells of different sizes meet all over a
    /// sphere — it is curved everywhere, so the tree is not uniform — and a join that did not
    /// account for that would leave cracks along every level change.
    #[test]
    fn an_adaptive_sphere_is_closed_and_manifold() {
        let mesh = mesh_field_adaptive(
            &Sphere { radius: 10.0 },
            box_of(16.0),
            &MeshTolerance {
                chord_mm: 0.05,
                angle_deg: 20.0,
                max_edge_mm: 0.0,
                min_cell_mm: 0.05,
            },
        );
        assert!(!mesh.triangles.is_empty(), "the sphere should mesh");

        let uses = edge_uses(&mesh);
        let torn = uses.values().filter(|n| **n == 1).count();
        let over = uses.values().filter(|n| **n > 2).count();
        assert_eq!(torn, 0, "{torn} edges belong to only one triangle");
        assert_eq!(over, 0, "{over} edges belong to more than two triangles");
    }

    /// And the vertices sit on the surface, so it is the right closed surface.
    #[test]
    fn an_adaptive_spheres_vertices_lie_on_the_sphere() {
        let radius = 10.0;
        let tol = MeshTolerance {
            chord_mm: 0.05,
            angle_deg: 20.0,
            max_edge_mm: 0.0,
            min_cell_mm: 0.05,
        };
        let mesh = mesh_field_adaptive(&Sphere { radius }, box_of(16.0), &tol);
        let worst = mesh
            .positions
            .iter()
            .map(|p| (p.coords.norm() - radius).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 0.2,
            "worst vertex sits {worst:.4} mm off a {radius} mm sphere"
        );
    }
}
