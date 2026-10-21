// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
// A lattice that follows a surface: the surface's buffers, the walk that asks where a point sits
// on it, and the lattice arithmetic over that answer.
//
// **Its own file, appended only to programs that use it, and that is a measured decision.** It
// lived in the prelude, reached from `lattice_distance` through `if map > 2.5` — a branch on a
// buffer value, which no shader compiler can resolve statically. So every lattice shader in every
// document carried a 64-deep traversal loop whether anything followed a surface or not, and the
// owner reported opening a lattice file hanging the application. Timed cold by
// `fieldform-gpu/examples/lattice_pipeline_cost.rs`, a plain rectangular gyroid in a cube:
//
// ```text
//                                      cold pipeline
//                                      plain gyroid   gyroid + 90 spheres
//   walk in the prelude, every lattice    2340.6 ms       5294.0 ms
//   walk appended only where used         1080.6 ms       3307.2 ms
// ```
//
// Measured with the same salt both sides (`FIELDFORM_LATTICE_SALT=1`), since the sources differ
// between the two builds and both are therefore cold.
//
// Codegen knows at generation time which lattices are conformal — the map is structure, and a
// change of tiling already recompiles — so it calls these functions by name for those and appends
// this file, and every other program is built without a line of it.

// Every conformal surface's floats and integers, each surface's run laid end to end.
//
// **Two buffers rather than six.** A binding is not free, and a layout that has to be agreed in
// two places is one that can be agreed differently — so the offsets travel with the data as a
// `SurfaceLayout` rather than being implied by which buffer something landed in.
// `PackedSurface` in `fieldform-field` states the layout; this reads it, and
// `a_packed_surface_answers_as_the_surface_does` is what holds the two together.
@group(0) @binding(9) var<storage, read> surface_floats: array<f32>;
@group(0) @binding(10) var<storage, read> surface_uints: array<u32>;

// Where one surface's runs begin, in elements of their own arrays.
struct SurfaceLayout {
    positions: u32,
    normals: u32,
    uvs: u32,
    gradients: u32,
    node_bounds: u32,
    triangles: u32,
    faces: u32,
    node_links: u32,
    order: u32,
    triangle_count: u32,
    // The fastest `s` or `t` varies anywhere on this surface, per millimetre. A bound rather
    // than a fact about a point: a ray has to know how fast the pattern can possibly change
    // before it steps, and the finest face may be the one it is about to arrive at.
    worst_gradient: f32,
}
@group(0) @binding(11) var<storage, read> surface_layouts: array<SurfaceLayout>;

// Where a point sits on a surface. `PackedHit` in `fieldform-field/src/surface_map.rs` is the
// same six answers, and `a_packed_surface_answers_as_the_surface_does` holds them together.
//
// No face id: the CPU's `SurfacePoint` carries one because callers ask which face they landed
// on, and a lattice never does -- its three arguments are `(s, t, offset)` and nothing else.
struct SurfaceHit {
    st: vec2<f32>,
    offset: f32,
    // The displacement along the surface's normal: `offset` over the surface, and continuous
    // past an open surface's rim where `offset` flips sign. See `SurfacePoint::along`.
    along: f32,
    // Unit vector from the closest point to the query: the band's own outward direction, which
    // is the normal over the surface and radial around an open surface's rim.
    outward: vec3<f32>,
    ds: vec3<f32>,
    dt: vec3<f32>,
    normal: vec3<f32>,
}

// Where a point sits on a conformal surface: `(s, t, offset, face)`.
//
// **A transcription, and deliberately a plain one.** `query_packed` in
// `fieldform-field/src/surface_map.rs` is the same walk in Rust over the same two arrays, and it
// is checked against `SurfaceMap::at` — which is itself checked against a panel whose answers can
// be worked out by hand. So the chain is: hand-checkable panel, then the real query, then the
// packed one, then this. A fault here is a fault in the transcription, which is a much smaller
// thing to look for than a fault somewhere in an algorithm and a layout together.
//
// The closest point on the surface decides everything: `s` and `t` come out of the triangle it
// landed on, barycentrically, and the offset is signed by which side of that triangle the point
// is on. An open surface has no inside, so side is the only sign there is.
// **A ray-coherent seed was built, measured and taken out** (2026-09-20, `conformal_cost.rs`).
// Seeding the search with the previous sample's triangle as an upper bound changed no answer and
// saved nothing: 73.7 ms against 72.3 at levels 3, 133 against 135 at levels 4. The walk already
// reaches its nearest leaf first and has its bound from there, so the seed spares it a dozen box
// tests. What is left is inherent to an exact closest point: a sample far from the surface has
// to visit every leaf nearer than it is. The lever that would pay is a bound that stops
// descending into a box small for its distance -- see `conformal-cell-map-spec.md` section 9.
fn surface_at(index: u32, p: vec3<f32>) -> SurfaceHit {
    let l = surface_layouts[index];
    var miss: SurfaceHit;
    miss.normal = vec3<f32>(0.0, 0.0, 1.0);
    if l.triangle_count == 0u {
        return miss;
    }

    var best = 1e30;
    var best_tri = 0xffffffffu;
    var best_point = vec3<f32>(0.0);
    var best_bary = vec3<f32>(0.0);

    // An explicit stack, fixed. A shader has no growable one, and a hierarchy over any mesh
    // this side of absurd is nowhere near this deep.
    var stack: array<u32, 64>;
    stack[0] = 0u;
    var depth = 1u;

    while (depth > 0u) {
        depth = depth - 1u;
        let node = stack[depth];
        if (surface_box_distance(l, node, p) >= best) {
            continue;
        }
        let links = l.node_links + node * 3u;
        let first = surface_uints[links];
        let count = surface_uints[links + 1u];
        let right = surface_uints[links + 2u];

        if (count > 0u) {
            for (var k = 0u; k < count; k = k + 1u) {
                let tri = surface_uints[l.order + first + k];
                let a = surface_vertex(l, tri, 0u);
                let b = surface_vertex(l, tri, 1u);
                let c = surface_vertex(l, tri, 2u);
                let hit = closest_on_triangle(p, a, b, c);
                let d = dot(hit.xyz - p, hit.xyz - p);
                // The same tie-break the field uses: by face, then by triangle. A point on a
                // seam is equidistant from both sides of it, and the order a walk reaches them
                // in is an accident of how the tree was built.
                var nearer = d < best;
                if (d == best && best_tri != 0xffffffffu) {
                    let mine = surface_uints[l.faces + tri];
                    let theirs = surface_uints[l.faces + best_tri];
                    nearer = mine < theirs || (mine == theirs && tri < best_tri);
                } else if (d == best) {
                    nearer = true;
                }
                if (nearer) {
                    best = d;
                    best_tri = tri;
                    best_point = hit.xyz;
                    best_bary = surface_barycentric(p, a, b, c, hit.xyz);
                }
            }
            continue;
        }

        // The nearer child is pushed last so it is popped first, which tightens `best` early
        // and is what makes the test above actually prune.
        let dl = surface_box_distance(l, first, p);
        let dr = surface_box_distance(l, right, p);
        var near = first;
        var far = right;
        if (dr < dl) {
            near = right;
            far = first;
        }
        if (depth + 2u <= 64u) {
            stack[depth] = far;
            stack[depth + 1u] = near;
            depth = depth + 2u;
        }
    }

    if (best_tri == 0xffffffffu) {
        return miss;
    }

    let uv0 = surface_uv(l, best_tri, 0u);
    let uv1 = surface_uv(l, best_tri, 1u);
    let uv2 = surface_uv(l, best_tri, 2u);
    let st = uv0 * best_bary.x + uv1 * best_bary.y + uv2 * best_bary.z;

    let a = surface_vertex(l, best_tri, 0u);
    let b = surface_vertex(l, best_tri, 1u);
    let c = surface_vertex(l, best_tri, 2u);
    // **The surface's own normal, interpolated, and not the triangle's.** A facet normal stood
    // here first and was wrong twice over: it signs the offset differently from the field
    // wherever the surface curves, and a conformal lattice carries this direction straight into
    // its gradient -- so the shader would draw a normal the field does not have. Measured 4.3
    // degrees apart on a dome.
    let n = normalize(
        surface_normal(l, best_tri, 0u) * best_bary.x
            + surface_normal(l, best_tri, 1u) * best_bary.y
            + surface_normal(l, best_tri, 2u) * best_bary.z
    );
    let away = p - best_point;
    let along = dot(away, n);
    var offset = length(away);
    if (along < 0.0) {
        offset = -offset;
    }

    // **Interpolated from the corners, not derived from the triangle.** How fast the surface's
    // own coordinates change per millimetre is what sizes a conformal lattice's walls, and a
    // triangle's own gradient is constant on it and jumps at every edge -- which a convex
    // surface puts a wedge of real volume nearest. Carried per corner by the tessellation for
    // exactly this reason; `SurfaceMap::at` blends the same three the same way.
    let g0 = surface_gradients(l, best_tri, 0u);
    let g1 = surface_gradients(l, best_tri, 1u);
    let g2 = surface_gradients(l, best_tri, 2u);
    var hit: SurfaceHit;
    hit.st = clamp(st, vec2<f32>(0.0), vec2<f32>(1.0));
    hit.offset = offset;
    hit.along = along;
    hit.normal = n;
    let reach = length(away);
    if (reach > 1e-6) {
        hit.outward = away / reach;
    } else {
        hit.outward = n;
    }
    hit.ds = g0.ds * best_bary.x + g1.ds * best_bary.y + g2.ds * best_bary.z;
    hit.dt = g0.dt * best_bary.x + g1.dt * best_bary.y + g2.dt * best_bary.z;
    return hit;
}

// `(grad s, grad t)` at one corner of one triangle: six floats, `grad s` first.
struct SurfaceGradients {
    ds: vec3<f32>,
    dt: vec3<f32>,
}

fn surface_gradients(l: SurfaceLayout, tri: u32, k: u32) -> SurfaceGradients {
    let base = l.gradients + tri * 18u + k * 6u;
    var g: SurfaceGradients;
    g.ds = vec3<f32>(
        surface_floats[base],
        surface_floats[base + 1u],
        surface_floats[base + 2u],
    );
    g.dt = vec3<f32>(
        surface_floats[base + 3u],
        surface_floats[base + 4u],
        surface_floats[base + 5u],
    );
    return g;
}

fn surface_normal(l: SurfaceLayout, tri: u32, k: u32) -> vec3<f32> {
    let v = surface_uints[l.triangles + tri * 3u + k];
    let base = l.normals + v * 3u;
    return vec3<f32>(
        surface_floats[base],
        surface_floats[base + 1u],
        surface_floats[base + 2u],
    );
}

// **Built, measured, and taken out again**, recorded here so it is not rebuilt.
//
// A shelled conformal lattice is `max(cells, |offset| - reach)`, and `|offset|` is at least the
// distance to any box containing the surface — so a ray outside the hierarchy's root box could
// be answered `box_distance - reach` without walking a triangle. It is a valid lower bound and
// it is worth about 40%:
//
// ```text
//                              512 px   1024 px
//   levels 3, shell 10 mm       45.2     112.1     with the early exit
//   levels 3, shell 10 mm       65.8     188.1     without
// ```
//
// It is also wrong in the way a lower bound is allowed to be and a *marcher* is not. The root
// box's ceiling is the surface's highest point, so above a domed panel's rim the box says
// "close" while the surface is the whole rise further down — the exit returns nearly zero where
// the truth is tens of millimetres, the ray crawls, and it runs out of steps in mid-air. Against
// the field: **42.1 mm of disagreement with it, 8.3 mm without.**
//
// A tighter version means descending the hierarchy a few levels rather than reading its root,
// which is most of the walk it was meant to avoid. Worth revisiting only with a measurement
// showing what a depth-limited descent actually saves.

fn surface_vertex(l: SurfaceLayout, tri: u32, k: u32) -> vec3<f32> {
    let v = surface_uints[l.triangles + tri * 3u + k];
    let base = l.positions + v * 3u;
    return vec3<f32>(
        surface_floats[base],
        surface_floats[base + 1u],
        surface_floats[base + 2u],
    );
}

fn surface_uv(l: SurfaceLayout, tri: u32, k: u32) -> vec2<f32> {
    let base = l.uvs + tri * 6u + k * 2u;
    return vec2<f32>(surface_floats[base], surface_floats[base + 1u]);
}

// Squared distance from `p` to a node's box, which is what prunes the walk.
fn surface_box_distance(l: SurfaceLayout, node: u32, p: vec3<f32>) -> f32 {
    let base = l.node_bounds + node * 6u;
    let lo = vec3<f32>(
        surface_floats[base],
        surface_floats[base + 1u],
        surface_floats[base + 2u],
    );
    let hi = vec3<f32>(
        surface_floats[base + 3u],
        surface_floats[base + 4u],
        surface_floats[base + 5u],
    );
    let out = max(max(lo - p, p - hi), vec3<f32>(0.0));
    return dot(out, out);
}

// The closest point on a triangle, by the Voronoi-region method the field uses.
fn closest_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return vec4<f32>(a, 0.0);
    }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return vec4<f32>(b, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        return vec4<f32>(a + ab * (d1 / (d1 - d3)), 0.0);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return vec4<f32>(c, 0.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        return vec4<f32>(a + ac * (d2 / (d2 - d6)), 0.0);
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return vec4<f32>(b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))), 0.0);
    }
    let denom = 1.0 / (va + vb + vc);
    return vec4<f32>(a + ab * (vb * denom) + ac * (vc * denom), 0.0);
}

// The barycentric weights of a point already known to lie on the triangle.
//
// Recovered rather than carried out of `closest_on_triangle`, because WGSL has no tuples and a
// second return value would mean a struct for one call site. The areas are what define them.
fn surface_barycentric(
    p: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    q: vec3<f32>,
) -> vec3<f32> {
    let n = cross(b - a, c - a);
    let area = dot(n, n);
    if (area < 1e-24) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    let wa = dot(cross(c - b, q - b), n) / area;
    let wb = dot(cross(a - c, q - c), n) / area;
    let wc = 1.0 - wa - wb;
    return vec3<f32>(wa, wb, wc);
}

// The fastest a conformal map's pattern varies anywhere, which is the bound a ray steps by.
//
// **Constant, and taken over the whole surface rather than at the point.** `s` and `t` do not
// vary with any radius -- they vary with face size, and a ray does not know which face it is
// about to meet. Asking where it is standing would let it step into a finer region and through
// a wall, which is the failure this bound exists to prevent.
fn conformal_frequency(index: u32, cells: f32, depth: f32) -> f32 {
    let l = surface_layouts[index];
    return max(6.2831853 * cells * l.worst_gradient, 6.2831853 / max(depth, 1e-9));
}

// The three cell arguments over a surface: `2*pi*N*s`, `2*pi*N*t`, and `2*pi*d/depth`.
fn conformal_args(hit: SurfaceHit, cells: f32, depth: f32) -> vec3<f32> {
    let n = 6.2831853 * cells;
    return vec3<f32>(n * hit.st.x, n * hit.st.y, 6.2831853 * hit.along / max(depth, 1e-9));
}

// **One walk per sample, which is why this is a branch rather than four more map arms.** The
// generic path asks the map five separate questions -- arguments, frequency, radius, collapse
// radius, feature size -- and under a conformal map four of those would each mean their own
// closest-point search. Every one of them is answerable from a single hit, so a single hit is
// what this takes.
//
// Nothing collapses on a surface, so there is no degenerate radius here and no early solid
// return: `s` and `t` run nought to one across a face however big it is, where an angular
// argument sweeps a fixed count through a shrinking circle.
fn conformal_lattice_distance(
    p: vec3<f32>,
    cell: f32,
    index: u32,
    cells: f32,
    depth: f32,
    phase: f32,
    sizing: f32,
    shell: f32,
    shift: f32,
) -> f32 {
    // Halved here rather than in the slot, so the slot carries the number the user typed and
    // means one thing. `LatticeMap::Conformal::reach` is the same halving on the field's side.
    let reach = shell * 0.5;
    let hit = surface_at(index, p);
    let args = conformal_args(hit, cells, depth);
    let n = 6.2831853 * cells;
    // The bound is constant here, so the generic path's two-step -- estimate, then re-ask
    // further in -- collapses to one evaluation. Both steps would return this same number.
    let bound = conformal_frequency(index, cells, depth) * lattice_gradient_bound(cell);
    // The band's own face, an exact distance -- `max` of it and a lower bound is still a lower
    // bound, which is the contract this field owes.
    var band = -3.4e38;
    if shell >= 0.0 {
        band = abs(hit.offset) - reach;
    }
    if cell > 5.5 {
        let to_axis = beam_axis_distance(cell, args / 6.2831853) * 6.2831853;
        return max(to_axis / bound - sizing, band);
    }
    var raw: f32;
    if phase < 0.5 {
        // `‖∇arg‖` per argument: `2*pi*N*‖∇s‖` across the surface and `2*pi/depth` through it.
        // Shear in the quad's parametrization makes `∇s` and `∇t` neither orthogonal nor axis
        // aligned, and only their norms reach this formula -- `cell-map-family.md` §1.
        let w = vec3<f32>(
            n * length(hit.ds),
            n * length(hit.dt),
            6.2831853 / max(depth, 1e-9),
        );
        let g = tpms_surface_gradient(cell) * 0.5773503;
        raw = abs(tpms_f(cell, args) - shift * length(w * g)) - sizing * 0.5 * length(w * g);
    } else {
        raw = tpms_f(cell, args) - tpms_density_isovalue(cell, sizing);
    }
    return max(raw / bound, band);
}

// **Exact, and the chain rule is all of it.** `∇F = Σ (∂F/∂arg_k) ∇arg_k`, and a surface hands
// over all three argument gradients as world vectors already -- so unlike the cylindrical and
// spherical arms there is no basis to rotate out of.
fn conformal_lattice_normal(
    p: vec3<f32>,
    cell: f32,
    index: u32,
    cells: f32,
    depth: f32,
    phase: f32,
    sizing: f32,
    shell: f32,
    shift: f32,
) -> vec3<f32> {
    let hit = surface_at(index, p);
    let args = conformal_args(hit, cells, depth);
    let n = 6.2831853 * cells;
    var g = tpms_gradient(cell, args);
    if cell > 5.5 {
        let h = 1e-3;
        let uu = args / 6.2831853;
        g = vec3<f32>(
            beam_axis_distance(cell, uu + vec3<f32>(h, 0.0, 0.0))
                - beam_axis_distance(cell, uu - vec3<f32>(h, 0.0, 0.0)),
            beam_axis_distance(cell, uu + vec3<f32>(0.0, h, 0.0))
                - beam_axis_distance(cell, uu - vec3<f32>(0.0, h, 0.0)),
            beam_axis_distance(cell, uu + vec3<f32>(0.0, 0.0, h))
                - beam_axis_distance(cell, uu - vec3<f32>(0.0, 0.0, h)),
        );
    } else if phase < 0.5 {
        // A wall has two sides and a labyrinth has one -- see `lattice_normal`, where the same
        // flip is explained at length. A solid must not get it. About the wall's own middle.
        let ws = vec3<f32>(
            n * length(hit.ds),
            n * length(hit.dt),
            6.2831853 / max(depth, 1e-9),
        );
        let gs = tpms_surface_gradient(cell) * 0.5773503;
        g = g * sign(tpms_f(cell, args) - shift * length(ws * gs));
    }
    // **On the shell's face, the surface's normal is the answer.** `max` took whichever term
    // was larger, so where the band is the larger one the outward direction is the band's, not
    // the cell's -- and shading the cell's there lights a flat cut as though it were a gyroid
    // wall. The same comparison the distance made, made again.
    if shell >= 0.0 {
        let reach = shell * 0.5;
        let band = abs(hit.offset) - reach;
        var cell_value: f32;
        if cell > 5.5 {
            cell_value = beam_axis_distance(cell, args / 6.2831853) * 6.2831853
                / (conformal_frequency(index, cells, depth) * lattice_gradient_bound(cell))
                - sizing;
        } else if phase < 0.5 {
            let w = vec3<f32>(
                n * length(hit.ds),
                n * length(hit.dt),
                6.2831853 / max(depth, 1e-9),
            );
            let gg = tpms_surface_gradient(cell) * 0.5773503;
            cell_value = (abs(tpms_f(cell, args) - shift * length(w * gg))
                - sizing * 0.5 * length(w * gg))
                / (conformal_frequency(index, cells, depth) * lattice_gradient_bound(cell));
        } else {
            cell_value = (tpms_f(cell, args) - tpms_density_isovalue(cell, sizing))
                / (conformal_frequency(index, cells, depth) * lattice_gradient_bound(cell));
        }
        if band > cell_value {
            // The band's own gradient: the normal over the surface, radial around a rim.
            return hit.outward;
        }
    }
    let dir = hit.ds * (g.x * n)
        + hit.dt * (g.y * n)
        + hit.normal * (g.z * 6.2831853 / max(depth, 1e-9));
    let len = length(dir);
    if len < 1e-9 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return dir / len;
}
