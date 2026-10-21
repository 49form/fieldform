// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Import/export, deliberately kept at the boundary of the app:
//! - OBJ import for control cages (a cage authored elsewhere becomes editable geometry).
//! - Mesh export (STL/OBJ) via `fidget-mesh`'s Manifold Dual Contouring, for 3D
//!   printing/CAM/downstream tools. Export-only — polygons produced here are never fed
//!   back into the editing representation.

pub mod document;
mod format;
mod iges;
pub mod mesh_wire;
pub mod stl;
/// The STEP reader, behind a feature because it links OpenCASCADE.
///
/// A 2.9 GB build dependency, and 29.4 MB of any executable that reaches it, for one file
/// format. Off by default so the application — which never imports in-process — does not carry
/// it; `fieldform-import` turns it on and is the only thing that does. It also means a machine
/// without OCCT can still build and test everything else here.
#[cfg(feature = "step")]
pub mod step;

pub use document::{from_archive, open, save_new, to_archive, update_archive};
pub use format::{Archive, FormatError};
pub use iges::{import_iges, IgesError, IgesImport};
// Re-exported from `fieldform-scene`, where the settings moved so the *document* can hold
// them: an import is a block now, and a block's settings have to survive a save. Nothing about
// the split changed — they are still distances, an angle and two switches with no knowledge of
// OCCT, which is what kept the kernel out of the application in the first place.
pub use fieldform_scene::StepOptions;
#[cfg(feature = "step")]
pub use step::{import_step_with, StepError, StepReport};

use fieldform_cage::{Cage, CageError};
use fieldform_scene::{MeshData, Unit};
use nalgebra::Point3;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum ImportError {
    #[error("failed to load OBJ file: {0}")]
    Obj(#[from] tobj::LoadError),
    #[error("OBJ file contains no objects")]
    Empty,
    #[error("object {index} (\"{name}\") is not a valid cage: {source}")]
    InvalidCage {
        index: usize,
        name: String,
        #[source]
        source: CageError,
    },
}

/// Imports every object/group in an OBJ file as a separate [`Cage`], preserving each face's
/// original vertex count (quads stay quads) rather than triangulating — `Cage::from_faces`
/// is what actually enforces the closed/manifold/all-quad invariants, so a file containing
/// triangles, n-gons, or an open mesh fails here with a specific error rather than silently
/// becoming an unusable cage.
pub fn import_cages(path: &Path) -> Result<Vec<Cage>, ImportError> {
    import_cages_in(path, Unit::Millimetre)
}

/// Imports an OBJ, interpreting its coordinates as `unit`.
///
/// **OBJ carries no unit at all** — the format has no field for one, and no convention either.
/// A file saying `v 0.5 0.5 -0.5` might be half a millimetre or half a metre, and nothing in
/// it can tell you which. DCC tools, which write most of the OBJs that reach a modeller, treat
/// one unit as one metre; CAD exporters often mean millimetres.
///
/// So this cannot be inferred and must not be guessed silently: importing a 1-unit DCC-tool
/// cube as 1 mm rather than 1000 mm is a factor of a thousand, and it looks like the geometry
/// arrived correctly. The unit is therefore an explicit argument, chosen at the point of
/// import, and converted here into the document's canonical millimetres — the same boundary
/// where IGES's declared unit flag is applied.
pub fn import_cages_in(path: &Path, unit: Unit) -> Result<Vec<Cage>, ImportError> {
    let (models, _materials) = tobj::load_obj(path, &quad_options())?;
    if models.is_empty() {
        return Err(ImportError::Empty);
    }

    models
        .into_iter()
        .enumerate()
        .map(|(index, model)| {
            let (positions, faces) = model_faces(&model.mesh, unit);
            Cage::from_faces(positions, &faces).map_err(|source| ImportError::InvalidCage {
                index,
                name: model.name,
                source,
            })
        })
        .collect()
}

/// Imports every object in an OBJ as an **open quad surface** — a shape to follow, not a solid.
///
/// The third door, and the one item 45 opened. `import_cages_in` refuses a mesh with a boundary
/// because a `Cage` is closed by construction: `CageField` signs a point by enclosure, and an
/// open surface has no inside to be in. A panel or a shell is exactly the open surface somebody
/// wants a conformal lattice laid over, so it needs somewhere to go that is not a body — hence
/// `Document::quad_meshes`, which holds a surface to reference rather than geometry to draw.
///
/// Still all quads: a `QuadMesh` is the input to Catmull-Clark, so a triangle in the file is
/// refused here exactly as it is for a cage. What is *not* refused is a boundary.
pub fn import_surfaces_in(
    path: &Path,
    unit: Unit,
) -> Result<Vec<(String, fieldform_cage::QuadMesh)>, ImportError> {
    let (models, _materials) = tobj::load_obj(path, &quad_options())?;
    if models.is_empty() {
        return Err(ImportError::Empty);
    }
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string());
    let many = models.len() > 1;
    models
        .into_iter()
        .enumerate()
        .map(|(index, model)| {
            let name = match (model.name.is_empty(), &stem) {
                (false, _) => model.name.clone(),
                (true, Some(stem)) if many => format!("{stem} ({})", index + 1),
                (true, Some(stem)) => stem.clone(),
                (true, None) => format!("surface {}", index + 1),
            };
            let (positions, faces) = model_faces(&model.mesh, unit);
            fieldform_cage::QuadMesh::from_faces(positions, &faces)
                .map(|mesh| (name, mesh))
                .map_err(|source| ImportError::InvalidCage {
                    index,
                    name: model.name,
                    source,
                })
        })
        .collect()
}

/// What both quad doors ask `tobj` for: faces as they were written, not triangulated.
fn quad_options() -> tobj::LoadOptions {
    tobj::LoadOptions {
        triangulate: false,
        ignore_points: true,
        ignore_lines: true,
        ..Default::default()
    }
}

/// One model's vertices and faces, in canonical millimetres.
///
/// Shared by both quad doors so the two cannot read the same file differently — the only thing
/// that distinguishes them is which constructor the faces are handed to, which is precisely the
/// closed/open question and nothing else.
fn model_faces(mesh: &tobj::Mesh, unit: Unit) -> (Vec<Point3<f64>>, Vec<Vec<u32>>) {
    // OBJ carries no unit and `tobj` parses to f32; widening here puts imported geometry into
    // the document's canonical f64 millimetres. Widening cannot add precision the file never
    // had, but it stops the pipeline from compounding f32 error downstream.
    let scale = unit.in_millimetres();
    let positions: Vec<Point3<f64>> = mesh
        .positions
        .chunks_exact(3)
        .map(|c| {
            Point3::new(
                c[0] as f64 * scale,
                c[1] as f64 * scale,
                c[2] as f64 * scale,
            )
        })
        .collect();

    // `face_arities` is only empty when every face already has arity 3 (or when triangulated,
    // which we don't request) — in that case treat `indices` as a flat run of triangles rather
    // than silently producing zero faces.
    let arities: Vec<u32> = if mesh.face_arities.is_empty() && !mesh.indices.is_empty() {
        vec![3; mesh.indices.len() / 3]
    } else {
        mesh.face_arities.clone()
    };
    let mut faces = Vec::with_capacity(arities.len());
    let mut offset = 0usize;
    for arity in arities {
        let arity = arity as usize;
        faces.push(mesh.indices[offset..offset + arity].to_vec());
        offset += arity;
    }
    (positions, faces)
}

/// Imports every object in an OBJ as a triangle mesh body.
///
/// The other door, and the one that should usually be used. `import_cages_in` asks for a
/// **cage** — closed, manifold, all quads — because a cage is scaffolding for a subdivision
/// surface, and most OBJs are not that and never will be. As the import design note
/// puts it: imported geometry is *finished geometry to be referenced, cut against or blended
/// with, not scaffolding to be pushed and pulled*.
///
/// So this triangulates and hands back meshes, which the document can hold beside whatever
/// else is in it. A mesh with holes still comes through — `open_edges` records how many, and
/// the caller decides what that means. Refusing a file because its tessellator dropped a face
/// would throw away the geometry that did arrive.
pub fn import_meshes_in(path: &Path, unit: Unit) -> Result<Vec<MeshData>, ImportError> {
    let options = tobj::LoadOptions {
        triangulate: true,
        ignore_points: true,
        ignore_lines: true,
        ..Default::default()
    };
    let (models, _materials) = tobj::load_obj(path, &options)?;
    if models.is_empty() {
        return Err(ImportError::Empty);
    }

    let scale = unit.in_millimetres();
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string());
    let many = models.len() > 1;
    Ok(models
        .into_iter()
        .enumerate()
        .filter_map(|(index, model)| {
            let mesh = model.mesh;
            if mesh.indices.len() < 3 {
                return None;
            }
            let positions: Vec<[f64; 3]> = mesh
                .positions
                .chunks_exact(3)
                .map(|c| [c[0] as f64 * scale, c[1] as f64 * scale, c[2] as f64 * scale])
                .collect();
            let triangles: Vec<[u32; 3]> =
                mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
            // The object's own name when it has one, since an OBJ can carry several and they
            // are what the file itself calls them. Numbered only when there is more than one,
            // the same rule the STEP importer follows.
            let name = match (model.name.is_empty(), &stem) {
                (false, _) => model.name.clone(),
                (true, Some(stem)) if many => format!("{stem} ({})", index + 1),
                (true, Some(stem)) => stem.clone(),
                (true, None) => format!("object {}", index + 1),
            };
            let open_edges = MeshData::count_open_edges(&triangles);
            Some(MeshData {
                name,
                positions,
                triangles,
                // An OBJ is triangles and nothing else: there is no B-rep behind it, so there
                // are no edges to draw over it.
                edges: Vec::new(),
                // An OBJ is already triangles: there is no curved surface behind them and so
                // no tolerance they were produced at. Inventing one would be a lie about how
                // faithful the file is.
                tolerance_mm: None,
                open_edges,
                source: None,
            })
        })
        .collect())
}

// STL export is `stl::write_stl_cancellable`. OBJ export is not built: STL is what a slicer
// reads, and a second text format with the same content is worth adding when something asks
// for it.

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_obj(name: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path
    }

    const CUBE_OBJ: &str = "\
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 4 8 7 3
f 1 5 8 4
f 2 3 7 6
";

    #[test]
    fn imports_a_closed_quad_cube() {
        let path = write_temp_obj("fieldform_test_cube.obj", CUBE_OBJ);
        let cages = import_cages(&path).unwrap();
        assert_eq!(cages.len(), 1);
        assert_eq!(cages[0].vertex_count(), 8);
        assert_eq!(cages[0].face_count(), 6);
        assert_eq!(cages[0].edge_count(), 12);
    }

    /// The bug this argument exists to prevent: a 1-unit DCC-tool cube is one *metre*, and
    /// importing it as one millimetre is wrong by a factor of a thousand while looking
    /// perfectly plausible on screen.
    #[test]
    fn the_import_unit_scales_geometry_into_millimetres() {
        let path = write_temp_obj("fieldform_test_unit_cube.obj", CUBE_OBJ);

        let mm = import_cages_in(&path, Unit::Millimetre).unwrap();
        let metres = import_cages_in(&path, Unit::Metre).unwrap();

        let span = |c: &Cage| {
            let xs: Vec<f64> = c.vertex_ids().map(|v| c.vertex_position(v).x).collect();
            xs.iter().cloned().fold(f64::MIN, f64::max)
                - xs.iter().cloned().fold(f64::MAX, f64::min)
        };

        assert!((span(&mm[0]) - 1.0).abs() < 1e-9, "{}", span(&mm[0]));
        assert!((span(&metres[0]) - 1000.0).abs() < 1e-6, "{}", span(&metres[0]));
        // The default stays millimetres, so existing behaviour is unchanged.
        assert!((span(&import_cages(&path).unwrap()[0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_a_triangulated_file() {
        let triangulated = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 4 3
f 1 3 2
f 5 6 7
f 5 7 8
";
        let path = write_temp_obj("fieldform_test_triangulated.obj", triangulated);
        let result = import_cages(&path);
        assert!(matches!(result, Err(ImportError::InvalidCage { .. })));
    }

    #[test]
    fn rejects_an_open_mesh() {
        let open = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
";
        let path = write_temp_obj("fieldform_test_open.obj", open);
        let result = import_cages(&path);
        assert!(matches!(result, Err(ImportError::InvalidCage { .. })));
    }
}
