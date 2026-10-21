// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Writing a mesh out as STL, which is what a printer and a slicer read.
//!
//! # Binary, not ASCII
//!
//! The two formats describe the same thing and one of them is 5–6x the size. A 425 000-triangle
//! export is 21 MB binary and around 120 MB as text, and nothing reads the text that cannot read
//! the binary. ASCII STL is worth having only for looking at by eye, which a mesh this size is
//! not.
//!
//! # What STL cannot say
//!
//! It has no units, no vertex sharing and no normals worth the name. Each triangle repeats its
//! three vertices in full and carries a facet normal that most readers ignore in favour of the
//! winding — so a 500 000-triangle mesh writes 1.5 million vertices, and the same corner appears
//! once per triangle that touches it.
//!
//! That is not a defect to work around here. It is what the format is, every slicer expects it,
//! and a document that wanted vertex sharing and units would be saved as a `.fform` rather
//! than exported. **The unit is the one to be careful about**: geometry is canonical millimetres
//! (D2) and STL declares nothing, so a file lands in whatever the reader assumes — almost always
//! millimetres, which is why the canonical unit being millimetres is what makes this safe.

use std::io::{BufWriter, Write};
use std::path::Path;

use nalgebra::Point3;

/// Why a mesh could not be written.
#[derive(Debug, thiserror::Error)]
pub enum StlError {
    #[error("could not write the file: {0}")]
    Io(#[from] std::io::Error),
    #[error("there is no geometry to export")]
    Empty,
    /// Stopped by its block before it finished. Not a failure of the file or the mesh, and
    /// nothing was left behind: see [`write_stl_cancellable`].
    #[error("the export was stopped")]
    Cancelled,
    #[error(
        "STL counts its triangles in 32 bits and this mesh has {0} — export a coarser mesh, or \
         a part of it"
    )]
    TooManyTriangles(usize),
}

/// Writes a mesh to `path` as binary STL, with no way to stop it.
///
/// The application always exports through [`write_stl_cancellable`], since a block cancels its
/// write whenever its mesh goes stale, so only the tests write without a stop, and only the
/// tests compile this (decision 1.4).
#[cfg(test)]
fn write_stl(
    path: &Path,
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
) -> Result<(), StlError> {
    write_stl_cancellable(path, positions, triangles, &|| false)
}

/// Writes a mesh to `path` as binary STL, stoppably, and **never leaving half a file behind**.
///
/// **Positions and triangles, not a mesh type.** STL has nowhere to put a per-vertex normal, so
/// a writer that took one would be asking for something it throws away — and taking slices keeps
/// this crate off `fieldform-field`, which is a *dev*-dependency here on purpose: the STEP
/// importer links this crate and has no use for a BVH.
///
/// The normal written per facet is the **triangle's own**. A reader that uses it wants the flat
/// facet normal for the triangle it belongs to, and writing one of the three smooth vertex
/// normals into that slot would light a flat triangle as though it were curved.
///
/// Written beside the target under a temporary name and renamed over it only once the last byte
/// is down. So a cancel, a full disk or a crash leaves either the previous file or none — never
/// a truncated STL with a header claiming more triangles than follow, which a slicer reads as a
/// corrupt part and a person reads as the export having worked. The same rule a cancelled import
/// already follows. Beside the target rather than in a temporary directory because a rename is
/// only atomic within one volume.
///
/// `stop` is asked every sixty-four thousand triangles — a few milliseconds of writing. A
/// question rather than a token type, because this crate does not link the field crate the
/// token lives in, and keeping it that way is what keeps the STEP importer small.
pub fn write_stl_cancellable(
    path: &Path,
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
    stop: &dyn Fn() -> bool,
) -> Result<(), StlError> {
    if triangles.is_empty() {
        return Err(StlError::Empty);
    }
    let partial = {
        let mut name = path.as_os_str().to_owned();
        name.push(".partial");
        std::path::PathBuf::from(name)
    };
    let written = write_binary(&partial, positions, triangles, stop);
    match written {
        Ok(()) => std::fs::rename(&partial, path).map_err(|e| {
            let _ = std::fs::remove_file(&partial);
            StlError::Io(e)
        }),
        Err(e) => {
            // Whatever stopped it — a cancel or an error — the partial file is not the export.
            let _ = std::fs::remove_file(&partial);
            Err(e)
        }
    }
}

fn write_binary(
    path: &Path,
    positions: &[Point3<f64>],
    triangles: &[[u32; 3]],
    stop: &dyn Fn() -> bool,
) -> Result<(), StlError> {
    let count = u32::try_from(triangles.len())
        .map_err(|_| StlError::TooManyTriangles(triangles.len()))?;

    let file = std::fs::File::create(path)?;
    let mut out = BufWriter::new(file);

    // 80 bytes of header that must not begin with "solid", or a reader sniffing the format
    // takes the file for ASCII and fails on the binary that follows. Naming the writer is the
    // conventional use of the space and doubles as the thing that makes the first word safe.
    let mut header = [0u8; 80];
    let tag = b"fieldform binary STL";
    header[..tag.len()].copy_from_slice(tag);
    out.write_all(&header)?;
    out.write_all(&count.to_le_bytes())?;

    for (n, tri) in triangles.iter().enumerate() {
        if n % 65536 == 0 && stop() {
            return Err(StlError::Cancelled);
        }
        let p = [
            positions[tri[0] as usize],
            positions[tri[1] as usize],
            positions[tri[2] as usize],
        ];
        // From the winding, so the two agree by construction. A facet normal that disagreed
        // with its own winding is the one thing an STL can be internally inconsistent about,
        // and readers split on which they believe.
        let normal = (p[1] - p[0])
            .cross(&(p[2] - p[0]))
            .try_normalize(1e-12)
            .unwrap_or_default();
        for v in [normal.x, normal.y, normal.z] {
            out.write_all(&(v as f32).to_le_bytes())?;
        }
        for vertex in p {
            for v in [vertex.x, vertex.y, vertex.z] {
                out.write_all(&(v as f32).to_le_bytes())?;
            }
        }
        // The attribute byte count, which is zero in every STL anyone writes. Some tools store
        // a colour here; nothing reads it reliably, so nothing is stored.
        out.write_all(&0u16.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

/// How many bytes `write_stl_cancellable` will produce for a mesh of `triangles`.
///
/// Offered so a caller can say the size before writing it. An export is the one operation whose
/// output leaves the application, and "how big is this going to be" is the question a person
/// asks before choosing a resolution rather than after.
pub fn stl_size_bytes(triangles: usize) -> usize {
    84 + triangles * 50
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A stopped export leaves the previous file exactly as it was, and no partial one.** The
    /// block cancels a write whenever its mesh goes stale, so this is the ordinary path rather than
    /// an edge case — and a truncated STL is a corrupt part that looks like a finished one.
    #[test]
    fn a_cancelled_write_leaves_the_old_file_and_no_partial_one() {
        let dir = std::env::temp_dir().join(format!("ff-stl-cancel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("part.stl");
        std::fs::write(&path, b"the previous export").unwrap();

        let (positions, triangles) = tetrahedron();
        let result = write_stl_cancellable(&path, &positions, &triangles, &|| true);
        assert!(matches!(result, Err(StlError::Cancelled)), "got {result:?}");
        assert_eq!(std::fs::read(&path).unwrap(), b"the previous export");
        assert!(!dir.join("part.stl.partial").exists(), "the partial file was left behind");

        // And one that finishes replaces it, leaving no partial file either.
        write_stl(&path, &positions, &triangles).expect("writes");
        assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, stl_size_bytes(4));
        assert!(!dir.join("part.stl.partial").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tetrahedron() -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
        (
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
    }

    fn triangle() -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
        (
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fieldform-stl-{name}"));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir.join("out.stl")
    }

    /// The layout, byte for byte, because a slicer that reads this file will not be forgiving
    /// about it and there is no partial credit.
    #[test]
    fn a_binary_stl_has_the_layout_every_reader_expects() {
        let path = scratch("layout");
        let (positions, triangles) = triangle();
        write_stl(&path, &positions, &triangles).expect("writes");
        let bytes = std::fs::read(&path).expect("reads back");

        assert_eq!(bytes.len(), stl_size_bytes(1), "84 bytes of preamble and 50 per triangle");
        assert_eq!(u32::from_le_bytes(bytes[80..84].try_into().unwrap()), 1);
        // The header must not begin with "solid", or the file is sniffed as ASCII.
        assert!(!bytes.starts_with(b"solid"));
        // 84 of preamble, then twelve floats, then the attribute pair that closes the record.
        let attrs = 84 + 12 * 4;
        assert_eq!(u16::from_le_bytes(bytes[attrs..attrs + 2].try_into().unwrap()), 0);
    }

    /// The facet normal agrees with the winding, which is the one inconsistency an STL invites.
    #[test]
    fn the_facet_normal_agrees_with_the_winding() {
        let path = scratch("normal");
        let (positions, triangles) = triangle();
        write_stl(&path, &positions, &triangles).expect("writes");
        let bytes = std::fs::read(&path).expect("reads back");
        let at = |i: usize| f32::from_le_bytes(bytes[84 + i * 4..88 + i * 4].try_into().unwrap());
        // The triangle lies in z = 0 wound anticlockwise, so its normal is +Z.
        assert!((at(0)).abs() < 1e-6);
        assert!((at(1)).abs() < 1e-6);
        assert!((at(2) - 1.0).abs() < 1e-6);
    }

    /// An empty mesh is refused rather than written as a file with no triangles in it.
    ///
    /// A zero-triangle STL is *valid*, which is what makes it the wrong answer: a slicer opens
    /// it, shows nothing, and the person is left asking whether the export or the model was
    /// empty. Saying so here answers that.
    #[test]
    fn an_empty_mesh_is_refused_rather_than_written_as_an_empty_file() {
        let path = scratch("empty");
        assert!(matches!(write_stl(&path, &[], &[]), Err(StlError::Empty)));
    }
}
