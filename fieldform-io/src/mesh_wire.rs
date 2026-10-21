// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A compact encoding for handing a mesh between two processes.
//!
//! Not a file format and not part of the document — [`crate::document`] owns that, and this
//! deliberately does not touch it. This exists for one hop: the import subprocess writes a
//! mesh, the parent reads it, the file is deleted. Nothing else may depend on its layout.
//!
//! # Why not JSON
//!
//! The document stores meshes as JSON and should: an archive entry is read by other builds and
//! by people. This is neither. A 425 000-triangle import is around 90 MB of JSON and takes
//! longer to parse than the import it is reporting, which would put the cost straight back
//! into the wait this whole mechanism exists to remove. Raw little-endian arrays are a few
//! lines each way and the parent spends no measurable time on them.
//!
//! Version-tagged anyway, because a stale child binary left in a build directory would
//! otherwise be read as geometry.

use std::io::{Read, Write};

use fieldform_scene::{Curve, MeshData};

/// Bumped whenever the layout below changes. A mismatch is an error, never a guess.
const VERSION: u32 = 3;
const MAGIC: [u8; 4] = *b"FFMW";

/// Why an encoded mesh could not be read back.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("the import result is not readable: {0}")]
    Io(#[from] std::io::Error),
    #[error("the import result is not a mesh (wrong magic)")]
    NotAMesh,
    #[error("the import result was written by a different build (version {found}, expected {VERSION})")]
    Version { found: u32 },
    #[error("the import result is truncated")]
    Truncated,
}

/// Writes several meshes and no curves.
///
/// The importer always sends the curves beside the meshes, through [`write_payload`], so only
/// the tests write meshes alone, and only the tests compile this, as with [`read_all`].
#[cfg(test)]
fn write_all(out: &mut impl Write, meshes: &[MeshData]) -> std::io::Result<()> {
    write_payload(out, meshes, &[])
}

/// Writes several meshes, which is what one import produces — a STEP assembly is many solids
/// and each becomes a body of its own — with the curves the source held that bound no face.
///
/// The format is one payload whether or not there are curves: a file with none carries an
/// empty curve list, not a missing one, so there is a single thing to read.
pub fn write_payload(
    out: &mut impl Write,
    meshes: &[MeshData],
    curves: &[Curve],
) -> std::io::Result<()> {
    out.write_all(&(meshes.len() as u64).to_le_bytes())?;
    for mesh in meshes {
        write(out, mesh)?;
    }
    // After the meshes, so a reader that has just finished them is positioned here. Serialised
    // as JSON rather than field by field: a curve is a handful of numbers where a mesh is
    // millions, so the compactness the rest of this format exists for buys nothing, and
    // `CurveData` already round-trips through serde as the document's own on-disk form.
    let encoded = serde_json::to_vec(
        &curves.iter().cloned().map(fieldform_scene::CurveData::from).collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| b"[]".to_vec());
    out.write_all(&(encoded.len() as u64).to_le_bytes())?;
    out.write_all(&encoded)?;
    Ok(())
}

/// Reads back what [`write_all`] produced, keeping only the meshes.
///
/// The application reads with [`read_payload`], since the importer sends curves beside the
/// meshes, so only the tests read meshes alone and only the tests compile this.
#[cfg(test)]
fn read_all(input: &mut impl Read) -> Result<Vec<MeshData>, WireError> {
    read_payload(input).map(|(meshes, _)| meshes)
}

/// Reads back what [`write_payload`] produced.
pub fn read_payload(input: &mut impl Read) -> Result<(Vec<MeshData>, Vec<Curve>), WireError> {
    let count = u64::from_le_bytes(exact::<8>(input)?) as usize;
    // Not pre-allocated to `count`: it is a number from a file, and trusting it would let a
    // truncated result ask for gigabytes before failing.
    let mut meshes = Vec::new();
    for _ in 0..count {
        meshes.push(read(input)?);
    }
    let length = u64::from_le_bytes(exact::<8>(input)?) as usize;
    // Read in fixed chunks for the same reason the count is not pre-allocated.
    let mut encoded = Vec::new();
    let mut left = length;
    while left > 0 {
        let mut chunk = vec![0u8; left.min(64 * 1024)];
        input.read_exact(&mut chunk).map_err(|e| match e.kind() {
            std::io::ErrorKind::UnexpectedEof => WireError::Truncated,
            _ => WireError::Io(e),
        })?;
        left -= chunk.len();
        encoded.extend_from_slice(&chunk);
    }
    let data: Vec<fieldform_scene::CurveData> =
        serde_json::from_slice(&encoded).map_err(|_| WireError::Truncated)?;
    // A curve whose numbers do not make a curve is dropped rather than failing the import, the
    // same way the importer treats one it could not read: the meshes beside it are still good.
    let curves = data.into_iter().filter_map(|d| Curve::try_from(d).ok()).collect();
    Ok((meshes, curves))
}

/// Exactly `N` bytes, or [`WireError::Truncated`].
///
/// Reading exact amounts rather than slurping the whole stream is what lets meshes be read one
/// after another from the same reader — `read_to_end` would consume the second mesh while
/// reading the first.
fn exact<const N: usize>(input: &mut impl Read) -> Result<[u8; N], WireError> {
    let mut buf = [0u8; N];
    match input.read_exact(&mut buf) {
        Ok(()) => Ok(buf),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(WireError::Truncated),
        Err(e) => Err(WireError::Io(e)),
    }
}

/// Writes a mesh in the layout [`read`] expects.
pub(crate) fn write(out: &mut impl Write, mesh: &MeshData) -> std::io::Result<()> {
    out.write_all(&MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;

    let name = mesh.name.as_bytes();
    out.write_all(&(name.len() as u64).to_le_bytes())?;
    out.write_all(name)?;

    // Tolerance as a pair rather than an Option, since there is no bit pattern for "absent"
    // that is not also a number someone could legitimately mean.
    out.write_all(&[u8::from(mesh.tolerance_mm.is_some())])?;
    out.write_all(&mesh.tolerance_mm.unwrap_or(0.0).to_le_bytes())?;
    out.write_all(&(mesh.open_edges as u64).to_le_bytes())?;

    out.write_all(&(mesh.positions.len() as u64).to_le_bytes())?;
    for p in &mesh.positions {
        for v in p {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.write_all(&(mesh.triangles.len() as u64).to_le_bytes())?;
    for t in &mesh.triangles {
        for v in t {
            out.write_all(&v.to_le_bytes())?;
        }
    }

    // The solid's own edges, as polylines of varying length — so each carries its own count
    // rather than the block carrying one stride. Added in version 2, which is why the version
    // is a version rather than a formality: a stale child binary writing version 1 would
    // otherwise be read as geometry that stopped halfway.
    out.write_all(&(mesh.edges.len() as u64).to_le_bytes())?;
    for line in &mesh.edges {
        out.write_all(&(line.len() as u64).to_le_bytes())?;
        for p in line {
            for v in p {
                out.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Reads one mesh, leaving the reader positioned at whatever follows it.
pub(crate) fn read(input: &mut impl Read) -> Result<MeshData, WireError> {
    if exact::<4>(input)? != MAGIC {
        return Err(WireError::NotAMesh);
    }
    let found = u32::from_le_bytes(exact::<4>(input)?);
    if found != VERSION {
        return Err(WireError::Version { found });
    }

    let name_len = u64::from_le_bytes(exact::<8>(input)?) as usize;
    let mut name = vec![0u8; name_len];
    input.read_exact(&mut name).map_err(|_| WireError::Truncated)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    let has_tolerance = exact::<1>(input)?[0] != 0;
    let tolerance = f64::from_le_bytes(exact::<8>(input)?);
    let open_edges = u64::from_le_bytes(exact::<8>(input)?) as usize;

    let position_count = u64::from_le_bytes(exact::<8>(input)?) as usize;
    let mut positions = Vec::new();
    for _ in 0..position_count {
        positions.push([
            f64::from_le_bytes(exact::<8>(input)?),
            f64::from_le_bytes(exact::<8>(input)?),
            f64::from_le_bytes(exact::<8>(input)?),
        ]);
    }

    let triangle_count = u64::from_le_bytes(exact::<8>(input)?) as usize;
    let mut triangles = Vec::new();
    for _ in 0..triangle_count {
        triangles.push([
            u32::from_le_bytes(exact::<4>(input)?),
            u32::from_le_bytes(exact::<4>(input)?),
            u32::from_le_bytes(exact::<4>(input)?),
        ]);
    }

    let edge_count = u64::from_le_bytes(exact::<8>(input)?) as usize;
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let points = u64::from_le_bytes(exact::<8>(input)?) as usize;
        let mut line = Vec::with_capacity(points);
        for _ in 0..points {
            line.push([
                f64::from_le_bytes(exact::<8>(input)?),
                f64::from_le_bytes(exact::<8>(input)?),
                f64::from_le_bytes(exact::<8>(input)?),
            ]);
        }
        edges.push(line);
    }

    Ok(MeshData {
        // The wire carries geometry, not provenance: the importer is a separate process and
        // has no idea which document it is feeding. The application tags the meshes.
        source: None,
        name,
        positions,
        triangles,
        edges,
        tolerance_mm: has_tolerance.then_some(tolerance),
        open_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MeshData {
        MeshData {
            source: None,
            // Two edges of differing length, so the per-line count is exercised rather than
            // assumed: a stride would round-trip a fixture where every line is the same size
            // and lose the second point of every line in a real one.
            edges: vec![
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]],
            ],
            name: "a mesh with a name".to_owned(),
            positions: vec![[0.0, 1.5, -2.25], [1e6, -1e-6, 0.0]],
            triangles: vec![[0, 1, 0], [1, 0, 1]],
            tolerance_mm: Some(0.05),
            open_edges: 7,
        }
    }

    /// Curves cross the wire with their numbers intact, which is what makes them importable.
    ///
    /// The importer runs as its own process, so a curve that survives OCCT and then loses a
    /// weight or a knot on the way back is no better than one that was never read. Asserted on
    /// the numbers rather than on the count, since a curve with the right shape and the wrong
    /// knot vector is exactly the failure that would look fine in a list.
    #[test]
    fn curves_round_trip_through_the_wire_beside_the_meshes() {
        use fieldform_scene::{Curve, CurveId};
        use nalgebra::Point3;

        // A rational quadratic, so the weights are not all one and a dropped weight shows.
        let curve = Curve::new(
            CurveId::new(),
            "arc".to_string(),
            2,
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(20.0, 0.0, 0.0),
            ],
            vec![1.0, 0.5_f64.sqrt(), 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a valid rational quadratic");

        let mut buffer = Vec::new();
        write_payload(&mut buffer, &[], std::slice::from_ref(&curve)).expect("writes");
        let (meshes, back) = read_payload(&mut buffer.as_slice()).expect("reads");

        assert!(meshes.is_empty(), "no meshes were written");
        assert_eq!(back.len(), 1, "the curve did not survive");
        let back = &back[0];
        assert_eq!(back.degree(), curve.degree());
        assert_eq!(back.control_points(), curve.control_points());
        assert_eq!(back.weights(), curve.weights());
        assert_eq!(back.knots(), curve.knots());
        assert_eq!(back.name, curve.name);
    }

    /// Every field survives the trip, exactly. A mesh that came back subtly different would
    /// be geometry that is wrong in a way nothing downstream could notice.
    #[test]
    fn a_mesh_round_trips_through_the_wire_unchanged() {
        let mesh = sample();
        let mut buf = Vec::new();
        write(&mut buf, &mesh).expect("writing to a Vec cannot fail");
        let back = read(&mut buf.as_slice()).expect("what we just wrote");
        assert_eq!(back, mesh);
    }

    /// An absent tolerance is absent on the other side, rather than becoming zero — which
    /// would claim the source was tessellated infinitely finely.
    #[test]
    fn an_absent_tolerance_stays_absent() {
        let mesh = MeshData { tolerance_mm: None, ..sample() };
        let mut buf = Vec::new();
        write(&mut buf, &mesh).expect("writing to a Vec cannot fail");
        assert_eq!(read(&mut buf.as_slice()).expect("valid").tolerance_mm, None);
    }

    /// Anything that is not this format is refused rather than interpreted. The realistic way
    /// to meet one is a stale child binary in a build directory.
    #[test]
    fn something_that_is_not_a_mesh_is_refused() {
        let err = read(&mut b"not a mesh at all".as_slice()).expect_err("not a mesh");
        assert!(matches!(err, WireError::NotAMesh), "{err:?}");

        let mut wrong_version = Vec::new();
        wrong_version.extend_from_slice(&MAGIC);
        wrong_version.extend_from_slice(&99u32.to_le_bytes());
        let err = read(&mut wrong_version.as_slice()).expect_err("wrong version");
        assert!(matches!(err, WireError::Version { found: 99 }), "{err:?}");
    }

    /// Several meshes in one result, which is what a STEP assembly produces. The second must
    /// come back whole, which only works because each mesh reads exactly its own bytes rather
    /// than slurping the stream.
    #[test]
    fn a_list_of_meshes_round_trips_in_order() {
        let meshes = vec![
            sample(),
            MeshData { name: "second".to_owned(), open_edges: 0, ..sample() },
        ];
        let mut buf = Vec::new();
        write_all(&mut buf, &meshes).expect("writing to a Vec cannot fail");
        let back = read_all(&mut buf.as_slice()).expect("what we just wrote");
        assert_eq!(back, meshes);
    }

    /// An empty list is a list, not an error: a file holding no geometry still has to come
    /// back as "no meshes" rather than as a broken result.
    #[test]
    fn an_empty_list_round_trips() {
        let mut buf = Vec::new();
        write_all(&mut buf, &[]).expect("writing to a Vec cannot fail");
        assert!(read_all(&mut buf.as_slice()).expect("valid").is_empty());
    }

    /// A result cut short — the child was killed mid-write — is an error, not a mesh missing
    /// its last triangles.
    #[test]
    fn a_truncated_result_is_refused_rather_than_half_read() {
        let mesh = sample();
        let mut buf = Vec::new();
        write(&mut buf, &mesh).expect("writing to a Vec cannot fail");
        buf.truncate(buf.len() - 5);
        let err = read(&mut buf.as_slice()).expect_err("truncated");
        assert!(matches!(err, WireError::Truncated), "{err:?}");
    }
}
