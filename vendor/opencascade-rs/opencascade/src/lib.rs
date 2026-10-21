use thiserror::Error;

/// Re-exported because this crate's public API takes and returns `glam` types, so a caller
/// cannot place a point without it — and depending on it separately would let a consumer pin a
/// different version and get a type mismatch instead of an answer.
pub use glam;

pub mod angle;
pub mod bounding_box;
pub mod mesh;
pub mod primitives;
pub mod section;
pub mod workplane;

mod law_function;
mod make_pipe_shell;

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to write STL file")]
    StlWriteFailed,
    #[error("failed to read STEP file")]
    StepReadFailed,
    #[error("failed to read IGES file")]
    IgesReadFailed,
    #[error("at least one shape is required to write a STEP file")]
    StepWriteNoShapes,
    #[error("failed to transfer shape to STEP writer")]
    StepWriteTransferFailed,
    #[error("failed to write STEP file")]
    StepWriteFailed,
    #[error("failed to write IGES file")]
    IgesWriteFailed,
    #[error("failed to read BREP file")]
    BrepReadFailed,
    #[error("failed to write BREP file")]
    BrepWriteFailed,
    #[error("failed to triangulate Shape")]
    TriangulationFailed,
    #[error("encountered a face with no triangulation")]
    UntriangulatedFace,
    #[error("at least 2 points are required for creating a wire")]
    NotEnoughPoints,
}

/// Serialises the tests, because OCCT is not thread-safe.
///
/// Upstream's tests assume they can run in parallel and OCCT does not allow it: six runs of
/// this binary measured three access violations under the default thread count and none under
/// `--test-threads=1`. The fault is in the library, not in any one test, so the fix is a lock
/// rather than a rewrite of the tests that happened to collide.
///
/// A *modification to vendored code*, which `vendor/OPENCASCADE_VENDOR.md` permits for this
/// crate and forbids for Fidget — the licences differ. Poison is ignored: a failing test must
/// not turn every later test into a different failure.
#[cfg(test)]
pub(crate) fn occt_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static OCCT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    OCCT.lock().unwrap_or_else(|e| e.into_inner())
}
