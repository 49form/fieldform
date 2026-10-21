pub use inner::*;

#[cxx::bridge]
mod inner {
    unsafe extern "C++" {
        include!("opencascade-sys/include/b_rep_mesh.hxx");

        type TopoDS_Shape = crate::topo_ds::TopoDS_Shape;

        type BRepMesh_IncrementalMesh;
        #[cxx_name = "construct_unique"]
        fn IncrementalMesh_new(
            shape: &TopoDS_Shape,
            deflection: f64,
        ) -> UniquePtr<BRepMesh_IncrementalMesh>;
        /// As above, with the angular criterion set as well, and optionally threaded.
        ///
        /// Added here (see `vendor/OPENCASCADE_VENDOR.md`): the two-argument form leaves the
        /// angular deflection at OCCT's 0.5 rad default, which is what faceting on a curved
        /// face actually is.
        fn IncrementalMesh_new_angular(
            shape: &TopoDS_Shape,
            deflection: f64,
            angular_deflection: f64,
            parallel: bool,
        ) -> UniquePtr<BRepMesh_IncrementalMesh>;
        fn Shape(self: &BRepMesh_IncrementalMesh) -> &TopoDS_Shape;
        fn IsDone(self: &BRepMesh_IncrementalMesh) -> bool;
    }
}
