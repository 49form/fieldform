pub use inner::*;

#[cxx::bridge]
mod inner {
    unsafe extern "C++" {
        include!("opencascade-sys/include/shape_fix.hxx");

        type TopoDS_Shape = crate::topo_ds::TopoDS_Shape;

        type BRepBuilderAPI_Sewing;
        pub fn Sewing_new(tolerance: f64) -> UniquePtr<BRepBuilderAPI_Sewing>;
        pub fn Sewing_add(sewing: Pin<&mut BRepBuilderAPI_Sewing>, shape: &TopoDS_Shape);
        pub fn Sewing_perform(sewing: Pin<&mut BRepBuilderAPI_Sewing>);
        pub fn Sewing_sewed_shape(sewing: &BRepBuilderAPI_Sewing) -> UniquePtr<TopoDS_Shape>;

        type ShapeFix_Shape;
        pub fn ShapeFix_Shape_new(shape: &TopoDS_Shape) -> UniquePtr<ShapeFix_Shape>;
        pub fn ShapeFix_Shape_perform(fix: Pin<&mut ShapeFix_Shape>) -> bool;
        pub fn ShapeFix_Shape_shape(fix: &ShapeFix_Shape) -> UniquePtr<TopoDS_Shape>;
    }
}
