// fieldform addition: sewing and healing, which upstream does not bind.
//
// An imported STEP solid frequently is not one: exporters leave gaps between faces that are
// within tolerance of each other but not topologically joined, so what arrives is a bag of
// faces rather than a closed shell. Sewing stitches those, which is the operation a CAD tool
// means by "heal on import" and the difference between geometry that can be cut with and
// geometry that can only be looked at.
//
// Every OCCT entry point here takes a Message_ProgressRange it defaults; the wrappers omit it
// rather than binding a progress type nothing reports.

#include <BRepBuilderAPI_Sewing.hxx>
#include <ShapeFix_Shape.hxx>
#include <TopoDS_Shape.hxx>
#include <bindings_common.hxx>

inline std::unique_ptr<BRepBuilderAPI_Sewing> Sewing_new(double tolerance) {
  return std::unique_ptr<BRepBuilderAPI_Sewing>(new BRepBuilderAPI_Sewing(tolerance));
}

inline void Sewing_add(BRepBuilderAPI_Sewing &sewing, const TopoDS_Shape &shape) { sewing.Add(shape); }

inline void Sewing_perform(BRepBuilderAPI_Sewing &sewing) { sewing.Perform(); }

inline std::unique_ptr<TopoDS_Shape> Sewing_sewed_shape(const BRepBuilderAPI_Sewing &sewing) {
  return std::unique_ptr<TopoDS_Shape>(new TopoDS_Shape(sewing.SewedShape()));
}

inline std::unique_ptr<ShapeFix_Shape> ShapeFix_Shape_new(const TopoDS_Shape &shape) {
  return std::unique_ptr<ShapeFix_Shape>(new ShapeFix_Shape(shape));
}

inline bool ShapeFix_Shape_perform(ShapeFix_Shape &fix) { return fix.Perform(); }

inline std::unique_ptr<TopoDS_Shape> ShapeFix_Shape_shape(const ShapeFix_Shape &fix) {
  return std::unique_ptr<TopoDS_Shape>(new TopoDS_Shape(fix.Shape()));
}
