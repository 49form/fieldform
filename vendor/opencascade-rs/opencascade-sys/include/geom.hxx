#include <Geom_BSplineCurve.hxx>
#include <GeomConvert.hxx>
#include <Geom_BezierCurve.hxx>
#include <Geom_BezierSurface.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Surface.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <bindings_common.hxx>

inline std::unique_ptr<RustHandle_Geom_CylindricalSurface> Geom_CylindricalSurface_new(const gp_Ax3 &axis,
                                                                                       double radius) {
  return std::unique_ptr<RustHandle_Geom_CylindricalSurface>(
      new opencascade::handle<Geom_CylindricalSurface>(new Geom_CylindricalSurface(axis, radius)));
}

inline std::unique_ptr<RustHandle_Geom_Surface>
cylinder_to_surface(const RustHandle_Geom_CylindricalSurface &cylinder_handle) {
  return std::unique_ptr<RustHandle_Geom_Surface>(new opencascade::handle<Geom_Surface>(cylinder_handle));
}

inline std::unique_ptr<RustHandle_Geom_BezierSurface> Geom_BezierSurface_new(const TColgp_Array2OfPnt &poles) {
  return std::unique_ptr<RustHandle_Geom_BezierSurface>(
      new opencascade::handle<Geom_BezierSurface>(new Geom_BezierSurface(poles)));
}

inline const gp_Pnt &handle_geom_plane_location(const RustHandle_Geom_Plane &plane) { return plane->Location(); }

inline std::unique_ptr<RustHandle_Geom_BezierCurve>
Geom_BezierCurve_to_handle(std::unique_ptr<Geom_BezierCurve> bezier_curve) {
  return std::unique_ptr<RustHandle_Geom_BezierCurve>(new RustHandle_Geom_BezierCurve(bezier_curve.release()));
}

inline std::unique_ptr<RustHandle_Geom_Surface> bezier_to_surface(const RustHandle_Geom_BezierSurface &bezier_handle) {
  return std::unique_ptr<RustHandle_Geom_Surface>(new opencascade::handle<Geom_Surface>(bezier_handle));
}

inline std::unique_ptr<RustHandle_Geom_Plane>
new_HandleGeomPlane_from_HandleGeomSurface(const RustHandle_Geom_Surface &surface) {
  RustHandle_Geom_Plane plane_handle = opencascade::handle<Geom_Plane>::DownCast(surface);
  return std::unique_ptr<RustHandle_Geom_Plane>(new opencascade::handle<Geom_Plane>(plane_handle));
}

inline std::unique_ptr<gp_Pnt> HandleGeomCurve_Value(const RustHandle_Geom_Curve &curve, const Standard_Real U) {
  return std::unique_ptr<gp_Pnt>(new gp_Pnt(curve->Value(U)));
}

inline const RustHandle_Standard_Type &DynamicType(const RustHandle_Geom_Surface &surface) {
  return surface->DynamicType();
}

// The bounded piece of a curve an edge actually is, as a B-spline.
//
// **Trimmed first, and that is not a refinement.** `BRep_Tool::Curve` hands back the curve the
// edge lies *on*, which for a straight edge is a `Geom_Line` -- infinite in both directions. A
// B-spline is finite, so converting one throws a C++ exception that unwinds through Rust and
// aborts the process. The edge's own parameter range is what makes it a segment, so it belongs
// in the conversion rather than being applied afterwards.
//
// OCCT's conics survive this exactly: a circle is a rational B-spline of degree two, so the
// conversion changes representation without fitting anything.
//
// Caught rather than allowed to escape. These curves come out of files written by other
// programs, and a shape OCCT will read but not convert is a thing that happens -- a null handle
// is an edge the caller skips, where an exception is an aborted application.
inline std::unique_ptr<RustHandle_Geom_BSplineCurve>
curve_to_bspline(const RustHandle_Geom_Curve &curve, double first, double last) {
  try {
    Handle(Geom_TrimmedCurve) piece = new Geom_TrimmedCurve(curve, first, last);
    return std::unique_ptr<RustHandle_Geom_BSplineCurve>(
        new RustHandle_Geom_BSplineCurve(GeomConvert::CurveToBSplineCurve(piece)));
  } catch (...) {
    return std::unique_ptr<RustHandle_Geom_BSplineCurve>(
        new RustHandle_Geom_BSplineCurve(Handle(Geom_BSplineCurve)()));
  }
}

// The pieces a NURBS curve is made of. Indices are OCCT's, which are one-based.
inline int bspline_degree(const RustHandle_Geom_BSplineCurve &c) { return c->Degree(); }
inline int bspline_num_poles(const RustHandle_Geom_BSplineCurve &c) { return c->NbPoles(); }
inline int bspline_num_knots(const RustHandle_Geom_BSplineCurve &c) { return c->NbKnots(); }
inline bool bspline_is_rational(const RustHandle_Geom_BSplineCurve &c) { return c->IsRational(); }
inline bool bspline_is_periodic(const RustHandle_Geom_BSplineCurve &c) { return c->IsPeriodic(); }

inline std::unique_ptr<gp_Pnt> bspline_pole(const RustHandle_Geom_BSplineCurve &c, int i) {
  return std::unique_ptr<gp_Pnt>(new gp_Pnt(c->Pole(i)));
}

// One for a non-rational curve, which is what OCCT reports when no weights were stored.
inline double bspline_weight(const RustHandle_Geom_BSplineCurve &c, int i) { return c->Weight(i); }

inline double bspline_knot(const RustHandle_Geom_BSplineCurve &c, int i) { return c->Knot(i); }
inline int bspline_multiplicity(const RustHandle_Geom_BSplineCurve &c, int i) {
  return c->Multiplicity(i);
}

// The same curve in clamped form, when OCCT hands back a periodic one.
//
// A circle converted to a B-spline is periodic, and a periodic curve counts its poles and knots
// by a different convention -- the knot vector is not `poles + degree + 1` and indexing one
// against the other is wrong. `SetNotPeriodic` rewrites it to the clamped form that convention
// expects, repeating the poles the wrap used to imply. The shape is unchanged.
inline std::unique_ptr<RustHandle_Geom_BSplineCurve>
bspline_clamped(const RustHandle_Geom_BSplineCurve &c) {
  try {
    Handle(Geom_BSplineCurve) copy = Handle(Geom_BSplineCurve)::DownCast(c->Copy());
    copy->SetNotPeriodic();
    return std::unique_ptr<RustHandle_Geom_BSplineCurve>(new RustHandle_Geom_BSplineCurve(copy));
  } catch (...) {
    return std::unique_ptr<RustHandle_Geom_BSplineCurve>(
        new RustHandle_Geom_BSplineCurve(Handle(Geom_BSplineCurve)()));
  }
}
