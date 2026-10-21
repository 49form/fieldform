#include <BRepMesh_IncrementalMesh.hxx>
#include <bindings_common.hxx>

// The angular criterion as well as the linear one.
//
// `BRepMesh_IncrementalMesh(shape, deflection)` leaves the angular deflection at OCCT's
// default of 0.5 rad — about 28 degrees between neighbouring facet normals. Chord height
// alone does not bound that: it bounds how far a facet sits from the surface, and on a small
// radius a tight chord still turns a large angle. Shading is the normal, so that default is
// what a user sees as faceting on curved faces.
inline std::unique_ptr<BRepMesh_IncrementalMesh> IncrementalMesh_new_angular(
    const TopoDS_Shape &shape, double deflection, double angular_deflection, bool parallel) {
  return std::unique_ptr<BRepMesh_IncrementalMesh>(new BRepMesh_IncrementalMesh(
      shape, deflection, Standard_False, angular_deflection, parallel));
}
