#include "TrackPropagation/NavGeometry/interface/NavSurfaceBuilder.h"
#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Cone.h"
#include "TrackPropagation/NavGeometry/interface/NavPlane.h"
#include "TrackPropagation/NavGeometry/interface/NavCylinder.h"
#include "TrackPropagation/NavGeometry/interface/NavCone.h"

NavSurface* NavSurfaceBuilder::build( const Surface& surface) const
{
  const Plane* plane = dynamic_cast<const Plane*>(&surface);
  if (plane != 0) {
    return new NavPlane( plane);
  }
    
  const Cylinder* cylinder = dynamic_cast<const Cylinder*>(&surface);
  if (cylinder != 0) {
    return new NavCylinder( cylinder);
  }
    
  const Cone* cone = dynamic_cast<const Cone*>(&surface);
  if (cone != 0) {
    return new NavCone( cone);
  }
    
  return 0;
}
