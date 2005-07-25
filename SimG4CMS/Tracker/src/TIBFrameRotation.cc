#include "SimG4CMS/Tracker/interface/TIBFrameRotation.h"

Local3DPoint TIBFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
  return Local3DPoint(point.x()/cm,point.z()/cm,-point.y()/cm);
}
