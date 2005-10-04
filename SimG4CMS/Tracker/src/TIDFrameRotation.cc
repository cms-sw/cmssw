#include "SimG4CMS/Tracker/interface/TIDFrameRotation.h"

Local3DPoint TIDFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
  return Local3DPoint(point.x()/cm,point.y()/cm,point.z()/cm);
}
