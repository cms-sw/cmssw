#include "SimG4CMS/Tracker/interface/TrackerFrameRotation.h"
#include "G4SystemOfUnits.hh"

Local3DPoint TrackerFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
  return Local3DPoint(point.x()/cm,point.y()/cm,point.z()/cm);
}
