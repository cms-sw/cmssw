#include "SimG4CMS/Tracker/interface/TrackerFrameRotation.h"
#include "G4SystemOfUnits.hh"

Local3DPoint TrackerFrameRotation::transformPoint(const Local3DPoint & point,const G4VPhysicalVolume*) const {
  return Local3DPoint(point.x()/cm,point.y()/cm,point.z()/cm);
}
