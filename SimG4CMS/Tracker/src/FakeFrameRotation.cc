#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"
#include "G4SystemOfUnits.hh"

Local3DPoint FakeFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
  return Local3DPoint(point.x()/cm,point.z()/cm,-point.y()/cm);
}
