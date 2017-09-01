#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"
#include "G4SystemOfUnits.hh"

Local3DPoint FakeFrameRotation::transformPoint(const Local3DPoint & point, const G4VPhysicalVolume *) const {
  return std::move(Local3DPoint(point.x()/cm,point.z()/cm,-point.y()/cm));
}
