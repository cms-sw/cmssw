#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"
#include "G4SystemOfUnits.hh"

Local3DPoint FakeFrameRotation::transformPoint(const Local3DPoint & point,const G4VPhysicalVolume*) const {
  return Local3DPoint(point.x()*invcm,point.z()*invcm,-point.y()*invcm);
}
