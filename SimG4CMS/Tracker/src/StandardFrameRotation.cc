#include "SimG4CMS/Tracker/interface/StandardFrameRotation.h"
#include "G4SystemOfUnits.hh"

Local3DPoint StandardFrameRotation::transformPoint(const Local3DPoint & point, const G4VPhysicalVolume *) const {
  return Local3DPoint(point.x()*invcm,point.y()*invcm,point.z()*invcm);
}
