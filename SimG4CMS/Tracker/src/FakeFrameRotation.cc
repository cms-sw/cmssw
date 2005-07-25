#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"

Local3DPoint FakeFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
  return Local3DPoint(point.x(),point.y(),point.z());
}
