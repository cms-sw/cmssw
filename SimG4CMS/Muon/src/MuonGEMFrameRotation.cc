#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"

#include "G4Step.hh"

MuonGEMFrameRotation::MuonGEMFrameRotation(const MuonGeometryConstants&) : MuonFrameRotation::MuonFrameRotation() {}

MuonGEMFrameRotation::~MuonGEMFrameRotation() {}

Local3DPoint MuonGEMFrameRotation::transformPoint(const Local3DPoint& point, const G4Step*) const {
  return Local3DPoint(point.x(), point.z(), -point.y());
}
