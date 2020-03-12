#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include "G4Step.hh"

MuonGEMFrameRotation::MuonGEMFrameRotation(const MuonDDDConstants&) : MuonFrameRotation::MuonFrameRotation() {}

MuonGEMFrameRotation::~MuonGEMFrameRotation() {}

Local3DPoint MuonGEMFrameRotation::transformPoint(const Local3DPoint& point, const G4Step*) const {
  return Local3DPoint(point.x(), point.z(), -point.y());
}
