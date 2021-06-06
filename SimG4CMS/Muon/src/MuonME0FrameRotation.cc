#include "SimG4CMS/Muon/interface/MuonME0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"

#include "G4Step.hh"

MuonME0FrameRotation::MuonME0FrameRotation(const MuonGeometryConstants&) : MuonFrameRotation::MuonFrameRotation() {}

MuonME0FrameRotation::~MuonME0FrameRotation() {}

Local3DPoint MuonME0FrameRotation::transformPoint(const Local3DPoint& point, const G4Step*) const {
  return Local3DPoint(point.x(), point.z(), -point.y());
}
