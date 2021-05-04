#include "SimG4CMS/Muon/interface/MuonRPCFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

#include "G4Step.hh"

MuonRPCFrameRotation::MuonRPCFrameRotation(const MuonGeometryConstants& muonConstants,
                                           const MuonOffsetMap* offMap,
                                           bool dd4hep)
    : MuonFrameRotation::MuonFrameRotation() {
  g4numbering = new MuonG4Numbering(muonConstants, offMap, dd4hep);
  int theLevelPart = muonConstants.getValue("level");
  theRegion = muonConstants.getValue("mr_region") / theLevelPart;
}

MuonRPCFrameRotation::~MuonRPCFrameRotation() { delete g4numbering; }

Local3DPoint MuonRPCFrameRotation::transformPoint(const Local3DPoint& point, const G4Step* aStep) const {
  //check if endcap
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);
  bool endcap_muon = (num.getSuperNo(theRegion) != 1);
  return (endcap_muon) ? Local3DPoint(point.x(), point.z(), -point.y()) : Local3DPoint(point.x(), point.y(), point.z());
}
