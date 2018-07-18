#include "SimG4CMS/Muon/interface/MuonME0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"

MuonME0FrameRotation::MuonME0FrameRotation(const MuonDDDConstants& muonConstants) 
  : MuonFrameRotation::MuonFrameRotation() {
  g4numbering     = new MuonG4Numbering(muonConstants);
  int theLevelPart= muonConstants.getValue("level");
  theSectorLevel  = muonConstants.getValue("mg_sector")/theLevelPart;
  edm::LogVerbatim("MuonME0FrameRotation")
    <<"MuonME0FrameRotation: theSectorLevel " << theSectorLevel;
}

MuonME0FrameRotation::~MuonME0FrameRotation() {
  delete g4numbering;
}

Local3DPoint 
MuonME0FrameRotation::transformPoint(const Local3DPoint& point,const G4Step *) const {
  //---VI: sector level and g4numbering are not used
  return Local3DPoint(point.x(),point.z(),-point.y());
}
