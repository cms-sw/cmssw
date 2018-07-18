#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

#include "G4Step.hh"

MuonGEMFrameRotation::MuonGEMFrameRotation(const MuonDDDConstants& muonConstants) 
  : MuonFrameRotation::MuonFrameRotation() {
  g4numbering     = new MuonG4Numbering(muonConstants);
  int theLevelPart= muonConstants.getValue("level");
  theSectorLevel  = muonConstants.getValue("mg_sector")/theLevelPart;
}

MuonGEMFrameRotation::~MuonGEMFrameRotation() {
  delete g4numbering;
}

Local3DPoint MuonGEMFrameRotation::transformPoint(const Local3DPoint & point,const G4Step *) const {
  //---VI: theSectorLevel and g4numbering are not used
  return Local3DPoint(point.x(),point.z(),-point.y());
}
