#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"

//#define LOCAL_DEBUG

MuonGEMFrameRotation::MuonGEMFrameRotation(const DDCompactView& cpv) : MuonFrameRotation::MuonFrameRotation(cpv) {
  g4numbering     = new MuonG4Numbering(cpv);
  MuonDDDConstants muonConstants(cpv);
  int theLevelPart= muonConstants.getValue("level");
  theSectorLevel  = muonConstants.getValue("mg_sector")/theLevelPart;
#ifdef LOCAL_DEBUG
  std::cout << "MuonGEMFrameRotation: theSectorLevel " << theSectorLevel 
	    << std::endl;
#endif
}

MuonGEMFrameRotation::~MuonGEMFrameRotation() {
  delete g4numbering;
}

Local3DPoint MuonGEMFrameRotation::transformPoint(const Local3DPoint & point,const G4Step * aStep=0) const {
  if (!aStep) return Local3DPoint(0.,0.,0.);  

  return Local3DPoint(point.x(),point.z(),-point.y());
}
