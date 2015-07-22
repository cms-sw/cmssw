#include "SimG4CMS/Muon/interface/MuonME0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"

//#define LOCAL_DEBUG

MuonME0FrameRotation::MuonME0FrameRotation(const DDCompactView& cpv) : MuonFrameRotation::MuonFrameRotation(cpv) {
  g4numbering     = new MuonG4Numbering(cpv);
  MuonDDDConstants muonConstants(cpv);
  int theLevelPart= muonConstants.getValue("level");
  theSectorLevel  = muonConstants.getValue("mg_sector")/theLevelPart;
#ifdef LOCAL_DEBUG
  std::cout << "MuonME0FrameRotation: theSectorLevel " << theSectorLevel 
	    << std::endl;
#endif
  edm::LogVerbatim("MuonME0FrameRotation")<<"MuonME0FrameRotation: theSectorLevel " << theSectorLevel;
}

MuonME0FrameRotation::~MuonME0FrameRotation() {
  delete g4numbering;
}

Local3DPoint MuonME0FrameRotation::transformPoint(const Local3DPoint & point,const G4Step * aStep=0) const {
  if (!aStep) return Local3DPoint(0.,0.,0.);  

  edm::LogVerbatim("MuonME0FrameRotation")<<"MuonME0FrameRotation transformPoint :: Local3DPoint (" <<point.x()<<","<<point.z()<<","<<-point.y()<<")" ;
  return Local3DPoint(point.x(),point.z(),-point.y());
}
