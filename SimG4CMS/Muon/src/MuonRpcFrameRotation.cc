#include "SimG4CMS/Muon/interface/MuonRpcFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"

MuonRpcFrameRotation::MuonRpcFrameRotation(const DDCompactView& cpv) : MuonFrameRotation::MuonFrameRotation(cpv) {
  g4numbering = new MuonG4Numbering(cpv);
  MuonDDDConstants muonConstants(cpv);
  int theLevelPart=muonConstants.getValue("level");
  theRegion=muonConstants.getValue("mr_region")/theLevelPart;
}

MuonRpcFrameRotation::~MuonRpcFrameRotation(){
  delete g4numbering;
}

Local3DPoint MuonRpcFrameRotation::transformPoint(const Local3DPoint & point,const G4Step * aStep=0) const {
  if (!aStep)
    return Local3DPoint(0.,0.,0.);  

  //check if endcap
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);
  bool endcap_muon = (num.getSuperNo(theRegion)!=1);
  if (endcap_muon){
    return Local3DPoint(point.x(),point.z(),-point.y());
  } else {
    return point; 
  }
}
