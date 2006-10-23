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

Local3DPoint MuonRpcFrameRotation::transformPoint(Local3DPoint & point,G4Step * aStep=0) const {
  if (aStep) {
    //check if endcap
    G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
    G4TouchableHistory * theTouchable=(G4TouchableHistory *)
                                      (preStepPoint->GetTouchable());
    const G4ThreeVector trans=theTouchable->GetTranslation();
    
    //    G4VPhysicalVolume * pv = s->GetPreStepPoint()->GetPhysicalVolume();

    //old way with strings
    //string name=v->GetName();
    //string det_name=name(1,1);
    //bool endcap_muon=(det_name == "E");
    
    //new way with base number
    MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);
    bool endcap_muon = (num.getSuperNo(theRegion)!=1);
    if (endcap_muon){
      if (trans.z()<0.) {
      //      return Local3DPoint(point.x(),point.y(),-point.z());
	return Local3DPoint(point.x(),-point.z(),point.y());
      } else {
	return Local3DPoint(point.x(),point.z(),-point.y());
      }
    } else {
      return point; 
    }
  } else {
    return Local3DPoint(0.,0.,0.);
  }
}
