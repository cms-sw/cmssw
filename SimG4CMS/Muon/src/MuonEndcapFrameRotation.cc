#include "SimG4CMS/Muon/interface/MuonEndcapFrameRotation.h"

#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"

Local3DPoint MuonEndcapFrameRotation::transformPoint(const Local3DPoint & point,const G4Step * s=0) const {
  if (!s)
    return Local3DPoint(0.,0.,0.);
      
  const G4StepPoint * preStepPoint = s->GetPreStepPoint();
  const G4TouchableHistory * theTouchable = (const G4TouchableHistory *)preStepPoint->GetTouchable();
  const G4ThreeVector trans=theTouchable->GetTranslation();
  
  if (trans.z()<0) {
    //      return Local3DPoint(point.x(),-point.z(),point.y());
    //      return Local3DPoint(-point.x(),point.z(),-point.y());
    return Local3DPoint(-point.x(),-point.z(),-point.y());
  } else {
    //      return Local3DPoint(point.x(),point.z(),-point.y());
    return Local3DPoint(point.x(),point.z(),-point.y());
  }
} 
