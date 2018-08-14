#include "SimG4CMS/Muon/interface/MuonEndcapFrameRotation.h"
#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"

Local3DPoint MuonEndcapFrameRotation::transformPoint(const Local3DPoint& point,const G4Step* step) const {
      
  const G4StepPoint * preStepPoint = step->GetPreStepPoint();
  const G4TouchableHistory * theTouchable = (const G4TouchableHistory *)preStepPoint->GetTouchable();
  const G4ThreeVector& trans=theTouchable->GetTranslation();
  
  return (trans.z()<0) 
    ? Local3DPoint(-point.x(),-point.z(),-point.y())
    : Local3DPoint(point.x(),point.z(),-point.y());
} 
