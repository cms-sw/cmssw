#include "SimG4CMS/Muon/interface/MuonEndcapFrameRotation.h"

#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"

Local3DPoint MuonEndcapFrameRotation::transformPoint(const Local3DPoint & point,const G4Step * step) const {
  if (!step)
    return std::move(Local3DPoint(0.,0.,0.));
      
  const G4ThreeVector trans = step->GetPreStepPoint()->GetTouchable()->GetTranslation();
  
  return std::move((trans.z()<0) ? Local3DPoint(-point.x(),-point.z(),-point.y()) 
		   : Local3DPoint(point.x(),point.z(),-point.y()));
} 
