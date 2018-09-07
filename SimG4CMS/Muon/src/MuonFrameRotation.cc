#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"
#include "G4Step.hh"

Local3DPoint MuonFrameRotation::transformPoint(const Local3DPoint& point,const G4Step*) const {
  return point;
} 
