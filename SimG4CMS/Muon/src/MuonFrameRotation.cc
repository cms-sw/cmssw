#include "G4Step.hh"
#include "SimG4CMS/Muon/interface/MuonFrameRotation.h"

Local3DPoint MuonFrameRotation::transformPoint(const Local3DPoint &point,
                                               const G4Step *) const {
  return point;
}
