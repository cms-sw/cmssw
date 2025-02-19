#include "TrackPropagation/Geant4e/interface/Geant4eSteppingAction.h"
#include "G4Step.hh"

void Geant4eSteppingAction::UserSteppingAction(const G4Step* step) {
  theTrackLength += step->GetStepLength();
}
