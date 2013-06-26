#include "SimG4Core/KillSecondaries/interface/KillSecondariesStackingAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"

#include "G4Track.hh"

G4ClassificationOfNewTrack KillSecondariesStackingAction::ClassifyNewTrack(const G4Track * aTrack) {

  NewTrackAction newTA;
  if (aTrack->GetCreatorProcess()==0 || aTrack->GetParentID()==0) {
    newTA.primary(aTrack);
    return fUrgent;
  } else {
    const G4Track * mother = CurrentG4Track::track();
    newTA.secondary(aTrack, *mother, 0);
    return fKill;
  }

}









































































































