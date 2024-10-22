#include "SimG4Core/KillSecondaries/interface/KillSecondariesStackingAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/MCTruthUtil.h"

#include "G4Track.hh"

G4ClassificationOfNewTrack KillSecondariesStackingAction::ClassifyNewTrack(const G4Track *aTrack) {
  auto track = const_cast<G4Track *>(aTrack);
  if (aTrack->GetCreatorProcess() == nullptr || aTrack->GetParentID() == 0) {
    MCTruthUtil::primary(track);
    return fUrgent;
  } else {
    const G4Track *mother = CurrentG4Track::track();
    MCTruthUtil::secondary(track, *mother, 0);
    return fKill;
  }
}
