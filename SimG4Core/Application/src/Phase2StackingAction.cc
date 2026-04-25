#include "SimG4Core/Application/interface/Phase2StackingAction.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"
#include "SimG4Core/Notification/interface/MCTruthUtil.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4TransportationManager.hh"
#include "G4VSolid.hh"

Phase2StackingAction::Phase2StackingAction(const edm::ParameterSet& p, const CMSSteppingVerbose* sv)
    : steppingVerbose(sv) {
  trackNeutrino = p.getParameter<bool>("TrackNeutrino");
  worldSolid = G4TransportationManager::GetTransportationManager()
                   ->GetNavigatorForTracking()
                   ->GetWorldVolume()
                   ->GetLogicalVolume()
                   ->GetSolid();
}

G4ClassificationOfNewTrack Phase2StackingAction::ClassifyNewTrack(const G4Track* aTrack) {
  G4ClassificationOfNewTrack classification = fUrgent;
  if (fStopAndKill == aTrack->GetTrackStatus()) {
    classification = fKill;
  }
  if (classification != fKill && worldSolid->Inside(aTrack->GetPosition()) == kOutside) {
    classification = fKill;
  }

  const int pdg = aTrack->GetDefinition()->GetPDGEncoding();
  const int abspdg = std::abs(pdg);

  if (classification != fKill && !trackNeutrino && (abspdg == 12 || abspdg == 14 || abspdg == 16 || abspdg == 18)) {
    classification = fKill;
  }

  if (classification != fKill) {
    const int pID = aTrack->GetParentID();
    auto track = const_cast<G4Track*>(aTrack);
    const G4VProcess* creatorProc = aTrack->GetCreatorProcess();
    if (creatorProc == nullptr || pID == 0) {
      MCTruthUtil::primary(track);
    } else {
      MCTruthUtil::updateSecondary(track);
    }
  }
  if (nullptr != steppingVerbose) {
    steppingVerbose->stackFilled(aTrack, (classification == fKill));
  }
  return classification;
}
