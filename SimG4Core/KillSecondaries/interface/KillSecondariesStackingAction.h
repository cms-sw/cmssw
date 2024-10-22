#ifndef SimG4Core_KillSecondariesStackingAction_h
#define SimG4Core_KillSecondariesStackingAction_h

#include "G4UserStackingAction.hh"

class G4Track;

class KillSecondariesStackingAction : public G4UserStackingAction {
public:
  KillSecondariesStackingAction() {}
  ~KillSecondariesStackingAction() override = default;
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track *) override;
};

#endif
