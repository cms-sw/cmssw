#ifndef SimG4Core_Application_Phase2StackingAction_H
#define SimG4Core_Application_Phase2StackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserStackingAction.hh"
#include "G4Track.hh"

class CMSSteppingVerbose;
class G4VSolid;

class Phase2StackingAction : public G4UserStackingAction {
public:
  Phase2StackingAction(const edm::ParameterSet& ps, const CMSSteppingVerbose*);

  ~Phase2StackingAction() override = default;

  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* aTrack) override;

private:
  const CMSSteppingVerbose* steppingVerbose;
  G4VSolid* worldSolid;
  bool trackNeutrino;
};

#endif
