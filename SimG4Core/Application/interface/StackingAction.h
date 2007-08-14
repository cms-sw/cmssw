#ifndef SimG4Core_StackingAction_H
#define SimG4Core_StackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserStackingAction.hh"
#include "G4Track.hh"

class StackingAction : public G4UserStackingAction {

public:
  StackingAction(const edm::ParameterSet & ps);
  virtual ~StackingAction();
  virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track * aTrack);
  virtual void NewStage();
  virtual void PrepareNewEvent();
private:
  bool   savePrimaryDecayProductsAndConversions, suppressHeavy;
  double pmaxIon, pmaxNeutron, pmaxProton;
};

#endif
