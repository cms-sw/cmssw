#ifndef SimG4Core_StackingAction_H
#define SimG4Core_StackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserStackingAction.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

class StackingAction : public G4UserStackingAction {

public:
  StackingAction(const edm::ParameterSet & ps);
  virtual ~StackingAction();
  virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track * aTrack);
  virtual void NewStage();
  virtual void PrepareNewEvent();
private:
  void   initPointer();
  bool   isThisVolume(const G4VTouchable*, G4VPhysicalVolume* ) const;
  int    isItPrimaryDecayProductOrConversion(const G4Track*, const G4Track &) const;
  int    isItFromPrimary(const G4Track &, int) const;
private:
  G4VPhysicalVolume *tracker, *calo, *muon;
  bool   savePDandCinTracker, savePDandCinCalo, savePDandCinMuon;
  bool   killHeavy, trackNeutrino, saveFirstSecondary;
  double kmaxIon, kmaxNeutron, kmaxProton;
};

#endif
