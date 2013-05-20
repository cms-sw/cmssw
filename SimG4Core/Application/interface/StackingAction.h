#ifndef SimG4Core_StackingAction_H
#define SimG4Core_StackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserStackingAction.hh"
#include "G4Region.hh"
#include "G4Track.hh"
#include "G4LogicalVolume.hh"

#include <string>
#include <vector>

class StackingAction : public G4UserStackingAction {

public:
  StackingAction(const edm::ParameterSet & ps);
  virtual ~StackingAction();
  virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track * aTrack);
  virtual void NewStage();
  virtual void PrepareNewEvent();
private:
  void   initPointer();
  bool   isThisVolume(const G4VTouchable*, std::vector<G4LogicalVolume*>&) const;
  int    isItPrimaryDecayProductOrConversion(const G4Track*, const G4Track &) const;
  int    isItFromPrimary(const G4Track &, int) const;
  bool   isItLongLived(const G4Track*) const;
private:
  bool                          savePDandCinTracker, savePDandCinCalo;
  bool                          savePDandCinMuon, saveFirstSecondary;
  bool                          killInCalo, killInCaloEfH;
  bool                          killHeavy, trackNeutrino, killDeltaRay;
  double                        kmaxIon, kmaxNeutron, kmaxProton;
  double                        maxTrackTime;
  std::vector<double>           maxTrackTimes;
  std::vector<std::string>      maxTimeNames;
  std::vector<G4Region*>        maxTimeRegions;
  std::vector<G4LogicalVolume*> tracker, calo, muon;

  // Russian roulette regions
  G4Region*                     regionEcal;
  G4Region*                     regionHcal;
  G4Region*                     regionQuad;
  G4Region*                     regionMuonIron;
  G4Region*                     regionPreShower;
  G4Region*                     regionCastor;
  G4Region*                     regionBeamPipeOut;
  G4Region*                     regionWorld;

  // Russian roulette energy limits
  double                        nRusRoEnerLim;
  double                        pRusRoEnerLim;

  // Russian roulette factors
  double                        nRusRoEcal;
  double                        pRusRoEcal;
  double                        nRusRoHcal;
  double                        pRusRoHcal;
  double                        nRusRoQuad;
  double                        pRusRoQuad;
  double                        nRusRoMuonIron;
  double                        pRusRoMuonIron;
  double                        nRusRoPreShower;
  double                        pRusRoPreShower;
  double                        nRusRoCastor;
  double                        pRusRoCastor;
  double                        nRusRoBeam;
  double                        pRusRoBeam;
  double                        nRusRoWorld;
  double                        pRusRoWorld;
  // flags
  bool                          nRRactive;
  bool                          pRRactive;
};

#endif
