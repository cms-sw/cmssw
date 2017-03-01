#ifndef SimG4Core_StackingAction_H
#define SimG4Core_StackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"

#include "G4UserStackingAction.hh"
#include "G4Region.hh"
#include "G4Track.hh"
#include "G4LogicalVolume.hh"

#include <string>
#include <vector>

class NewTrackAction;
class TrackingAction;
class CMSSteppingVerbose;

class StackingAction : public G4UserStackingAction {

public:
  explicit StackingAction(const TrackingAction*, const edm::ParameterSet & ps, 
			  const CMSSteppingVerbose*);

  virtual ~StackingAction();

  virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track * aTrack) final;

  void NewStage();
  void PrepareNewEvent();

private:

  void initPointer();

  int  isItPrimaryDecayProductOrConversion(const G4Track*, const G4Track &) const;

  int  isItFromPrimary(const G4Track &, int) const;

  bool rrApplicable(const G4Track*, const G4Track&) const;

  bool isItOutOfTimeWindow(const G4Region*, const G4Track*) const;

  bool isThisRegion(const G4Region*, std::vector<const G4Region*>&) const;

  void printRegions(const std::vector<const G4Region*>& reg, 
		    const std::string& word) const;

private:

  bool                          savePDandCinTracker, savePDandCinCalo;
  bool                          savePDandCinMuon, saveFirstSecondary;
  bool                          savePDandCinAll;
  bool                          killInCalo, killInCaloEfH;
  bool                          killHeavy, trackNeutrino, killDeltaRay;
  bool                          killExtra;
  bool                          killGamma;
  double                        limitEnergyForVacuum;
  double                        kmaxIon, kmaxNeutron, kmaxProton;
  double                        kmaxGamma;
  double                        maxTrackTime;
  unsigned int                  numberTimes;
  std::vector<double>           maxTrackTimes;
  std::vector<std::string>      maxTimeNames;
  std::vector<std::string>      deadRegionNames;

  std::vector<const G4Region*>  maxTimeRegions;
  std::vector<const G4Region*>  trackerRegions;
  std::vector<const G4Region*>  muonRegions;
  std::vector<const G4Region*>  caloRegions;
  std::vector<const G4Region*>  lowdensRegions;
  std::vector<const G4Region*>  deadRegions;

  G4VSolid*                     worldSolid;
  const TrackingAction*         trackAction;
  const CMSSteppingVerbose*     steppingVerbose;
  NewTrackAction*               newTA;
  TrackInformationExtractor     extractor;

  // Russian roulette regions
  const G4Region*               regionEcal;
  const G4Region*               regionHcal;
  const G4Region*               regionMuonIron;
  const G4Region*               regionPreShower;
  const G4Region*               regionCastor;
  const G4Region*               regionWorld;

  // Russian roulette energy limits
  double                        gRusRoEnerLim;
  double                        nRusRoEnerLim;

  // Russian roulette factors
  double                        gRusRoEcal;
  double                        nRusRoEcal;
  double                        gRusRoHcal;
  double                        nRusRoHcal;
  double                        gRusRoMuonIron;
  double                        nRusRoMuonIron;
  double                        gRusRoPreShower;
  double                        nRusRoPreShower;
  double                        gRusRoCastor;
  double                        nRusRoCastor;
  double                        gRusRoWorld;
  double                        nRusRoWorld;
  // flags
  bool                          gRRactive;
  bool                          nRRactive;
};

#endif
