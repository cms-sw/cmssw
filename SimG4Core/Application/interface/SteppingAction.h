#ifndef SimG4Core_SteppingAction_H
#define SimG4Core_SteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4LogicalVolume.hh"
#include "G4Region.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Track.hh"

#include <string>
#include <vector>

class EventAction;
class CMSSteppingVerbose;

enum TrackStatus {
  sAlive = 0,
  sKilledByProcess = 1,
  sDeadRegion = 2,
  sOutOfTime = 3,
  sLowEnergy = 4,
  sLowEnergyInVacuum = 5,
  sEnergyDepNaN = 6,
  sVeryForward = 7
};

class SteppingAction : public G4UserSteppingAction {
public:
  explicit SteppingAction(EventAction* ea, const edm::ParameterSet& ps, const CMSSteppingVerbose*, bool hasW);
  ~SteppingAction() override;

  void UserSteppingAction(const G4Step* aStep) final;

  SimActivityRegistry::G4StepSignal m_g4StepSignal;

private:
  bool initPointer();

  inline bool isInsideDeadRegion(const G4Region* reg) const;
  inline bool isOutOfTimeWindow(const G4Track* theTrack, const G4Region* reg) const;
  inline bool isThisVolume(const G4VTouchable* touch, const G4VPhysicalVolume* pv) const;

  bool isLowEnergy(const G4Step* aStep) const;
  void PrintKilledTrack(const G4Track*, const TrackStatus&) const;

  EventAction* eventAction_;
  const G4VPhysicalVolume *tracker, *calo;
  const CMSSteppingVerbose* steppingVerbose;
  double theCriticalEnergyForVacuum;
  double theCriticalDensity;
  double maxTrackTime;
  double maxTrackTimeForward;
  double maxZCentralCMS;
  std::vector<double> maxTrackTimes, ekinMins;
  std::vector<std::string> maxTimeNames, ekinNames, ekinParticles;
  std::vector<std::string> deadRegionNames;
  std::vector<const G4Region*> maxTimeRegions;
  std::vector<const G4Region*> deadRegions;
  std::vector<G4LogicalVolume*> ekinVolumes;
  std::vector<int> ekinPDG;
  unsigned int numberTimes;
  unsigned int numberEkins;
  unsigned int numberPart;
  unsigned int ndeadRegions;
  unsigned int nWarnings;

  bool initialized;
  bool killBeamPipe;
  bool hasWatcher;
};

inline bool SteppingAction::isInsideDeadRegion(const G4Region* reg) const {
  bool res = false;
  for (auto& region : deadRegions) {
    if (reg == region) {
      res = true;
      break;
    }
  }
  return res;
}

inline bool SteppingAction::isOutOfTimeWindow(const G4Track* theTrack, const G4Region* reg) const {
  double tofM = maxTrackTime;
  for (unsigned int i = 0; i < numberTimes; ++i) {
    if (reg == maxTimeRegions[i]) {
      tofM = maxTrackTimes[i];
      break;
    }
  }
  return (theTrack->GetGlobalTime() > tofM);
}

inline bool SteppingAction::isThisVolume(const G4VTouchable* touch, const G4VPhysicalVolume* pv) const {
  int level = (touch->GetHistoryDepth()) + 1;
  return (level >= 3) ? (touch->GetVolume(level - 3) == pv) : false;
}

#endif
