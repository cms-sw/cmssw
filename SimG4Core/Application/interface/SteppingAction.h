#ifndef SimG4Core_SteppingAction_H
#define SimG4Core_SteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Application/interface/SteppingTrackStatus.h"

#include "G4LogicalVolume.hh"
#include "G4Region.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Track.hh"

#include <string>
#include <vector>

class CMSSteppingVerbose;

class SteppingAction : public G4UserSteppingAction {
public:
  explicit SteppingAction(const CMSSteppingVerbose*, const edm::ParameterSet&, bool, bool);
  ~SteppingAction() override = default;

  void UserSteppingAction(const G4Step* aStep) final;

  SimActivityRegistry::G4StepSignal m_g4StepSignal;

private:
  bool initPointer();

  inline bool isInsideDeadRegion(const G4Region* reg) const;
  inline bool isOutOfTimeWindow(const G4Region* reg, const double& time) const;
  inline bool isForZDC(const G4LogicalVolume* lv, int pdg) const;

  bool isLowEnergy(const G4LogicalVolume*, const G4Track*) const;
  void PrintKilledTrack(const G4Track*, const TrackStatus&) const;

  const G4VPhysicalVolume* tracker{nullptr};
  const G4VPhysicalVolume* calo{nullptr};
  const CMSSteppingVerbose* steppingVerbose{nullptr};
  const G4LogicalVolume* m_CMStoZDC{nullptr};
  const G4Region* m_ZDCRegion{nullptr};
  double theCriticalEnergyForVacuum;
  double theCriticalDensity;
  double maxTrackTime;
  double maxTrackTimeForward;
  double maxZCentralCMS;

  unsigned int numberTimes;
  unsigned int numberEkins;
  unsigned int numberPart;
  unsigned int ndeadRegions;
  unsigned int nWarnings{0};
  G4int maxNumberOfSteps;

  bool initialized{false};
  bool killBeamPipe{false};
  bool m_CMStoZDCtransport;
  bool hasWatcher;
  bool dd4hep_;

  std::vector<double> maxTrackTimes, ekinMins;
  std::vector<std::string> maxTimeNames, ekinNames, ekinParticles;
  std::vector<std::string> deadRegionNames;
  std::vector<const G4Region*> maxTimeRegions;
  std::vector<const G4Region*> deadRegions;
  std::vector<G4LogicalVolume*> ekinVolumes;
  std::vector<int> ekinPDG;
  std::string trackerName_, caloName_, cms2ZDCName_;
};

inline bool SteppingAction::isInsideDeadRegion(const G4Region* reg) const {
  bool res = false;
  for (auto const& region : deadRegions) {
    if (reg == region) {
      res = true;
      break;
    }
  }
  return res;
}

inline bool SteppingAction::isOutOfTimeWindow(const G4Region* reg, const double& time) const {
  double tofM = maxTrackTime;
  for (unsigned int i = 0; i < numberTimes; ++i) {
    if (reg == maxTimeRegions[i]) {
      tofM = maxTrackTimes[i];
      break;
    }
  }
  return (time > tofM);
}

inline bool SteppingAction::isForZDC(const G4LogicalVolume* lv, int pdg) const {
  return (m_CMStoZDCtransport && lv == m_CMStoZDC && (pdg == 22 || pdg == 2112));
}

#endif
