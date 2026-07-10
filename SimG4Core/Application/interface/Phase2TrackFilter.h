#ifndef SimG4Core_Application_Phase2TrackFilter_H
#define SimG4Core_Application_Phase2TrackFilter_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Region.hh"
#include "G4Track.hh"
#include "G4LogicalVolume.hh"
#include "G4UserStackingAction.hh"

#include <string>
#include <vector>

class CMSSteppingVerbose;
class G4VProcess;

class Phase2TrackFilter {
public:
  explicit Phase2TrackFilter(const edm::ParameterSet& ps, const CMSSteppingVerbose*);

  ~Phase2TrackFilter() = default;

  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* aTrack);

  inline void setMother(const G4Track*);

private:
  void initPointer();

  int isItPrimaryDecayProductOrConversion(const int subtype) const;

  int isItFromPrimary(int) const;

  bool rrApplicable(const G4Track* current) const;

  bool isItOutOfTimeWindow(const G4Region*, const double&) const;

  bool isThisRegion(const G4Region*, std::vector<const G4Region*>&) const;

  void printRegions(const std::vector<const G4Region*>& reg, const std::string& word) const;

private:
  bool savePDandCinTracker, savePDandCinCalo;
  bool savePDandCinMuon, saveFirstSecondary;
  bool savePDandCinAll;
  bool killInCalo{false};
  bool killInCaloEfH{false};
  bool killHeavy, trackNeutrino, killDeltaRay;
  bool killExtra;
  bool killGamma;
  double limitEnergyForVacuum;
  double kmaxIon, kmaxNeutron, kmaxProton;
  double kmaxGamma;
  double maxTrackTime;
  double maxTrackTimeForward;
  double maxZCentralCMS;
  unsigned int numberTimes;
  std::vector<double> maxTrackTimes;
  std::vector<std::string> maxTimeNames;
  std::vector<std::string> deadRegionNames;

  std::vector<const G4Region*> maxTimeRegions;
  std::vector<const G4Region*> trackerRegions;
  std::vector<const G4Region*> muonRegions;
  std::vector<const G4Region*> caloRegions;
  std::vector<const G4Region*> lowdensRegions;
  std::vector<const G4Region*> deadRegions;

  G4VSolid* worldSolid;
  const G4Track* mother;
  const CMSSteppingVerbose* steppingVerbose;

  // Russian roulette regions
  const G4Region* regionEcal{nullptr};
  const G4Region* regionHcal{nullptr};
  const G4Region* regionMuonIron{nullptr};
  const G4Region* regionZDC{nullptr};
  const G4Region* regionHGcal{nullptr};
  const G4Region* regionWorld{nullptr};

  // Russian roulette energy limits
  double gRusRoEnerLim;
  double nRusRoEnerLim;

  // Russian roulette factors
  double gRusRoEcal;
  double nRusRoEcal;
  double gRusRoHcal;
  double nRusRoHcal;
  double gRusRoMuonIron;
  double nRusRoMuonIron;
  double gRusRoZDC;
  double nRusRoZDC;
  double gRusRoHGcal;
  double nRusRoHGcal;
  double gRusRoWorld;
  double nRusRoWorld;
  // flags
  bool gRRactive{false};
  bool nRRactive{false};
};

inline void Phase2TrackFilter::setMother(const G4Track* p) { mother = p; }

#endif
