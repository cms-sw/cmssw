#ifndef SimG4Core_CustomPhysics_RHadronPythiaDecayDataManager_H
#define SimG4Core_CustomPhysics_RHadronPythiaDecayDataManager_H

#include <vector>
#include "G4Track.hh"
#include <CLHEP/Units/SystemOfUnits.h>

// Class to manage storage of R-hadron decay information between RHadronPythiaDecayer and RHDecayTracer

class RHadronPythiaDecayDataManager {
public:
  struct TrackData {
    unsigned int trackID;
    int pdgID;
    double charge;
    double px, py, pz, energy;
    double x, y, z, time;

    // Constructor to extract data from G4Track. Necessary to avoid storing G4Track pointers that may become invalid.
    TrackData()
        : trackID(0),
          pdgID(0),
          charge(0),
          px(0),
          py(0),
          pz(0),
          energy(0),
          x(0),
          y(0),
          z(0),
          time(0) {}  // Default constructor
    TrackData(const G4Track& track)
        : trackID(track.GetTrackID()),
          pdgID(track.GetDefinition()->GetPDGEncoding()),
          charge(track.GetDefinition()->GetPDGCharge()),
          px(track.GetMomentum().x() / CLHEP::GeV),
          py(track.GetMomentum().y() / CLHEP::GeV),
          pz(track.GetMomentum().z() / CLHEP::GeV),
          energy(track.GetTotalEnergy() / CLHEP::GeV),
          x(track.GetPosition().x() / CLHEP::cm),
          y(track.GetPosition().y() / CLHEP::cm),
          z(track.GetPosition().z() / CLHEP::cm),
          time(track.GetGlobalTime()) {}
  };

  RHadronPythiaDecayDataManager() : decayCounter_(0) {}
  ~RHadronPythiaDecayDataManager() = default;

  void addDecayParent(const G4Track& aTrack) {
    decayCounter_++;
    storedDecayParents_[decayCounter_] = TrackData(aTrack);
  }

  void addDecayDaughter(const G4Track& aTrack) { storedDecayDaughters_[decayCounter_].emplace_back(aTrack); }

  void getDecayInfo(std::map<int, TrackData>& decayParents, std::map<int, std::vector<TrackData>>& decayDaughters) {
    decayParents = storedDecayParents_;
    decayDaughters = storedDecayDaughters_;
  }

  void clearDecayInfo() {
    decayCounter_ = 0;
    storedDecayParents_.clear();
    storedDecayDaughters_.clear();
  }

private:
  int decayCounter_;
  std::map<int, TrackData> storedDecayParents_;
  std::map<int, std::vector<TrackData>> storedDecayDaughters_;
};

extern RHadronPythiaDecayDataManager* gRHadronPythiaDecayDataManager;

#endif
