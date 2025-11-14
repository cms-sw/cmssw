#ifndef RHadronPythiaDecayDataManager_H
#define RHadronPythiaDecayDataManager_H

#include <vector>
#include <mutex>
#include "G4Track.hh"

// Class to manage storage of R-hadron decay information between RHadronPythiaDecayer and RHDecayTracer

class RHadronPythiaDecayDataManager {
public:
    struct TrackData {
        int trackID;
        int pdgID;
        double px, py, pz, energy;
        double x, y, z, time;
        
        // Constructor to extract data from G4Track. Necessary to avoid storing G4Track pointers that may become invalid.
        TrackData() : trackID(0), pdgID(0), px(0), py(0), pz(0), energy(0), x(0), y(0), z(0), time(0) {} // Default constructor
        TrackData(const G4Track& track) 
            : trackID(track.GetTrackID()),
              pdgID(track.GetDefinition()->GetPDGEncoding()),
              px(track.GetMomentum().x()),
              py(track.GetMomentum().y()),
              pz(track.GetMomentum().z()),
              energy(track.GetTotalEnergy()),
              x(track.GetPosition().x()),
              y(track.GetPosition().y()),
              z(track.GetPosition().z()),
              time(track.GetGlobalTime()) {}
    };

    static RHadronPythiaDecayDataManager& getInstance() {
        static RHadronPythiaDecayDataManager instance;
        return instance;
    }
    
    void addDecayParent(const G4Track& aTrack) {
        std::lock_guard<std::mutex> lock(dataMutex_);
        decayCounter_++;
        storedDecayParents_[decayCounter_] = TrackData(aTrack);
    }

    void addDecayDaughter(const G4Track& aTrack) {
        std::lock_guard<std::mutex> lock(dataMutex_);
        storedDecayDaughters_[decayCounter_].emplace_back(aTrack);
    }

    void getDecayInfo(std::map<int, TrackData>& decayParents, std::map<int, std::vector<TrackData>>& decayDaughters) {
        std::lock_guard<std::mutex> lock(dataMutex_);
        decayParents = storedDecayParents_;
        decayDaughters = storedDecayDaughters_;
    }

    void clearDecayInfo() {
        std::lock_guard<std::mutex> lock(dataMutex_);
        decayCounter_ = 0;
        storedDecayParents_.clear();
        storedDecayDaughters_.clear();
    }

private:
    RHadronPythiaDecayDataManager() {}
    std::mutex dataMutex_;
    int decayCounter_ = 0;
    std::map<int, TrackData> storedDecayParents_;
    std::map<int, std::vector<TrackData>> storedDecayDaughters_;
};

#endif