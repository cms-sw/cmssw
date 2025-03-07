#ifndef SimDataFormats_TrackingAnalysis_SimDoublets_h
#define SimDataFormats_TrackingAnalysis_SimDoublets_h

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitFwd.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/** @brief Semi-Monte Carlo truth information used for pixel-tracking opimization.
 *
 * SimDoublets hold references to all pixel RecHits of a simulated TrackingParticle.
 * Ones those RecHits are sorted according to their position relative to the particle vertex
 * by the method sortRecHits(), you can create the true doublets of RecHits that the 
 * TrackingParticle left in the detector. These SimDoublets::Doublet objects can be used  
 * to optimize the doublet creation in the reconstruction.
 *
 * The Doublets are generated as the RecHit pairs between two consecutively hit layers.
 * I.e., if a TrackingParticle produces
 *  - 1 hit (A) in 1st layer
 *  - 2 hits (B, C) in 3rd layer
 *  - 1 hit (D) in 4th layer
 * then, the true Doublets are:
 *  (A-B), (A-C), (B-D) and (C-D).
 * So, neither does it matter that the 2nd layer got "skipped" as there are no hits,
 * nor is the Doublet of (A-D) formed since there is a layer with hits in between.
 * Doublets are not created between hits within the same layer.
 *
 * @author Jan Schulz (jan.gerrit.schulz@cern.ch)
 * @date January 2025
 */
class SimDoublets {
public:
  /**
    * Sub-class for true doublets of RecHits
    *  - first hit = inner RecHit
    *  - second hit = outer RecHit
    */
  class Doublet {
  public:
    // possible states of the doublet (could be set by an analyzer according to doublet cuts)
    enum class Status : uint8_t { undef, alive, killed };

    struct Neighbor {
      Neighbor(size_t neighborIndex) : neighborIndex_(neighborIndex), status_(Status::undef) {}

      size_t neighborIndex() const { return neighborIndex_; }
      Status status() const { return status_; }

      // methods to set status to undef, alive or killed
      void setUndef() { status_ = Status::undef; }
      void setAlive() { status_ = Status::alive; }
      void setKilled() { status_ = Status::killed; }

      // methods to check if status is undef, alive or killed
      bool isUndef() const { return status_ == Status::undef; }
      bool isAlive() const { return status_ == Status::alive; }
      bool isKilled() const { return status_ == Status::killed; }

      size_t neighborIndex_;
      Status status_;
    };

    // default constructor
    Doublet() = default;

    // constructor
    Doublet(SimDoublets const&, size_t const, size_t const, const TrackerTopology*, std::vector<size_t> const&);

    // method to get the layer pair
    std::pair<uint8_t, uint8_t> layerIds() const { return layerIds_; }

    // method to get the RecHit pair
    std::pair<SiPixelRecHitRef, SiPixelRecHitRef> recHits() const { return recHitRefs_; }

    // method to get the number of skipped layers
    int8_t numSkippedLayers() const { return numSkippedLayers_; }

    // method to get the layer pair ID
    int16_t layerPairId() const { return layerPairId_; }

    // method to get the inner layerId
    uint8_t innerLayerId() const { return layerIds_.first; }
    // method to get the outer layerId
    uint8_t outerLayerId() const { return layerIds_.second; }

    // method to get a reference to the inner RecHit
    SiPixelRecHitRef innerRecHit() const { return recHitRefs_.first; }
    // method to get a reference to the outer RecHit
    SiPixelRecHitRef outerRecHit() const { return recHitRefs_.second; }

    // method to get the global position of the inner RecHit
    GlobalPoint innerGlobalPos() const;
    // method to get the global position of the outer RecHit
    GlobalPoint outerGlobalPos() const;

    // methods to set status to undef, alive or killed
    void setUndef() { status_ = Status::undef; }
    void setAlive() { status_ = Status::alive; }
    void setKilled() { status_ = Status::killed; }

    // methods to check if status is undef, alive or killed
    bool isUndef() const { return status_ == Status::undef; }
    bool isAlive() const { return status_ == Status::alive; }
    bool isKilled() const { return status_ == Status::killed; }

    // method to get the vector of inner nieghboring doublets
    std::vector<Neighbor>& innerNeighbors() { return innerNeighbors_; }
    std::vector<Neighbor> const& innerNeighborsView() const { return innerNeighbors_; }
    // method to get the number of neighbors
    size_t numInnerNeighbors() { return innerNeighbors_.size(); }

  private:
    TrackingParticleRef trackingParticleRef_;                   // reference to the TrackingParticle
    std::pair<SiPixelRecHitRef, SiPixelRecHitRef> recHitRefs_;  // reference pair to RecHits of the Doublet
    std::pair<uint8_t, uint8_t> layerIds_;                      // pair of layer IDs corresponding to the RecHits
    Status status_;                                             // status of the doublet
    int8_t numSkippedLayers_;                                   // number of layers skipped by the Doublet
    int16_t layerPairId_;            // ID of the layer pair as defined in the reconstruction for the doublets
    GlobalVector beamSpotPosition_;  // global position of the beam spot (needed to correct the global RecHit position)
    std::vector<Neighbor> innerNeighbors_{};  // indices of inner neighboring doublets
  };

  /**
    * Sub-class for true Ntuplets of the Tracking Particle
    * - keep track of length == number of doublets
    * - first and last layer
    * - whether the Ntuplet is actually created (survives all cuts)
    */
  class Ntuplet {
  public:
    // possible states of the Ntuplet (depending on its constituents)
    enum class Status : uint8_t {
      alive,
      undefDoubletCuts,
      undefDoubletConnectionCuts,
      killedByDoubletCuts,
      killedByDoubletConnectionCuts
    };

    // default constructor
    Ntuplet() = default;

    // constructor
    Ntuplet(int const numDoublets, Status const status, uint8_t const firstLayerId, uint8_t const lastLayerId)
        : numDoublets_(numDoublets), status_(status), firstLayerId_(firstLayerId), lastLayerId_(lastLayerId){};

    // accessing the different members
    int numDoublets() const { return numDoublets_; }
    int numRecHits() const { return (numDoublets_ + 1); }
    uint8_t firstLayerId() const { return firstLayerId_; }
    uint8_t lastLayerId() const { return lastLayerId_; }

    // methods to set status to alive, undef or killed
    void setAlive() { status_ = Status::alive; }
    void setUndefDoubletCuts() { status_ = Status::undefDoubletCuts; }
    void setUndefDoubletConnectionCuts() { status_ = Status::undefDoubletConnectionCuts; }
    void setKilledByDoubletCuts() { status_ = Status::killedByDoubletCuts; }
    void setKilledByDoubletConnectionCuts() { status_ = Status::killedByDoubletConnectionCuts; }

    // methods to check if status is undef, alive or killed
    bool isAlive() const { return status_ == Status::alive; }
    bool isUndefDoubletCuts() const { return status_ == Status::undefDoubletCuts; }
    bool isUndefDoubletConnectionCuts() const { return status_ == Status::undefDoubletConnectionCuts; }
    bool isUndef() const { return isUndefDoubletCuts() || isUndefDoubletConnectionCuts(); }
    bool isKilledByDoubletCuts() const { return status_ == Status::killedByDoubletCuts; }
    bool isKilledByDoubletConnectionCuts() const { return status_ == Status::killedByDoubletConnectionCuts; }
    bool isKilled() const { return isKilledByDoubletCuts() || isKilledByDoubletConnectionCuts(); }

  private:
    int numDoublets_;       // number of doublets in the Ntuplet
    Status status_;         // status of the Ntuplet (alive, killed, etc.)
    uint8_t firstLayerId_;  // index of the first layer of the Ntuplet
    uint8_t lastLayerId_;   // index of the last layer of the Ntuplet
  };

  // default contructor
  SimDoublets() = default;

  // constructor
  SimDoublets(TrackingParticleRef const trackingParticleRef, reco::BeamSpot const& beamSpot)
      : trackingParticleRef_(trackingParticleRef), beamSpotPosition_(beamSpot.x0(), beamSpot.y0(), beamSpot.z0()) {}

  // method to add a RecHitRef with its layer
  void addRecHit(SiPixelRecHitRef const recHitRef, uint8_t const layerId) {
    recHitsAreSorted_ = false;  // set sorted-bool to false again

    // check if the layerId is not present in the layerIdVector yet
    if (std::find(layerIdVector_.begin(), layerIdVector_.end(), layerId) == layerIdVector_.end()) {
      // if it does not exist, increment number of layers
      numLayers_++;
    }

    // add recHit and layerId to the vectors
    recHitRefVector_.push_back(recHitRef);
    layerIdVector_.push_back(layerId);
  }

  // method to get the reference to the TrackingParticle
  TrackingParticleRef trackingParticle() const { return trackingParticleRef_; }

  // method to get the reference vector to the RecHits
  SiPixelRecHitRefVector recHits() const { return recHitRefVector_; }

  // method to get a reference to the RecHit at index i
  SiPixelRecHitRef recHits(size_t i) const { return recHitRefVector_[i]; }

  // method to get the layer id vector
  std::vector<uint8_t> layerIds() const { return layerIdVector_; }

  // method to get the layer id at index i
  uint8_t layerIds(size_t i) const { return layerIdVector_[i]; }

  // method to get the beam spot position
  GlobalVector beamSpotPosition() const { return beamSpotPosition_; }

  // method to get the number of layers
  int numLayers() const { return numLayers_; }

  // method to get number of RecHits in the SimDoublets
  int numRecHits() const { return layerIdVector_.size(); }

  int numDoublets() const { return doublets_.size(); }

  // method to sort the RecHits according to the position
  void sortRecHits();

  // method to produce the SimDoublets from the RecHits
  void buildSimDoublets(const TrackerTopology* trackerTopology = nullptr) const;

  // method to access the SimDoublets
  std::vector<Doublet>& getSimDoublets() const { return doublets_; }
  std::vector<Doublet>& getSimDoublets(const TrackerTopology* trackerTopology) const {
    buildSimDoublets(trackerTopology);
    return doublets_;
  }

  // method to produce and get the SimNtuplets
  // (collection of all possible Ntuplets you can build from the SimDoublets)
  std::vector<Ntuplet> getSimNtuplets() const;

private:
  // function for recursive building of Ntuplets
  void buildSimNtuplets(std::vector<Ntuplet>& simNtuplets,
                        Doublet const& doublet,
                        size_t numSimDoublets,
                        size_t lastLayerId,
                        Ntuplet::Status status) const;
  // overload the function to deal with the starting configuration
  void buildSimNtuplets(std::vector<Ntuplet>& simNtuplets, Doublet const& doublet) const {
    buildSimNtuplets(simNtuplets, doublet, 0, doublet.outerLayerId(), Ntuplet::Status::alive);
  }

  // class members
  TrackingParticleRef trackingParticleRef_;  // reference to the TrackingParticle
  SiPixelRecHitRefVector recHitRefVector_;   // reference vector to RecHits associated to the TP (sorted after building)
  std::vector<uint8_t> layerIdVector_;       // vector of layer IDs corresponding to the RecHits
  GlobalVector beamSpotPosition_;  // global position of the beam spot (needed to correct the global RecHit position)
  bool recHitsAreSorted_{false};   // true if RecHits were sorted
  int numLayers_{0};               // number of layers hit by the TrackingParticle

  // non-persistent, mutable members:
  // vector of true doublets
  mutable std::vector<Doublet> doublets_{};
};

// collection of SimDoublets
typedef std::vector<SimDoublets> SimDoubletsCollection;

#endif
