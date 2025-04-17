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
    enum class Status : uint8_t { undef, alive, killedByCuts, killedByMissingLayerPair };

    struct Neighbor {
      Neighbor(size_t index) : index_(index), status_(Status::undef) {}

      size_t index() const { return index_; }

      // methods to set status to undef, alive or killed
      void setUndef() { status_ = Status::undef; }
      void setAlive() { status_ = Status::alive; }
      void setKilled() { status_ = Status::killedByCuts; }

      // methods to check if status is undef, alive or killed
      bool isUndef() const { return status_ == Status::undef; }
      bool isAlive() const { return status_ == Status::alive; }
      bool isKilled() const { return status_ == Status::killedByCuts; }

      size_t index_;   // index of the neighboring doublet
      Status status_;  // status of the connection to the neighboring doublet
    };

    // default constructor
    Doublet() = default;

    // constructor
    Doublet(SimDoublets const&, size_t const, size_t const, const TrackerTopology*, std::vector<size_t> const&);

    // method to get the layer pair
    std::pair<uint8_t, uint8_t> layerIds() const { return layerIds_; }

    // method to get the number of skipped layers
    int8_t numSkippedLayers() const { return numSkippedLayers_; }

    // method to get the layer pair ID
    int16_t layerPairId() const { return layerPairId_; }

    // methods to get the inner/outer layerId
    uint8_t innerLayerId() const { return layerIds_.first; }
    uint8_t outerLayerId() const { return layerIds_.second; }

    // methods to get the cluster size of the inner/outer RecHit
    int16_t innerClusterYSize() const { return clusterYSizes_.first; }
    int16_t outerClusterYSize() const { return clusterYSizes_.second; }

    // methods to get the module ids of the inner/outer RecHit
    unsigned int innerModuleId() const { return moduleIds_.first; }
    unsigned int outerModuleId() const { return moduleIds_.second; }

    // methods to get the global position of the inner/outer RecHit
    GlobalPoint innerGlobalPos() const { return globalPositions_.first; };
    GlobalPoint outerGlobalPos() const { return globalPositions_.second; };

    // methods to set status to undef, alive or killed
    void setUndef() { status_ = Status::undef; }
    void setAlive() { status_ = Status::alive; }
    void setKilledByCuts() { status_ = Status::killedByCuts; }
    void setKilledByMissingLayerPair() { status_ = Status::killedByMissingLayerPair; }

    // methods to check if status is undef, alive or killed
    bool isUndef() const { return status_ == Status::undef; }
    bool isAlive() const { return status_ == Status::alive; }
    bool isKilledByCuts() const { return status_ == Status::killedByCuts; }
    bool isKilledByMissingLayerPair() const { return status_ == Status::killedByMissingLayerPair; }
    bool isKilled() const { return isKilledByCuts() || isKilledByMissingLayerPair(); }

    // methods to get the vector of inner neighboring doublets
    std::vector<Neighbor>& innerNeighbors() { return innerNeighbors_; }
    std::vector<Neighbor> const& innerNeighborsView() const { return innerNeighbors_; }
    int innerNeighborIndex(int i) const { return innerNeighbors_.at(i).index(); }
    // method to get the number of neighbors
    int numInnerNeighbors() const { return innerNeighbors_.size(); }
    // method to get the inner layer ID of the neighbors
    uint8_t innerNeighborsInnerLayerId() const { return innerNeighborsInnerLayerId_; }

  private:
    std::pair<int, int> moduleIds_;                        // module Ids of the RecHits of the Doublet
    std::pair<GlobalPoint, GlobalPoint> globalPositions_;  // global position of the RecHits of the Doublet
                                                           // (corrected by beamspot)
    std::pair<uint8_t, uint8_t> layerIds_;                 // pair of layer IDs corresponding to the RecHits
    std::pair<int16_t, int16_t> clusterYSizes_;            // pair of cluster sizes corresponding to the RecHits
    Status status_;                                        // status of the doublet
    int8_t numSkippedLayers_;                              // number of layers skipped by the Doublet
    int16_t layerPairId_;                     // ID of the layer pair as defined in the reconstruction for the doublets
    std::vector<Neighbor> innerNeighbors_{};  // indices of inner neighboring doublets and the status of the connection
    uint8_t innerNeighborsInnerLayerId_{99};  // layer ID of the inner RecHit of the inner neighboring doublets
  };

  /**
    * Sub-class for true Ntuplets of the Tracking Particle
    * - keep track of length == number of doublets
    * - first and last layer
    * - whether the Ntuplet is actually created (survives all cuts)
    */
  class Ntuplet {
  public:
    // flags indicating qualities of Ntuplet (depending on its constituents)
    // The order is chosen in such a way that a smaller status value means that the Ntuplet get farther
    // in the reconstruction chain. Hence, a value of 0 corresponds to the Ntuplet surviving reconstruction.
    enum class StatusBit : uint8_t {
      hasMissingLayerPair = 1,
      hasUndefDoubletCuts = 1 << 1,
      hasKilledDoublets = 1 << 2,
      hasUndefDoubletConnectionCuts = 1 << 3,
      hasKilledConnections = 1 << 4,
      isTooShort = 1 << 5
    };

    // default constructor
    Ntuplet() = default;

    // constructor
    Ntuplet(uint8_t const numDoublets, uint8_t const status, uint8_t const firstLayerId, uint8_t const lastLayerId)
        : numDoublets_(numDoublets), status_(status), firstLayerId_(firstLayerId), lastLayerId_(lastLayerId){};

    // accessing the different members
    uint8_t numDoublets() const { return numDoublets_; }
    uint8_t numRecHits() const { return (numDoublets_ + 1); }
    uint8_t firstLayerId() const { return firstLayerId_; }
    uint8_t lastLayerId() const { return lastLayerId_; }

    // method to update an external status
    static uint8_t updateStatus(uint8_t status,
                                bool const hasUndefDoubletCuts,
                                bool const hasMissingLayerPair,
                                bool const hasKilledDoublets,
                                bool const hasUndefDoubletConnectionCuts,
                                bool const hasKilledConnections,
                                bool const isTooShort = false) {
      return status | (uint8_t(hasUndefDoubletCuts) * uint8_t(StatusBit::hasUndefDoubletCuts) +
                       uint8_t(hasMissingLayerPair) * uint8_t(StatusBit::hasMissingLayerPair) +
                       uint8_t(hasKilledDoublets) * uint8_t(StatusBit::hasKilledDoublets) +
                       uint8_t(hasUndefDoubletConnectionCuts) * uint8_t(StatusBit::hasUndefDoubletConnectionCuts) +
                       uint8_t(hasKilledConnections) * uint8_t(StatusBit::hasKilledConnections) +
                       uint8_t(isTooShort) * uint8_t(StatusBit::isTooShort));
    }

    // methods to set status to alive, undef or killed
    void setUndefDoubletCuts() { status_ |= uint8_t(StatusBit::hasUndefDoubletCuts); }
    void setUndefDoubletConnectionCuts() { status_ |= uint8_t(StatusBit::hasUndefDoubletConnectionCuts); }
    void setMissingLayerPair() { status_ |= uint8_t(StatusBit::hasMissingLayerPair); }
    void setKilledDoublets() { status_ |= uint8_t(StatusBit::hasKilledDoublets); }
    void setKilledConnections() { status_ |= uint8_t(StatusBit::hasKilledConnections); }
    void setTooShort() { status_ |= uint8_t(StatusBit::isTooShort); }

    // methods to check if status is undef, alive or killed
    bool hasUndefDoubletCuts() const { return status_ & uint8_t(StatusBit::hasUndefDoubletCuts); }
    bool hasUndefDoubletConnectionCuts() const { return status_ & uint8_t(StatusBit::hasUndefDoubletConnectionCuts); }
    bool hasUndef() const { return hasUndefDoubletCuts() || hasUndefDoubletConnectionCuts(); }
    bool hasMissingLayerPair() const { return status_ & uint8_t(StatusBit::hasMissingLayerPair); }
    bool hasKilledDoublets() const { return status_ & uint8_t(StatusBit::hasKilledDoublets); }
    bool hasKilledConnections() const { return status_ & uint8_t(StatusBit::hasKilledConnections); }
    bool isKilled() const { return hasMissingLayerPair() || hasKilledDoublets() || hasKilledConnections(); }
    bool isTooShort() const { return status_ & uint8_t(StatusBit::isTooShort); }
    bool isAlive() const { return !(status_); }  // if nothing is set (no undef and no kills) the tuplet is alive

    // method to compare the own status to a given one and check which one gets farther in the reconstruction chain
    bool getsFartherInRecoChainThanReference(uint8_t const referenceStatusBit) const {
      return status_ < referenceStatusBit;
    }
    bool getsFartherInRecoChainThanReference(Ntuplet const& referenceNtuplet) const {
      return status_ < referenceNtuplet.status_;
    }

  private:
    uint8_t numDoublets_;   // number of doublets in the Ntuplet
    uint8_t status_;        // status flags of the Ntuplet (missing layer pairs, undefined cuts, killed doublets, etc.)
    uint8_t firstLayerId_;  // index of the first layer of the Ntuplet
    uint8_t lastLayerId_;   // index of the last layer of the Ntuplet
  };

  // default contructor
  SimDoublets() = default;

  // constructor
  SimDoublets(TrackingParticleRef const trackingParticleRef, reco::BeamSpot const& beamSpot)
      : trackingParticleRef_(trackingParticleRef), beamSpotPosition_(beamSpot.x0(), beamSpot.y0(), beamSpot.z0()) {}

  // method to add a RecHit to the SimPixelTrack
  void addRecHit(BaseTrackerRecHit const& recHit,
                 uint8_t const layerId,
                 int16_t const clusterYSize,
                 unsigned int const detId,
                 int const moduleId);

  // method to get the reference to the TrackingParticle
  TrackingParticleRef trackingParticle() const { return trackingParticleRef_; }

  // method to get the detector id vector
  std::vector<unsigned int> detIds() const { return detIdVector_; }
  // method to get the detector id at index i
  unsigned int detIds(size_t const i) const { return detIdVector_[i]; }

  // method to get the module id vector
  std::vector<int> moduleIds() const { return moduleIdVector_; }
  // method to get the module id at index i
  int moduleIds(size_t const i) const { return moduleIdVector_[i]; }

  // method to get the global position vector of the RecHits
  std::vector<GlobalPoint> globalPositions() const { return globalPositionVector_; }
  // method to get the global position of the RecHit at index i
  GlobalPoint globalPositions(size_t const i) const { return globalPositionVector_[i]; }

  // method to get the layer id vector
  std::vector<uint8_t> layerIds() const { return layerIdVector_; }
  // method to get the layer id at index i
  uint8_t layerIds(size_t const i) const { return layerIdVector_[i]; }

  // method to get the cluster size vector
  std::vector<int16_t> clusterYSizes() const { return clusterYSizeVector_; }
  // method to get the cluster size at index i
  int16_t clusterYSizes(size_t const i) const { return clusterYSizeVector_[i]; }

  // method to get the beam spot position
  GlobalVector beamSpotPosition() const { return beamSpotPosition_; }

  // method to get the number of layers
  int numLayers() const { return numLayers_; }
  // method to get number of RecHits in the SimDoublets
  int numRecHits() const { return layerIdVector_.size(); }
  // method to get the number of SimDoublets
  int numDoublets() const { return doublets_.size(); }

  // method to sort the RecHits according to the position
  void sortRecHits();

  // method to produce the SimDoublets from the RecHits
  void buildSimDoublets(const TrackerTopology* trackerTopology) const;
  // method to access the SimDoublets
  std::vector<Doublet>& getSimDoublets() const { return doublets_; }
  // method to build and access the SimDoublets
  std::vector<Doublet>& buildAndGetSimDoublets(const TrackerTopology* trackerTopology) const {
    buildSimDoublets(trackerTopology);
    return doublets_;
  }
  // method to access a single SimDoublet
  Doublet const& getSimDoublet(int const index) const { return doublets_.at(index); }

  // method to build the SimNtuplets
  // minNumDoubletsToPass = the number of doublets required for the Ntuplet to not be considered too short
  void buildSimNtuplets(size_t const minNumDoubletsToPass = 0) const;
  // method to access the SimNtuplets
  std::vector<Ntuplet>& getSimNtuplets() const { return ntuplets_; };
  // method to build and access the SimNtuplets in one go
  // minNumDoubletsToPass = the number of doublets required for the Ntuplet to not be considered too short
  std::vector<Ntuplet>& buildAndGetSimNtuplets(size_t const minNumDoubletsToPass = 0) const {
    buildSimNtuplets(minNumDoubletsToPass);
    return ntuplets_;
  };

  // method to check if there are SimNtuplets
  bool hasSimNtuplet() const { return longestNtupletIndex_ != -1; }
  // method to check if there are alive SimNtuplet
  bool hasAliveSimNtuplet() const { return longestAliveNtupletIndex_ != -1; }

  // method to access the longest SimNtuplet
  Ntuplet const& longestSimNtuplet() const { return ntuplets_.at(longestNtupletIndex_); }
  // method to access the longest alive SimNtuplet
  Ntuplet const& longestAliveSimNtuplet() const { return ntuplets_.at(longestAliveNtupletIndex_); }

  // method to clear the mutable vectors once you finished using them
  void clearMutables() const {
    doublets_.clear();
    ntuplets_.clear();
  }

private:
  // function for recursive building of Ntuplets
  void buildSimNtuplets(Doublet const& doublet,
                        size_t numSimDoublets,
                        size_t lastLayerId,
                        uint8_t status,
                        size_t const minNumDoubletsToPass) const;

  // class members
  TrackingParticleRef trackingParticleRef_;        // reference to the TrackingParticle
  std::vector<unsigned int> detIdVector_;          // vector of the detector Ids of the RecHits
                                                   // associated to the TP
  std::vector<int> moduleIdVector_;                // vector of the module Ids of the RecHits
  std::vector<GlobalPoint> globalPositionVector_;  // vector of the global positions of the RecHits
                                                   // (corrected by beamspot)
  std::vector<uint8_t> layerIdVector_;             // vector of layer IDs corresponding to the RecHits
  std::vector<int16_t> clusterYSizeVector_;        // vector of cluster sizes (local y) corresponding to the RecHits
  GlobalVector beamSpotPosition_;  // global position of the beam spot (needed to correct the global RecHit position)
  bool recHitsAreSorted_{false};   // true if RecHits were sorted
  int numLayers_{0};               // number of layers hit by the TrackingParticle

  // non-persistent, mutable members:
  // vector of true doublets
  mutable std::vector<Doublet> doublets_{};
  // vector of true Ntuplets
  mutable std::vector<Ntuplet> ntuplets_{};
  // index of the longest SimNtuplet
  mutable int longestNtupletIndex_{-1};
  // index of the longest SimNtuplet that survives
  mutable int longestAliveNtupletIndex_{-1};
};

// collection of SimDoublets
typedef std::vector<SimDoublets> SimDoubletsCollection;

#endif
