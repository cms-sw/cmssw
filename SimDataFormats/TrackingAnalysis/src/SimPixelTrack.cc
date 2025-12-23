#include "SimDataFormats/TrackingAnalysis/interface/SimPixelTrack.h"
#include <cstddef>
#include <cstdint>

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
// #include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

namespace simpixeltracks {

  // Function that determines the number of skipped layers for a given pair of RecHits.
  int getNumSkippedLayers(std::pair<uint8_t, uint8_t> const& layerIds,
                          unsigned int const inner_detId,
                          unsigned int const outer_detId,
                          const TrackerTopology* trackerTopology) {
    // Possibility 0: invalid case (outer layer is not the outer one), set to -1 immediately
    if (layerIds.first >= layerIds.second) {
      return -1;
    }

    // get the detector Ids of the two RecHits
    DetId innerDetId(inner_detId);
    DetId outerDetId(outer_detId);

    // determine where the RecHits are
    bool innerInBarrel = (innerDetId.subdetId() == PixelSubdetector::PixelBarrel);
    bool outerInBarrel = (outerDetId.subdetId() == PixelSubdetector::PixelBarrel);
    bool innerInOTExtension = (innerDetId.subdetId() == SiStripSubdetector::TOB);
    bool outerInOTExtension = (outerDetId.subdetId() == SiStripSubdetector::TOB);
    bool innerInBackward = (!innerInBarrel) && (!innerInOTExtension) && (trackerTopology->pxfSide(innerDetId) == 1);
    bool outerInBackward = (!outerInBarrel) && (!outerInOTExtension) && (trackerTopology->pxfSide(outerDetId) == 1);
    bool innerInForward = (!innerInBarrel) && (!innerInOTExtension) && (trackerTopology->pxfSide(innerDetId) == 2);
    bool outerInForward = (!outerInBarrel) && (!outerInOTExtension) && (trackerTopology->pxfSide(outerDetId) == 2);

    // Possibility 1: both RecHits lie in the same detector part (barrel, forward, backward, extension)
    if ((innerInBarrel && outerInBarrel) || (innerInForward && outerInForward) ||
        (innerInBackward && outerInBackward) || (innerInOTExtension && outerInOTExtension)) {
      return (layerIds.second - layerIds.first - 1);
    }
    // Possibility 2: the inner RecHit is in the barrel while the outer is in extension
    else if (innerInBarrel && outerInOTExtension) {
      return (trackerTopology->tobLayer(outerDetId) - trackerTopology->pxbLayer(innerDetId) + 3);
    }
    // Possibility 3: the inner RecHit is in the encaps while the outer is in extension
    else if (outerInOTExtension) {
      return (trackerTopology->tobLayer(outerDetId) - 1);
    }
    // Possibility 4: the inner RecHit is in the barrel while the outer is in either forward or backward
    else if (innerInBarrel) {
      return (trackerTopology->pxfDisk(outerDetId) - 1);
    }
    // Possibility 5: invalid case (one is forward and the other in backward), set to -1
    else {
      return -1;
    }
  }

  // Function that, for a pair of two layers, gives a unique pair Id (innerLayerId * 100 + outerLayerId).
  int getLayerPairId(std::pair<uint8_t, uint8_t> const& layerIds) {
    // calculate the unique layer pair Id as (innerLayerId * 100 + outerLayerId)
    return (layerIds.first * 100 + layerIds.second);
  }
}  // end namespace simpixeltracks

// ------------------------------------------------------------------------------------------------------
// SimPixelTrack::Doublet class member functions
// ------------------------------------------------------------------------------------------------------

// constructor
SimPixelTrack::Doublet::Doublet(SimPixelTrack const& simPixelTrack,
                                size_t const innerIndex,
                                size_t const outerIndex,
                                const TrackerTopology* trackerTopology,
                                std::vector<size_t> const& innerNeighborsIndices)
    : moduleIds_(std::make_pair(simPixelTrack.moduleIds(innerIndex), simPixelTrack.moduleIds(outerIndex))),
      globalPositions_(
          std::make_pair(simPixelTrack.globalPositions(innerIndex), simPixelTrack.globalPositions(outerIndex))),
      layerIds_(std::make_pair(simPixelTrack.layerIds(innerIndex), simPixelTrack.layerIds(outerIndex))),
      clusterYSizes_(std::make_pair(simPixelTrack.clusterYSizes(innerIndex), simPixelTrack.clusterYSizes(outerIndex))),
      status_(SimPixelTrack::Doublet::Status::undef) {
  // determine number of skipped layers
  numSkippedLayers_ = simpixeltracks::getNumSkippedLayers(
      layerIds_, simPixelTrack.detIds(innerIndex), simPixelTrack.detIds(outerIndex), trackerTopology);

  // determine Id of the layer pair
  layerPairId_ = simpixeltracks::getLayerPairId(layerIds_);

  // fill the inner neighbors
  for (size_t const index : innerNeighborsIndices) {
    innerNeighbors_.emplace_back(
        SimPixelTrack::Doublet::Neighbor(index, simPixelTrack.getSimDoublet(index).numInnerNeighbors()));
  }

  // if there are neighbors, get their inner layerId
  if (!innerNeighborsIndices.empty()) {
    size_t index = innerNeighborsIndices.at(0);
    innerNeighborsInnerLayerId_ = simPixelTrack.getSimDoublet(index).innerLayerId();
  }
}

// ------------------------------------------------------------------------------------------------------
// SimPixelTrack class member functions
// ------------------------------------------------------------------------------------------------------

// method to add a RecHit to the SimPixelTrack
void SimPixelTrack::addRecHit(TrackingRecHit const& recHit,
                              uint8_t const layerId,
                              int16_t const clusterYSize,
                              unsigned int const detId,
                              int const moduleId) {
  recHitsAreSorted_ = false;  // set sorted-bool to false again

  // check if the layerId is not present in the layerIdVector yet
  if (std::find(layerIdVector_.begin(), layerIdVector_.end(), layerId) == layerIdVector_.end()) {
    // if it does not exist, increment number of layers
    numLayers_++;
  }

  // add detId, the corrected hit position, layerId and clusterSize to respective vectors
  detIdVector_.push_back(detId);
  moduleIdVector_.push_back(moduleId);
  globalPositionVector_.push_back(recHit.globalPosition() - beamSpotPosition_);
  layerIdVector_.push_back(layerId);
  clusterYSizeVector_.push_back(clusterYSize);
}

// method to sort the RecHits according to the position relative to the TP vertex
void SimPixelTrack::sortRecHits() {
  auto vertex = trackingParticleRef_->vertex();
  sortRecHits(vertex.x(), vertex.y(), vertex.z());
}
// method to sort the RecHits according to the position relative to a given reference
void SimPixelTrack::sortRecHits(float const x, float const y, float const z) {
  // get the production vertex of the TrackingParticle (corrected for beamspot)
  const GlobalVector vertex(x - beamSpotPosition_.x(), y - beamSpotPosition_.y(), z - beamSpotPosition_.z());

  // get the vector of squared magnitudes of the global RecHit positions relative to vertex
  std::vector<double> recHitMag2;
  recHitMag2.reserve(layerIdVector_.size());
  for (const auto& globalPosition : globalPositionVector_) {
    // relative RecHit position with respect to the production vertex
    Global3DPoint relativePosition = globalPosition - vertex;
    recHitMag2.push_back(relativePosition.mag2());
  }

  // find the permutation vector that sorts the magnitudes
  std::vector<std::size_t> sortedPerm(recHitMag2.size());
  std::iota(sortedPerm.begin(), sortedPerm.end(), 0);
  std::sort(sortedPerm.begin(), sortedPerm.end(), [&](std::size_t i, std::size_t j) {
    return (recHitMag2[i] < recHitMag2[j]);
  });

  // create the sorted vectors
  std::vector<unsigned int> sorted_detIdVector;
  std::vector<int> sorted_moduleIdVector;
  std::vector<GlobalPoint> sorted_globalPositionVector;
  std::vector<uint8_t> sorted_layerIdVector;
  std::vector<int16_t> sorted_clusterYSizeVector;
  sorted_detIdVector.reserve(sortedPerm.size());
  sorted_moduleIdVector.reserve(sortedPerm.size());
  sorted_globalPositionVector.reserve(sortedPerm.size());
  sorted_layerIdVector.reserve(sortedPerm.size());
  sorted_clusterYSizeVector.reserve(sortedPerm.size());
  for (size_t i : sortedPerm) {
    sorted_detIdVector.push_back(detIdVector_[i]);
    sorted_moduleIdVector.push_back(moduleIdVector_[i]);
    sorted_globalPositionVector.push_back(globalPositionVector_[i]);
    sorted_layerIdVector.push_back(layerIdVector_[i]);
    sorted_clusterYSizeVector.push_back(clusterYSizeVector_[i]);
  }

  // swap them with the class member
  detIdVector_.swap(sorted_detIdVector);
  moduleIdVector_.swap(sorted_moduleIdVector);
  globalPositionVector_.swap(sorted_globalPositionVector);
  layerIdVector_.swap(sorted_layerIdVector);
  clusterYSizeVector_.swap(sorted_clusterYSizeVector);

  // set sorted bool to true
  recHitsAreSorted_ = true;
}

// method to produce the true doublets
void SimPixelTrack::buildSimDoublets(const TrackerTopology* trackerTopology) const {
  // confirm that the RecHits are sorted
  assert(recHitsAreSorted_);

  // check if there are at least two hits
  if (numRecHits() < 2) {
    return;
  }

  // vector of length NrecHits that holds for each RecHit references to all the
  // doublets that have this hit as an outer hit
  std::vector<std::vector<size_t>> innerDoubletsOfRecHit{};
  // resize vector innerDoubletsOfRecHit to actual number of RecHits
  innerDoubletsOfRecHit.resize(numRecHits());

  // updatable current number of doublets
  size_t nDoublets{0};

  // loop over the RecHits/layer Ids
  for (size_t i = 0; i < layerIdVector_.size(); i++) {
    uint8_t innerLayerId = layerIdVector_[i];
    uint8_t outerLayerId{};
    size_t outerLayerStart{layerIdVector_.size()};

    // find the next layer Id + at which hit this layer starts
    for (size_t j = i + 1; j < layerIdVector_.size(); j++) {
      if (innerLayerId != layerIdVector_[j]) {
        outerLayerId = layerIdVector_[j];
        outerLayerStart = j;
        break;
      }
    }

    // build the doublets of the inner hit i with all outer hits j in the layer outerLayerId
    for (size_t j = outerLayerStart; j < layerIdVector_.size(); j++) {
      // break if the hit doesn't belong to the outer layer anymore
      if (outerLayerId != layerIdVector_[j]) {
        break;
      }

      // create and append new doublet
      doublets_.emplace_back(SimPixelTrack::Doublet(*this, i, j, trackerTopology, innerDoubletsOfRecHit.at(i)));

      // save the index of the new doublet in the outer RecHit's innerDoubletsOfRecHit
      innerDoubletsOfRecHit.at(j).push_back(nDoublets);

      // update the number of doublets
      nDoublets++;
    }
  }  // end loop over the RecHits/layer Ids
}

// function to recursively build the Ntuplets from a given starting doublet
// (the building starts from the outside and ends inside)
// at each addition of a SimDoublet, a new SimNtuplet is stored
void SimPixelTrack::buildSimNtuplets(SimPixelTrack::Doublet const& doublet,
                                     std::vector<bool> const& tripletConnections,
                                     size_t numSimDoublets,
                                     size_t const lastLayerId,
                                     uint8_t const status,
                                     uint8_t const numSkippedLayers,
                                     std::set<int> const& startingPairs,
                                     size_t const minNumDoubletsToPass) const {
  // update the number of SimDoublets once before looping over the actual neighbors to be added
  numSimDoublets++;

  // loop over the inner neighboring doublets of the current doublet
  for (size_t i{0}; auto const& neighbor : doublet.innerNeighborsView()) {
    // get the inner neighboring doublet and the status of this connection
    auto const& neighborDoublet = doublets_.at(neighbor.index());

    // update the status of the current SimNtuplet by adding the information from the new doublet
    uint8_t updatedStatus = SimPixelTrack::Ntuplet::updateStatus(
        status,                                                  // current status
        neighborDoublet.isUndef(),                               // doublet has undefined cuts
        neighborDoublet.isKilledByMissingLayerPair(),            // doublet is not built due to missing layer pair
        neighborDoublet.isKilledByCuts(),                        // doublet is killed by cuts
        neighbor.isUndef(),                                      // doublet connection has undefined cuts
        neighbor.isKilled(),                                     // doublet connection is killed by cuts
        (numSimDoublets > 2) ? tripletConnections.at(i) : false  // triplet connection is killed by cuts
    );

    // update number of skipped layers
    uint8_t updatedNumSkippedLayers = numSkippedLayers + doublet.numSkippedLayers();

    // add the current state as a new SimNtuplet to the collection
    ntuplets_.emplace_back(SimPixelTrack::Ntuplet(numSimDoublets,
                                                  updatedStatus,
                                                  neighborDoublet.innerLayerId(),
                                                  neighborDoublet.outerLayerId(),
                                                  lastLayerId,
                                                  updatedNumSkippedLayers));

    // change the status "TooShort" of the newly created SimNtuplet if it is indeed to short
    if (numSimDoublets < minNumDoubletsToPass) {
      ntuplets_.back().setTooShort();
    }

    // change the status "firstDoubletNotInStartingLayerPairs" of the newly created SimNtuplet if this is indeed the case
    if (!startingPairs.contains(neighborDoublet.layerPairId())) {
      ntuplets_.back().setFirstDoubletNotInStartingLayerPairs();
    }

    // check if the new SimNtuplet qualifies as longest SimNtuplet
    // A) if it's the first Ntuplet or longer than the current longest,
    //    it becomes automatically the longest
    // B) otherwise:
    //     - it needs to be at least as long as the current longest
    //     - and it needs to get farther in the reconstruction chain:
    //        1. Ntuplet is long enough
    //        2. no missing layer pairs
    //        3. all doublets survive
    //        4. all doublet connections survive
    //        5. all triplet connections survive
    //        6. first doublet from starting layer pair
    if ((!longestNtupletIndex_) || (numSimDoublets > ntuplets_.at(*longestNtupletIndex_).numDoublets())) {
      // case A)
      longestNtupletIndex_ = ntuplets_.size() - 1;
    } else if ((numSimDoublets == ntuplets_.at(*longestNtupletIndex_).numDoublets()) &&  // is at least as long
               ntuplets_.back().getsFartherInRecoChainThanReference(
                   ntuplets_.at(*longestNtupletIndex_))) {  // get farther in reconstruction
      // case B)
      longestNtupletIndex_ = ntuplets_.size() - 1;
    }

    // check if the new SimNtuplet qualifies as longest SimNtuplet alive
    if (ntuplets_.back().isAlive()) {      // obviously, it has to be alive
      if ((!longestAliveNtupletIndex_) ||  // it's the first SimNtuplet alive or
          ((numSimDoublets >= ntuplets_.at(*longestAliveNtupletIndex_).numDoublets()) &&  // is at least as long and
           (ntuplets_.back().firstLayerId() <=
            ntuplets_.at(*longestAliveNtupletIndex_).firstLayerId()))  // is at least as inside
      ) {
        longestAliveNtupletIndex_ = ntuplets_.size() - 1;
      }
    }

    // check if the new SimNtuplet qualifies as best SimNtuplet yet
    // A) if it's the first Ntuplet or farther in the reco chain than the current best,
    //    it becomes automatically the best
    // B) otherwise:
    //     - it needs to be at least as long as the current longest
    //     - and it needs to get at least as far in the reconstruction chain:
    //        1. Ntuplet is long enough
    //        2. no missing layer pairs
    //        3. all doublets survive
    //        4. all doublet connections survive
    //        5. all triplet connections survive
    //        6. first doublet from starting layer pair
    if ((!bestNtupletIndex_) ||
        ntuplets_.back().getsFartherInRecoChainThanReference(ntuplets_.at(*bestNtupletIndex_))) {
      // case A)
      bestNtupletIndex_ = ntuplets_.size() - 1;
    } else if ((numSimDoublets >= ntuplets_.at(*bestNtupletIndex_).numDoublets()) &&  // is at least as long
               ntuplets_.back().getsAsFarInRecoChainAsReference(
                   ntuplets_.at(*bestNtupletIndex_))) {  // get as far in reconstruction
      // case B)
      bestNtupletIndex_ = ntuplets_.size() - 1;
    }

    // call this function recursively
    // this will get the further neighboring doublets and build the next Ntuplet
    buildSimNtuplets(neighborDoublet,
                     neighbor.tripletConnections(),
                     numSimDoublets,
                     lastLayerId,
                     updatedStatus,
                     updatedNumSkippedLayers,
                     startingPairs,
                     minNumDoubletsToPass);

    i++;
  }
}

// method to produce the SimNtuplets
// (collection of all possible Ntuplets you can build from the SimDoublets)
void SimPixelTrack::buildSimNtuplets(std::set<int> const& startingPairs, size_t const minNumDoubletsToPass) const {
  // clear the Ntuplet collection and reset longest Ntuplet indices
  ntuplets_.clear();
  longestNtupletIndex_.reset();
  longestAliveNtupletIndex_.reset();
  bestNtupletIndex_.reset();

  // check if there are at least two doublets
  if (numDoublets() < 2) {
    return;
  }

  // loop over all SimDoublets, using them as starting points for building Ntuplets
  for (auto const& doublet : doublets_) {
    // intialize status according to the doublet properties
    uint8_t status = SimPixelTrack::Ntuplet::updateStatus(
        0,                                     // current status to be updated
        doublet.isUndef(),                     // doublet has undefined cuts
        doublet.isKilledByMissingLayerPair(),  // doublet is not built due to missing layer pair
        doublet.isKilledByCuts(),              // doublet is killed by cuts
        false,                                 // doublet connection has undefined cuts
        false,                                 // doublet connection is killed by cuts
        false                                  // triplet connection is killed by cuts
    );
    // initialize number of skipped layers
    uint8_t numSkippedLayers = doublet.numSkippedLayers();
    // build the Ntuplets recursively
    buildSimNtuplets(
        doublet, {}, 1, doublet.outerLayerId(), status, numSkippedLayers, startingPairs, minNumDoubletsToPass);
  }
}
