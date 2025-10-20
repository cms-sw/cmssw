#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"
#include <cstddef>
#include <cstdint>

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
// #include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

namespace simdoublets {

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
}  // end namespace simdoublets

// ------------------------------------------------------------------------------------------------------
// SimDoublets::Doublet class member functions
// ------------------------------------------------------------------------------------------------------

// constructor
SimDoublets::Doublet::Doublet(SimDoublets const& simDoublets,
                              size_t const innerIndex,
                              size_t const outerIndex,
                              const TrackerTopology* trackerTopology,
                              std::vector<size_t> const& innerNeighborsIndices)
    : moduleIds_(std::make_pair(simDoublets.moduleIds(innerIndex), simDoublets.moduleIds(outerIndex))),
      globalPositions_(
          std::make_pair(simDoublets.globalPositions(innerIndex), simDoublets.globalPositions(outerIndex))),
      layerIds_(std::make_pair(simDoublets.layerIds(innerIndex), simDoublets.layerIds(outerIndex))),
      clusterYSizes_(std::make_pair(simDoublets.clusterYSizes(innerIndex), simDoublets.clusterYSizes(outerIndex))),
      status_(SimDoublets::Doublet::Status::undef) {
  // determine number of skipped layers
  numSkippedLayers_ = simdoublets::getNumSkippedLayers(
      layerIds_, simDoublets.detIds(innerIndex), simDoublets.detIds(outerIndex), trackerTopology);

  // determine Id of the layer pair
  layerPairId_ = simdoublets::getLayerPairId(layerIds_);

  // fill the inner neighbors
  for (size_t const index : innerNeighborsIndices) {
    innerNeighbors_.emplace_back(SimDoublets::Doublet::Neighbor(index));
  }

  // if there are neighbors, get their inner layerId
  if (innerNeighborsIndices.size() > 0) {
    size_t index = innerNeighborsIndices.at(0);
    innerNeighborsInnerLayerId_ = simDoublets.getSimDoublet(index).innerLayerId();
  }
}

// ------------------------------------------------------------------------------------------------------
// SimDoublets class member functions
// ------------------------------------------------------------------------------------------------------

// method to add a RecHit to the SimPixelTrack
void SimDoublets::addRecHit(BaseTrackerRecHit const& recHit,
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

// method to sort the RecHits according to the position
void SimDoublets::sortRecHits() {
  // get the production vertex of the TrackingParticle (corrected for beamspot)
  const GlobalVector vertex(trackingParticleRef_->vx() - beamSpotPosition_.x(),
                            trackingParticleRef_->vy() - beamSpotPosition_.y(),
                            trackingParticleRef_->vz() - beamSpotPosition_.z());

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
void SimDoublets::buildSimDoublets(const TrackerTopology* trackerTopology) const {
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
  uint nDoublets{0};

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
      doublets_.emplace_back(SimDoublets::Doublet(*this, i, j, trackerTopology, innerDoubletsOfRecHit.at(i)));

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
void SimDoublets::buildSimNtuplets(SimDoublets::Doublet const& doublet,
                                   size_t numSimDoublets,
                                   size_t const lastLayerId,
                                   uint8_t const status,
                                   uint8_t const numSkippedLayers,
                                   size_t const minNumDoubletsToPass) const {
  // update the number of SimDoublets once before looping over the actual neighbors to be added
  numSimDoublets++;

  // loop over the inner neighboring doublets of the current doublet
  for (auto const& neighbor : doublet.innerNeighborsView()) {
    // get the inner neighboring doublet and the status of this connection
    auto const& neighborDoublet = doublets_.at(neighbor.index());

    // update the status of the current SimNtuplet by adding the information from the new doublet
    uint8_t updatedStatus = SimDoublets::Ntuplet::updateStatus(
        status,                                        // current status
        neighborDoublet.isUndef(),                     // doublet has undefined cuts
        neighborDoublet.isKilledByMissingLayerPair(),  // doublet is not built due to missing layer pair
        neighborDoublet.isKilledByCuts(),              // doublet is killed by cuts
        neighbor.isUndef(),                            // connection has undefined cuts
        neighbor.isKilled()                            // connection is killed by cuts
    );

    // update number of skipped layers
    uint8_t updatedNumSkippedLayers = numSkippedLayers + doublet.numSkippedLayers();

    // add the current state as a new SimNtuplet to the collection
    ntuplets_.emplace_back(SimDoublets::Ntuplet(
        numSimDoublets, updatedStatus, neighborDoublet.innerLayerId(), lastLayerId, updatedNumSkippedLayers));

    // change the status "TooShort" of the newly created SimNtuplet if it is indeed to short
    if (numSimDoublets < minNumDoubletsToPass) {
      ntuplets_.back().setTooShort();
    }

    // check if the new SimNtuplet qualifies as longest SimNtuplet>
    // A) if it's the first Ntuplet or longer than the current longest,
    //    it becomes automatically the longest
    // B) otherwise:
    //     - it needs to be at least as long as the current longest
    //     - and it needs to get at least as far in the reconstruction chain:
    //        1. no missing layer pairs
    //        2. all doublets survive
    //        3. all connections survive
    //        4. Ntuplet is long enough
    if ((longestNtupletIndex_ == -1) || (numSimDoublets > ntuplets_.at(longestNtupletIndex_).numDoublets())) {
      // case A)
      longestNtupletIndex_ = ntuplets_.size() - 1;
    } else if ((numSimDoublets == ntuplets_.at(longestNtupletIndex_).numDoublets()) &&  // is at least as long
               ntuplets_.back().getsFartherInRecoChainThanReference(
                   ntuplets_.at(longestNtupletIndex_))) {  // get farther in reconstruction
      // case B)
      longestNtupletIndex_ = ntuplets_.size() - 1;
    }

    // check if the new SimNtuplet qualifies as longest SimNtuplet alive
    if (ntuplets_.back().isAlive()) {           // obviously, it has to be alive
      if ((longestAliveNtupletIndex_ == -1) ||  // it's the first SimNtuplet alive or
          ((numSimDoublets >= ntuplets_.at(longestAliveNtupletIndex_).numDoublets()) &&  // is at least as long and
           (ntuplets_.back().firstLayerId() <=
            ntuplets_.at(longestAliveNtupletIndex_).firstLayerId()))  // is at least as inside
      ) {
        longestAliveNtupletIndex_ = ntuplets_.size() - 1;
      }
    }

    // call this function recursively
    // this will get the further neighboring doublets and build the next Ntuplet
    buildSimNtuplets(
        neighborDoublet, numSimDoublets, lastLayerId, updatedStatus, updatedNumSkippedLayers, minNumDoubletsToPass);
  }
}

// method to produce the SimNtuplets
// (collection of all possible Ntuplets you can build from the SimDoublets)
void SimDoublets::buildSimNtuplets(size_t const minNumDoubletsToPass) const {
  // clear the Ntuplet collection and reset longest Ntuplet indices
  ntuplets_.clear();
  longestNtupletIndex_ = -1;
  longestAliveNtupletIndex_ = -1;

  // check if there are at least two doublets
  if (numDoublets() < 2) {
    return;
  }

  // loop over all SimDoublets, using them as starting points for building Ntuplets
  for (auto const& doublet : doublets_) {
    // intialize status according to the doublet properties
    uint8_t status = SimDoublets::Ntuplet::updateStatus(
        0,                                     // current status to be updated
        doublet.isUndef(),                     // doublet has undefined cuts
        doublet.isKilledByMissingLayerPair(),  // doublet is not built due to missing layer pair
        doublet.isKilledByCuts(),              // doublet is killed by cuts
        false,                                 // connection has undefined cuts
        false                                  // connection is killed by cuts
    );
    // initialize number of skipped layers
    uint8_t numSkippedLayers = doublet.numSkippedLayers();
    // build the Ntuplets recursively
    buildSimNtuplets(doublet, 1, doublet.outerLayerId(), status, numSkippedLayers, minNumDoubletsToPass);
  }
}