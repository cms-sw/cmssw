#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"
#include <cstdint>

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

namespace simdoublets {

  // Function that gets the global position of a RecHit with respect to a reference point.
  GlobalPoint getGlobalHitPosition(SiPixelRecHitRef const& recHit, GlobalVector const& referencePosition) {
    return (recHit->globalPosition() - referencePosition);
  }

  // Function that determines the number of skipped layers for a given pair of RecHits.
  int getNumSkippedLayers(std::pair<uint8_t, uint8_t> const& layerIds,
                          std::pair<SiPixelRecHitRef, SiPixelRecHitRef> const& recHitRefs,
                          const TrackerTopology* trackerTopology) {
    // Possibility 0: invalid case (outer layer is not the outer one), set to -1 immediately
    if (layerIds.first >= layerIds.second) {
      return -1;
    }

    // get the detector Ids of the two RecHits
    DetId innerDetId(recHitRefs.first->geographicalId());
    DetId outerDetId(recHitRefs.second->geographicalId());

    // determine where the RecHits are
    bool innerInBarrel = (innerDetId.subdetId() == PixelSubdetector::PixelBarrel);
    bool outerInBarrel = (outerDetId.subdetId() == PixelSubdetector::PixelBarrel);
    bool innerInBackward = (!innerInBarrel) && (trackerTopology->pxfSide(innerDetId) == 1);
    bool outerInBackward = (!outerInBarrel) && (trackerTopology->pxfSide(outerDetId) == 1);
    bool innerInForward = (!innerInBarrel) && (!innerInBackward);
    bool outerInForward = (!outerInBarrel) && (!outerInBackward);

    // Possibility 1: both RecHits lie in the same detector part (barrel, forward or backward)
    if ((innerInBarrel && outerInBarrel) || (innerInForward && outerInForward) ||
        (innerInBackward && outerInBackward)) {
      return (layerIds.second - layerIds.first - 1);
    }
    // Possibility 2: the inner RecHit is in the barrel while the outer is in either forward or backward
    else if (innerInBarrel) {
      return (trackerTopology->pxfDisk(outerDetId) - 1);
    }
    // Possibility 3: invalid case (one is forward and the other in backward), set to -1
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
    : status_(SimDoublets::Doublet::Status::undef), beamSpotPosition_(simDoublets.beamSpotPosition()) {
  // fill recHits and layers
  recHitRefs_ = std::make_pair(simDoublets.recHits(innerIndex), simDoublets.recHits(outerIndex));
  layerIds_ = std::make_pair(simDoublets.layerIds(innerIndex), simDoublets.layerIds(outerIndex));

  // determine number of skipped layers
  numSkippedLayers_ = simdoublets::getNumSkippedLayers(layerIds_, recHitRefs_, trackerTopology);

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

GlobalPoint SimDoublets::Doublet::innerGlobalPos() const {
  // get the inner RecHit's global position
  return simdoublets::getGlobalHitPosition(recHitRefs_.first, beamSpotPosition_);
}

GlobalPoint SimDoublets::Doublet::outerGlobalPos() const {
  // get the outer RecHit's global position
  return simdoublets::getGlobalHitPosition(recHitRefs_.second, beamSpotPosition_);
}

// ------------------------------------------------------------------------------------------------------
// SimDoublets class member functions
// ------------------------------------------------------------------------------------------------------

// method to sort the RecHits according to the position
void SimDoublets::sortRecHits() {
  // get the production vertex of the TrackingParticle
  const GlobalVector vertex(trackingParticleRef_->vx(), trackingParticleRef_->vy(), trackingParticleRef_->vz());

  // get the vector of squared magnitudes of the global RecHit positions
  std::vector<double> recHitMag2;
  recHitMag2.reserve(layerIdVector_.size());
  for (const auto& recHit : recHitRefVector_) {
    // global RecHit position with respect to the production vertex
    Global3DPoint globalPosition = simdoublets::getGlobalHitPosition(recHit, vertex);
    recHitMag2.push_back(globalPosition.mag2());
  }

  // find the permutation vector that sort the magnitudes
  std::vector<std::size_t> sortedPerm(recHitMag2.size());
  std::iota(sortedPerm.begin(), sortedPerm.end(), 0);
  std::sort(sortedPerm.begin(), sortedPerm.end(), [&](std::size_t i, std::size_t j) {
    return (recHitMag2[i] < recHitMag2[j]);
  });

  // create the sorted recHitRefVector and the sorted layerIdVector accordingly
  SiPixelRecHitRefVector sorted_recHitRefVector;
  std::vector<uint8_t> sorted_layerIdVector;
  sorted_recHitRefVector.reserve(sortedPerm.size());
  sorted_layerIdVector.reserve(sortedPerm.size());
  for (size_t i : sortedPerm) {
    sorted_recHitRefVector.push_back(recHitRefVector_[i]);
    sorted_layerIdVector.push_back(layerIdVector_[i]);
  }

  // swap them with the class member
  recHitRefVector_.swap(sorted_recHitRefVector);
  layerIdVector_.swap(sorted_layerIdVector);

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
                                   uint8_t status,
                                   size_t const minNumDoubletsToPass) const {
  // loop over the inner neighboring doublets of the current doublet
  for (auto const& neighbor : doublet.innerNeighborsView()) {
    // get the inner neighboring doublet and the status of this connection
    auto const& neighborDoublet = doublets_.at(neighbor.index());

    // update the status of the current SimNtuplet by adding the information from the new doublet
    SimDoublets::Ntuplet::updateStatus(
        status,                                        // current status
        neighborDoublet.isUndef(),                     // doublet has undefined cuts
        neighborDoublet.isKilledByMissingLayerPair(),  // doublet is not built due to missing layer pair
        neighborDoublet.isKilledByCuts(),              // doublet is killed by cuts
        neighbor.isUndef(),                            // connection has undefined cuts
        neighbor.isKilled()                            // connection is killed by cuts
    );

    numSimDoublets++;

    // add the current state as a new SimNtuplet to the collection
    ntuplets_.emplace_back(SimDoublets::Ntuplet(numSimDoublets, status, neighborDoublet.innerLayerId(), lastLayerId));

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
    }
    else if ((numSimDoublets == ntuplets_.at(longestNtupletIndex_).numDoublets()) &&  // is at least as long
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
    buildSimNtuplets(neighborDoublet, numSimDoublets, lastLayerId, status, minNumDoubletsToPass);
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
    // intialize status to no cuts no undefined
    uint8_t status{0};
    // update the status according to the doublet properties
    SimDoublets::Ntuplet::updateStatus(
        status,                                // current status to be updated
        doublet.isUndef(),                     // doublet has undefined cuts
        doublet.isKilledByMissingLayerPair(),  // doublet is not built due to missing layer pair
        doublet.isKilledByCuts(),              // doublet is killed by cuts
        false,                                 // connection has undefined cuts
        false                                  // connection is killed by cuts
    );
    // build the Ntuplets recursively
    buildSimNtuplets(doublet, 1, doublet.outerLayerId(), status, minNumDoubletsToPass);
  }
}