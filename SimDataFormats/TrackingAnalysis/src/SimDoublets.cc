#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"

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
    : trackingParticleRef_(simDoublets.trackingParticle()),
      status_(SimDoublets::Doublet::Status::undef),
      beamSpotPosition_(simDoublets.beamSpotPosition()) {
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
void SimDoublets::buildSimNtuplets(std::vector<SimDoublets::Ntuplet>& simNtuplets,
                                   SimDoublets::Doublet const& doublet,
                                   size_t numSimDoublets,
                                   size_t const lastLayerId,
                                   SimDoublets::Ntuplet::Status status) const {
  // update the status of the current SimNtuplet by adding the information from the new doublet
  numSimDoublets++;
  switch (status) {
    case SimDoublets::Ntuplet::Status::killedByMissingLayerPair:
      // if the Ntuplet already got killed by missing layer pair,
      // nothing will change this state as this is first requirement
      break;
    case SimDoublets::Ntuplet::Status::killedByDoubletCuts:
      // if the Ntuplet already got killed by doublet cuts,
      // nothing will change this state as those are the first cuts applied
      // unless a layer pair is already missing
      if (doublet.isKilledByMissingLayerPair()) {
        status = SimDoublets::Ntuplet::Status::killedByMissingLayerPair;
      }
      break;
    case SimDoublets::Ntuplet::Status::undefDoubletCuts:
      // similar if previous doublet cuts were undefined,
      // nothing should change this state unless the new doublet got killed (stronger statement)
      if (doublet.isKilledByMissingLayerPair()) {
        status = SimDoublets::Ntuplet::Status::killedByMissingLayerPair;
      } else if (doublet.isKilledByCuts()) {
        status = SimDoublets::Ntuplet::Status::killedByDoubletCuts;
      }
      break;
    default:
      // in all other cases,
      // we want to check if the new doublet even survived or was checked at all
      if (doublet.isKilledByMissingLayerPair()) {
        status = SimDoublets::Ntuplet::Status::killedByMissingLayerPair;
      } else if (doublet.isKilledByCuts()) {
        status = SimDoublets::Ntuplet::Status::killedByDoubletCuts;
      } else if (doublet.isUndef()) {
        status = SimDoublets::Ntuplet::Status::undefDoubletCuts;
      }
      break;
  }

  // in case our current Ntuplet has more than 1 doublet, we add the current state
  // as a new SimNtuplet to the collection
  if (numSimDoublets > 1) {
    simNtuplets.emplace_back(SimDoublets::Ntuplet(numSimDoublets, status, doublet.innerLayerId(), lastLayerId));
  }

  // then loop over the inner neighboring doublets of the current doublet
  for (auto const& neighbor : doublet.innerNeighborsView()) {
    // get the inner neighboring doublet and the status of this connection
    auto const& neighborDoublet = doublets_.at(neighbor.index());

    // update the status according to the connection status to the neighbor
    switch (status) {
      case SimDoublets::Ntuplet::Status::undefDoubletConnectionCuts:
        // if previous connection cuts were undefined,
        // we only want to change the status if the new connection was actually killed
        if (neighbor.isKilled()) {
          status = SimDoublets::Ntuplet::Status::killedByDoubletConnectionCuts;
        }
        break;
      case SimDoublets::Ntuplet::Status::alive:
        // if the Ntuplet is still alive,
        // check if the new connection changes this
        if (neighbor.isKilled()) {
          status = SimDoublets::Ntuplet::Status::killedByDoubletConnectionCuts;
        } else if (neighbor.isUndef()) {
          status = SimDoublets::Ntuplet::Status::undefDoubletConnectionCuts;
        }
        break;
      default:
        // in all other cases,
        // we don't want the status to change as it already is in a stronger state
        break;
    }

    // call this function recursively
    // this will add the neighboring doublet and build the next Ntuplet
    buildSimNtuplets(simNtuplets, neighborDoublet, numSimDoublets, lastLayerId, status);
  }
}

// method to produce and get the SimNtuplets
// (collection of all possible Ntuplets you can build from the SimDoublets)
std::vector<SimDoublets::Ntuplet> SimDoublets::getSimNtuplets() const {
  // create the vector of SimNtuplets
  std::vector<SimDoublets::Ntuplet> simNtuplets{};

  // check if there are at least two doublets
  if (numDoublets() < 2) {
    return simNtuplets;
  }

  // // updatable number of SimDoublets in the SimNtuplet
  // size_t numSimDoublets, lastLayerId;
  // // updatable status of the currently build SimNtuplet
  // SimDoublets::Ntuplet::Status status;

  // loop over all SimDoublets, using them as starting points for building Ntuplets
  for (auto const& doublet : doublets_) {
    // // reset the number of SimDoublets in the Ntuplet(s) to 1
    // numSimDoublets = 1;
    // // reset the status of the Ntuplet to the default alive
    // status = SimDoublets::Ntuplet::Status::alive;
    // // set the lastLayerId to the one of the starting doublet (we build from outside to inside)
    // lastLayerId = doublet.outerLayerId();

    // recursively find the next inner neighbors and build SimNtuplets at every stage of adding
    buildSimNtuplets(simNtuplets, doublet);
  }

  return simNtuplets;
}