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
                              const TrackerTopology* trackerTopology)
    : trackingParticleRef_(simDoublets.trackingParticle()), beamSpotPosition_(simDoublets.beamSpotPosition()) {
  // fill recHits and layers
  recHitRefs_ = std::make_pair(simDoublets.recHits(innerIndex), simDoublets.recHits(outerIndex));
  layerIds_ = std::make_pair(simDoublets.layerIds(innerIndex), simDoublets.layerIds(outerIndex));

  // determine number of skipped layers
  numSkippedLayers_ = simdoublets::getNumSkippedLayers(layerIds_, recHitRefs_, trackerTopology);

  // determine Id of the layer pair
  layerPairId_ = simdoublets::getLayerPairId(layerIds_);
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

// method to produce the true doublets on the fly
std::vector<SimDoublets::Doublet> SimDoublets::getSimDoublets(const TrackerTopology* trackerTopology) const {
  // create output vector for the doublets
  std::vector<SimDoublets::Doublet> doubletVector;

  // confirm that the RecHits are sorted
  assert(recHitsAreSorted_);

  // check if there are at least two hits
  if (numRecHits() < 2) {
    return doubletVector;
  }

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

    // build the doublets of the inner hit i with all outer hits in the layer outerLayerId
    for (size_t j = outerLayerStart; j < layerIdVector_.size(); j++) {
      // break if the hit doesn't belong to the outer layer anymore
      if (outerLayerId != layerIdVector_[j]) {
        break;
      }

      doubletVector.push_back(SimDoublets::Doublet(*this, i, j, trackerTopology));
    }
  }  // end loop over the RecHits/layer Ids

  return doubletVector;
}