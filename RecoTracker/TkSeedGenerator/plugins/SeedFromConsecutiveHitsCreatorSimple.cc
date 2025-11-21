#include "SeedFromConsecutiveHitsCreatorSimple.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {

  template <class T>
  inline T sqr(T t) {
    return t * t;
  }

}  // namespace

SeedFromConsecutiveHitsCreatorSimple::SeedFromConsecutiveHitsCreatorSimple(const edm::ParameterSet& cfg,
                                                                           edm::ConsumesCollector&& iC) {}

SeedFromConsecutiveHitsCreatorSimple::~SeedFromConsecutiveHitsCreatorSimple() {}

void SeedFromConsecutiveHitsCreatorSimple::fillDescriptions(edm::ParameterSetDescription& desc) {
}

void SeedFromConsecutiveHitsCreatorSimple::init(const TrackingRegion& iregion,
                                                const edm::EventSetup& es,
                                                const SeedComparitor* ifilter) {
  region = &iregion;
  filter = ifilter;
}

void SeedFromConsecutiveHitsCreatorSimple::makeSeed(TrajectorySeedCollection& seedCollection, const SeedingHitSet& hits) {
  if (hits.size() < 2)
    return;

  TrajectoryStateOnSurface state;
  edm::OwnVector<TrackingRecHit> seedHits;

  const TrackingRecHit* hit = nullptr;
  for (unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit]->hit();
    if (!hit) {
      continue;
    }
    seedHits.push_back(*hit);
  }

  PTrajectoryStateOnDet const& PTraj =
      trajectoryStateTransform::persistentState(state, hit->geographicalId().rawId());
  seedCollection.emplace_back(PTraj, std::move(seedHits), alongMomentum);
}