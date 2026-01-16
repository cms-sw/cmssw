#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class dso_hidden SeedFromConsecutiveHitsCreatorFakeState : public SeedCreator {
public:
  SeedFromConsecutiveHitsCreatorFakeState(const edm::ParameterSet &, edm::ConsumesCollector &&);

  ~SeedFromConsecutiveHitsCreatorFakeState() override = default;

  static void fillDescriptions(edm::ParameterSetDescription &desc);
  static const char *fillDescriptionsLabel() { return "ConsecutiveHitsSimple"; }

  // initialize the "event dependent state"
  void init(const TrackingRegion &region, const edm::EventSetup &es, const SeedComparitor *filter) final;

  // make job
  // fill seedCollection with the "TrajectorySeed"
  void makeSeed(TrajectorySeedCollection &seedCollection, const SeedingHitSet &hits) final;

protected:
  const TrackingRegion *region = nullptr;
  const SeedComparitor *filter = nullptr;
};

SeedFromConsecutiveHitsCreatorFakeState::SeedFromConsecutiveHitsCreatorFakeState(const edm::ParameterSet &cfg,
                                                                                 edm::ConsumesCollector &&iC) {}

void SeedFromConsecutiveHitsCreatorFakeState::fillDescriptions(edm::ParameterSetDescription &desc) {}

void SeedFromConsecutiveHitsCreatorFakeState::init(const TrackingRegion &iregion,
                                                   const edm::EventSetup &es,
                                                   const SeedComparitor *ifilter) {
  region = &iregion;
  filter = ifilter;
}

void SeedFromConsecutiveHitsCreatorFakeState::makeSeed(TrajectorySeedCollection &seedCollection,
                                                       const SeedingHitSet &hits) {
  if (hits.size() < 2)
    return;

  edm::OwnVector<TrackingRecHit> seedHits;

  const TrackingRecHit *hit = nullptr;
  for (unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit]->hit();
    if (!hit) {
      continue;
    }
    seedHits.push_back(*hit);
  }

  // Since there is no valid state with valid trajectory parameters, the trajectory state creation is faked
  PTrajectoryStateOnDet const fakePTraj = PTrajectoryStateOnDet();

  seedCollection.emplace_back(fakePTraj, std::move(seedHits), alongMomentum);
}

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFromRegionHitsEDProducerT.h"
#include "FWCore/Framework/interface/MakerMacros.h"
using FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer =
    SeedCreatorFromRegionHitsEDProducerT<SeedFromConsecutiveHitsCreatorFakeState>;
DEFINE_FWK_MODULE(FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer);
