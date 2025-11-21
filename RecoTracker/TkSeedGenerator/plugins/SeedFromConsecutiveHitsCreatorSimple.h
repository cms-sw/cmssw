#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreatorSimple_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreatorSimple_H
#include "FWCore/Utilities/interface/Visibility.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

class FreeTrajectoryState;

class dso_hidden SeedFromConsecutiveHitsCreatorSimple : public SeedCreator {
public:
  SeedFromConsecutiveHitsCreatorSimple(const edm::ParameterSet &, edm::ConsumesCollector &&);

  ~SeedFromConsecutiveHitsCreatorSimple() override;

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
#endif
