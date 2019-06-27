#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"

#include <memory>

class TrackingRegion;

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class SeedGeneratorFromRegionHits {
public:
  SeedGeneratorFromRegionHits(std::unique_ptr<OrderedHitsGenerator> aGenerator,
                              std::unique_ptr<SeedComparitor> aComparitor = nullptr,
                              std::unique_ptr<SeedCreator> aSeedCreator = nullptr);

  // make job
  void run(TrajectorySeedCollection& seedCollection,
           const TrackingRegion& region,
           const edm::Event& ev,
           const edm::EventSetup& es);

private:
  std::unique_ptr<OrderedHitsGenerator> theHitsGenerator;
  std::unique_ptr<SeedComparitor> theComparitor;
  std::unique_ptr<SeedCreator> theSeedCreator;
};
#endif
