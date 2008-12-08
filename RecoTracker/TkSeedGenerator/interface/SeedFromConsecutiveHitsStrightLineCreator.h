#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStrightLineCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStrightLineCreator_H

#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
class FreeTrajectoryState;

class SeedFromConsecutiveHitsStrightLineCreator : public SeedFromConsecutiveHitsCreator {
public:

  SeedFromConsecutiveHitsStrightLineCreator( const edm::ParameterSet & cfg):
    SeedFromConsecutiveHitsCreator(cfg) { }

  virtual ~SeedFromConsecutiveHitsStrightLineCreator(){}

protected:

  virtual GlobalTrajectoryParameters initialKinematic(
      const SeedingHitSet::Hits & hits,
      const TrackingRegion & region,
      const edm::EventSetup& es) const;

};
#endif

