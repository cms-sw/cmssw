#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStraightLineCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStraightLineCreator_H

#include "SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
class FreeTrajectoryState;

class SeedFromConsecutiveHitsStraightLineCreator : public SeedFromConsecutiveHitsCreator {
public:

  SeedFromConsecutiveHitsStraightLineCreator( const edm::ParameterSet & cfg):
    SeedFromConsecutiveHitsCreator(cfg) { }

  virtual ~SeedFromConsecutiveHitsStraightLineCreator(){}

protected:

  virtual GlobalTrajectoryParameters initialKinematic(
      const SeedingHitSet & hits,
      const TrackingRegion & region,
      const edm::EventSetup& es,
      const SeedComparitor *filter,
      bool                 &passesFilter) const;

};
#endif

