#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStraightLineCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsStraightLineCreator_H

#include "SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
class FreeTrajectoryState;

class SeedFromConsecutiveHitsStraightLineCreator GCC11_FINAL : public SeedFromConsecutiveHitsCreator {
public:

  SeedFromConsecutiveHitsStraightLineCreator( const edm::ParameterSet & cfg):
    SeedFromConsecutiveHitsCreator(cfg) { }

  virtual ~SeedFromConsecutiveHitsStraightLineCreator(){}

private:

  virtual bool initialKinematic(GlobalTrajectoryParameters & kine,
				const SeedingHitSet & hits) const;


};
#endif

