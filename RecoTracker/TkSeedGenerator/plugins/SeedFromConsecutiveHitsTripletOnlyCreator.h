#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsTripletOnlyCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsTripletOnlyCreator_H

#include "SeedFromConsecutiveHitsCreator.h"

class dso_hidden SeedFromConsecutiveHitsTripletOnlyCreator GCC11_FINAL : public SeedFromConsecutiveHitsCreator {
public:

  SeedFromConsecutiveHitsTripletOnlyCreator( const edm::ParameterSet & cfg):
    SeedFromConsecutiveHitsCreator(cfg) { }

  virtual ~SeedFromConsecutiveHitsTripletOnlyCreator(){}

private:

  virtual bool initialKinematic(GlobalTrajectoryParameters & kine,
				const SeedingHitSet & hits) const;
 

};
#endif

