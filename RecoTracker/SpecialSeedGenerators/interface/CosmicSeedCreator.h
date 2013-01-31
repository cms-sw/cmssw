#ifndef RecoTracker_TkSeedGenerator_CosmicSeedCreator_H
#define RecoTracker_TkSeedGenerator_CosmicSeedCreator_H

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
class FreeTrajectoryState;

class CosmicSeedCreator : public SeedCreator {

 public:
  CosmicSeedCreator( const edm::ParameterSet & extra )
    {
      maxseeds_ = extra.getParameter<int>("maxseeds");
    }

  virtual ~CosmicSeedCreator(){}

 protected:
  const TrajectorySeed * trajectorySeed(TrajectorySeedCollection & seedCollection,
					const SeedingHitSet & ordered,
					const TrackingRegion & region,
					const edm::EventSetup& es,
                                        const SeedComparitor *filter);
  
private:

  unsigned int maxseeds_;
};
#endif 
