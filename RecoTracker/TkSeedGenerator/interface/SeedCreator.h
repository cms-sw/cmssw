#ifndef RecoTracker_TkSeedGenerator_SeedCreator_H
#define RecoTracker_TkSeedGenerator_SeedCreator_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

class TrackingRegion;
class SeedCreator;
class SeedingHitSet;
class SeedComparitor;

namespace edm { class Event; class EventSetup; }

class SeedCreator {
public:

  virtual ~SeedCreator(){}

  // make job
  virtual const TrajectorySeed *  trajectorySeed(TrajectorySeedCollection & seedCollection,
						 const SeedingHitSet & hits,
						 const TrackingRegion & region,
						 const edm::EventSetup& es,
                                                 const SeedComparitor *filter) = 0;
};
#endif 
