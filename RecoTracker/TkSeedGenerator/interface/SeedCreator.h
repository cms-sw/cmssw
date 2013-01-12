#ifndef RecoTracker_TkSeedGenerator_SeedCreator_H
#define RecoTracker_TkSeedGenerator_SeedCreator_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class TrackingRegion;
class SeedingHitSet;
class SeedComparitor;

namespace edm { class Event; class EventSetup; }

class SeedCreator {
public:

  virtual ~SeedCreator(){}

  // initialize the "event dependent state"
  virtual void init(const TrackingRegion & region,
		    const edm::EventSetup& es,
		    const SeedComparitor *filter) = 0;

  // make job 
  // fill seedCollection with the "TrajectorySeed"
  virtual void makeSeed(TrajectorySeedCollection & seedCollection,
			const SeedingHitSet & hits) = 0;
};
#endif 
