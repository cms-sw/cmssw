#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

namespace ctfseeding {

// TODO: See if these definitions could be moved to somewhere else?
// This class/structure is mainly a historical relic.
class SeedingLayer {
public:
  enum Side { Barrel = 0, NegEndcap =1,  PosEndcap = 2 }; 
  using TkHit = BaseTrackerRecHit;
  using TkHitRef = BaseTrackerRecHit const &;
  using HitPointer = mayown_ptr<BaseTrackerRecHit>;
  using Hits=std::vector<HitPointer>;
};

}
#endif
