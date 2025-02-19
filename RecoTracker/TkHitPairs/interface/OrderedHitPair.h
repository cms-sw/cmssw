#ifndef OrderedHitPair_H
#define OrderedHitPair_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"


class OrderedHitPair : public SeedingHitSet {
public:

  typedef SeedingHitSet::ConstRecHitPointer OuterRecHit; 
  typedef SeedingHitSet::ConstRecHitPointer InnerRecHit; 

  OrderedHitPair( const InnerRecHit & ih, const OuterRecHit & oh) : SeedingHitSet(ih, oh){}
 
  const InnerRecHit & inner() const { return get(0); }
  const OuterRecHit & outer() const { return get(1); } 
};

#endif

