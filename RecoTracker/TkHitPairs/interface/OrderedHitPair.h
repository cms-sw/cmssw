#ifndef OrderedHitPair_H
#define OrderedHitPair_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"


class OrderedHitPair : public SeedingHitSet {
public:

  typedef TransientTrackingRecHit::ConstRecHitPointer OuterRecHit; 
  typedef TransientTrackingRecHit::ConstRecHitPointer InnerRecHit; 

  OrderedHitPair( const InnerRecHit & ih, const OuterRecHit & oh) 
  {
    theRecHits.reserve(2);
    theRecHits.push_back(ih);
    theRecHits.push_back(oh);
  }

  virtual ~OrderedHitPair() {}

  const InnerRecHit & inner() const { return theRecHits.front(); }
  const OuterRecHit & outer() const { return theRecHits.back(); } 
};

#endif

