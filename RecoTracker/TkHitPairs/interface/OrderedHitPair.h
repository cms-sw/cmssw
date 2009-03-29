#ifndef OrderedHitPair_H
#define OrderedHitPair_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"


class OrderedHitPair : public SeedingHitSet {
public:

  typedef ctfseeding::SeedingHit OuterHit;
  typedef ctfseeding::SeedingHit InnerHit;

  typedef TransientTrackingRecHit::ConstRecHitPointer OuterRecHit; 
  typedef TransientTrackingRecHit::ConstRecHitPointer InnerRecHit; 

  OrderedHitPair( const InnerRecHit & ih, const OuterRecHit & oh) 
  {
    theRecHits.push_back(ih);
    theRecHits.push_back(oh);
  }

  OrderedHitPair( const InnerHit & ih, const OuterHit & oh)
  {
    add(ih);
    add(oh);
  }

  virtual ~OrderedHitPair() {}

  const InnerRecHit & inner() const { return theRecHits.front(); }
  const OuterRecHit & outer() const { return theRecHits.back(); } 
};

#endif

