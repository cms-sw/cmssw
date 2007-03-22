#ifndef OrderedHitPair_H
#define OrderedHitPair_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"


class OrderedHitPair : public SeedingHitSet {
public:

  typedef ctfseeding::SeedingHit OuterHit;
  typedef ctfseeding::SeedingHit InnerHit;

  OrderedHitPair( const InnerHit & ih, const OuterHit & oh)
  {
    theHits.push_back(ih);
    theHits.push_back(oh);
  }

  virtual ~OrderedHitPair() {}

  const InnerHit & inner() const { return theHits.front(); }
  const OuterHit & outer() const { return theHits.back(); } 
};

#endif

