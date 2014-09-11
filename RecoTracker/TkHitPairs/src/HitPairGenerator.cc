#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"

HitPairGenerator::HitPairGenerator(unsigned int nSize) : localRA(nSize) {}

const OrderedHitPairs & HitPairGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  assert(thePairs.size()==0); assert(thePairs.capacity()==0);
  thePairs.reserve(localRA.upper());
  hitPairs(region, thePairs, ev, es);
  thePairs.shrink_to_fit();
  return thePairs;
}


void HitPairGenerator::clear() 
{
  localRA.update(thePairs.size());
  thePairs.clear(); thePairs.shrink_to_fit();
}

