#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"

HitPairGenerator::HitPairGenerator(unsigned int nSize) : localRA(nSize)
{
//  thePairs.reserve(nSize);
}

const OrderedHitPairs & HitPairGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  //thePairs.clear();
  OrderedHitPairs tmp; tmp.reserve(localRA.upper()); tmp.swap(thePairs);
  hitPairs(region, thePairs, ev, es);
  thePairs.shrink_to_fit();
  return thePairs;
}
