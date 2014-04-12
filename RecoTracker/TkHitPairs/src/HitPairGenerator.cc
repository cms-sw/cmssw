#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"

HitPairGenerator::HitPairGenerator(unsigned int nSize) : m_capacity(nSize)
{
  thePairs.reserve(nSize);
}

const OrderedHitPairs & HitPairGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  thePairs.clear();
  hitPairs(region, thePairs, ev, es);
  return thePairs;
}
