#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"

MultiHitGenerator::MultiHitGenerator(unsigned int nSize)
{
  theHitSets.reserve(nSize);
}

const OrderedMultiHits & MultiHitGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  theHitSets.clear();
  hitSets(region, theHitSets, ev, es);
  return theHitSets;
}

void MultiHitGenerator::clear() 
{
  theHitSets.clear();
}

