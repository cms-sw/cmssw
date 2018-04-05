#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include<iostream>
#include<typeinfo>

const OrderedMultiHits & MultiHitGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  theHitSets.clear(); // called multiple time for the same seed collection
  theHitSets.reserve(localRA.upper());
  hitSets(region, theHitSets, ev, es);
  theHitSets.shrink_to_fit();
  localRA.update(theHitSets.size());
  return theHitSets;
}

void MultiHitGenerator::clear() 
{
  theHitSets.clear(); theHitSets.shrink_to_fit();
}

