#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include<iostream>
#include<typeinfo>

const OrderedMultiHits & MultiHitGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  //  std::cout << "MultiHitGenerator cache b " << cache.size() << std::endl;

  theHitSets.reserve(localRA.upper());
  hitSets(region, theHitSets, ev, es);
  //  std::cout << "MultiHitGenerator cache	a " << cache.size() << std::endl;
  theHitSets.shrink_to_fit();
  localRA.update(theHitSets.size());
  return theHitSets;
}

void MultiHitGenerator::clear() 
{
   theHitSets.clear(); theHitSets.shrink_to_fit();
  //std::cout << "MultiHitGenerator " << typeid(*this).name()
  //          <<" cache c " << cache.size() << ' ' << cache.capacity() << std::endl;
  cache.clear();
}

