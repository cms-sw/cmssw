#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include<iostream>
#include<typeinfo>

const OrderedMultiHits & MultiHitGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  //  std::cout << "MultiHitGenerator cache b " << cache.size() << std::endl;

  decltype(theHitSets) tmp; tmp.reserve(localRA.upper()); tmp.swap(theHitSets);
  hitSets(region, theHitSets, ev, es);
  //  std::cout << "MultiHitGenerator cache	a " << cache.size() << std::endl;
  theHitSets.shrink_to_fit();
  localRA.update(theHitSets.size());
  return theHitSets;
}

void MultiHitGenerator::clear() 
{
  decltype(theHitSets) tmp; tmp.swap(theHitSets);
  //std::cout << "MultiHitGenerator " << typeid(*this).name()
  //          <<" cache c " << cache.size() << ' ' << cache.capacity() << std::endl;
  cache.clear();
}

