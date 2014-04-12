#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include<iostream>
#include<typeinfo>

MultiHitGenerator::MultiHitGenerator(unsigned int nSize)
{
  theHitSets.reserve(nSize);
}

const OrderedMultiHits & MultiHitGenerator::run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  //  std::cout << "MultiHitGenerator cache b " << cache.size() << std::endl;
  theHitSets.clear();
  hitSets(region, theHitSets, ev, es);
  //  std::cout << "MultiHitGenerator cache	a " << cache.size() << std::endl;
  return theHitSets;
}

void MultiHitGenerator::clear() 
{
  //std::cout << "MultiHitGenerator " << typeid(*this).name()
  //          <<" cache c " << cache.size() << ' ' << cache.capacity() << std::endl;
  theHitSets.clear();
  cache.clear();
}

