#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"

#include <vector>

template <class T> T sqr( T t) {return t*t;}

SeedGeneratorFromRegionHits::SeedGeneratorFromRegionHits(
  OrderedHitsGenerator *ohg, const edm::ParameterSet & cfg)
  : theHitsGenerator(ohg), theConfig(cfg)
{ }

SeedGeneratorFromRegionHits::~SeedGeneratorFromRegionHits()
{
  delete theHitsGenerator;
}

void SeedGeneratorFromRegionHits::run(TrajectorySeedCollection & seedCollection, 
    const TrackingRegion & region, const edm::Event& ev, const edm::EventSetup& es)
{

  const OrderedSeedingHits & hitss = theHitsGenerator->run(region, ev, es);

  std::cout <<" size of hits is: " << hitss.size() << std::endl;

  GlobalError vtxerr( sqr(region.originRBound()), 0, sqr(region.originRBound()),
                                               0, 0, sqr(region.originZBound()));

  unsigned int nHitss =  hitss.size();
  for (unsigned int iHits = 0; iHits < nHitss; ++iHits) { 
    SeedFromConsecutiveHits seedfromhits( hitss[iHits], region.origin(), vtxerr, es, theConfig);
    if(seedfromhits.isValid()) seedCollection.push_back( seedfromhits.TrajSeed() );
  }
}
