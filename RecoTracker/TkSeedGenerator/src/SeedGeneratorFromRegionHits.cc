#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SeedFromConsecutiveHits.h"

#include <vector>

template <class T> T sqr( T t) {return t*t;}

SeedGeneratorFromRegionHits::SeedGeneratorFromRegionHits(
  TrackingRegionProducer* trp, OrderedHitsGenerator *ohg, const edm::ParameterSet & cfg)
  : theRegionProducer(trp), theHitsGenerator(ohg), theConfig(cfg)
{ }

SeedGeneratorFromRegionHits::~SeedGeneratorFromRegionHits()
{
  delete theRegionProducer;
  delete theHitsGenerator;
}

void SeedGeneratorFromRegionHits::run(TrajectorySeedCollection & seedCollection, edm::Event& ev, const edm::EventSetup& es)
{

  typedef std::vector<TrackingRegion* > Regions;
  Regions regions = theRegionProducer->regions(ev,es);
  for (Regions::const_iterator ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;
    std::cout << region.name() << std::endl;

    const OrderedSeedingHits & hitss = theHitsGenerator->run(region, ev, es);

    std::cout <<" size of hits is: " << hitss.size() << std::endl;

    GlobalError vtxerr( sqr(region.originRBound()), 0, sqr(region.originRBound()),
                                                 0, 0, sqr(region.originZBound()));

    std::cout<<" origin: "<<region.origin()<<std::endl;

    unsigned int nHitss =  hitss.size();
    for (unsigned int iHits = 0; iHits < nHitss; ++iHits) { 
      SeedFromConsecutiveHits seedfromhits( hitss[iHits], region.origin(), vtxerr, es, theConfig);
      if(seedfromhits.isValid()) seedCollection.push_back( seedfromhits.TrajSeed() );
    }

  }

}


