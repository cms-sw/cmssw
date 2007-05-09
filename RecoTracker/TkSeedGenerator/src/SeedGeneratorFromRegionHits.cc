#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"

#include <vector>

template <class T> T sqr( T t) {return t*t;}

SeedGeneratorFromRegionHits::SeedGeneratorFromRegionHits(
  OrderedHitsGenerator *ohg, const edm::ParameterSet & cfg,
  SeedComparitor * asc)
  : theHitsGenerator(ohg), theConfig(cfg), theComparitor(asc)
{ }

SeedGeneratorFromRegionHits::~SeedGeneratorFromRegionHits()
{
  delete theHitsGenerator;
  delete theComparitor;
}

void SeedGeneratorFromRegionHits::run(TrajectorySeedCollection & seedCollection, 
    const TrackingRegion & region, const edm::Event& ev, const edm::EventSetup& es)
{

  const OrderedSeedingHits & hitss = theHitsGenerator->run(region, ev, es);


  GlobalError vtxerr( sqr(region.originRBound()), 0, sqr(region.originRBound()),
                                               0, 0, sqr(region.originZBound()));

  unsigned int nHitss =  hitss.size();
  for (unsigned int iHits = 0; iHits < nHitss; ++iHits) { 
    const SeedingHitSet & hits =  hitss[iHits];
    if (!theComparitor || theComparitor->compatible( hits ) ) {
      SeedFromConsecutiveHits seedfromhits( hits, region.origin(), vtxerr, es);
      if(seedfromhits.isValid()) {
        seedCollection.push_back( seedfromhits.TrajSeed() );
      }
    }
  }
}
