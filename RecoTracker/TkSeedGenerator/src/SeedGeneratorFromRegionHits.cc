#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"




SeedGeneratorFromRegionHits::SeedGeneratorFromRegionHits(
    OrderedHitsGenerator *ohg, SeedComparitor* asc, SeedCreator* asp)
  : theHitsGenerator(ohg), theComparitor(asc), theSeedCreator(asp)
{ }


void SeedGeneratorFromRegionHits::run(TrajectorySeedCollection & seedCollection, 
    const TrackingRegion & region, const edm::Event& ev, const edm::EventSetup& es)
{
  if (theComparitor) theComparitor->init(ev, es);
  theSeedCreator->init(region, es, theComparitor.get());
  const OrderedSeedingHits & hitss = theHitsGenerator->run(region, ev, es);

  unsigned int nHitss =  hitss.size();
  if (seedCollection.empty()) seedCollection.reserve(nHitss); // don't do multiple reserves in the case of multiple regions: it would make things even worse
                                                              // as it will cause N re-allocations instead of the normal log(N)/log(2)
  for (unsigned int iHits = 0; iHits < nHitss; ++iHits) { 
    const SeedingHitSet & hits =  hitss[iHits];
    if (!theComparitor || theComparitor->compatible(hits, region) ) {
      theSeedCreator->makeSeed(seedCollection, hits);
    }
  }
  theHitsGenerator->clear();
}
