#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"


#include <memory>

class TrackingRegion;
 
namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits {
public:

  SeedGeneratorFromRegionHits(
      OrderedHitsGenerator * aGenerator, 
      SeedComparitor * aComparitor = 0,
      SeedCreator * aSeedCreator = 0
    );


  // make job
  void run(TrajectorySeedCollection & seedCollection, const TrackingRegion & region, 
	   const edm::Event& ev, const edm::EventSetup& es);
 
private:
  std::unique_ptr<OrderedHitsGenerator> theHitsGenerator;
  std::unique_ptr<SeedComparitor> theComparitor;
  std::unique_ptr<SeedCreator> theSeedCreator;
};
#endif 
