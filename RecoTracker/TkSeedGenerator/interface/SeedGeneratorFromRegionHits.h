#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include <string>

class TrackingRegion;
class OrderedHitsGenerator;
class SeedComparitor;
class SeedCreator;
 
namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits {
public:

  SeedGeneratorFromRegionHits(
      OrderedHitsGenerator * aGenerator, 
      SeedComparitor * aComparitor = 0,
      SeedCreator * aSeedCreator = 0
    );


  //dtor
  ~SeedGeneratorFromRegionHits();

  // make job
  void run(TrajectorySeedCollection & seedCollection, const TrackingRegion & region, 
      const edm::Event& ev, const edm::EventSetup& es);
 
private:
  OrderedHitsGenerator * theHitsGenerator;
  SeedComparitor * theComparitor;
  SeedCreator * theSeedCreator;
};
#endif 
