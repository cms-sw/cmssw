#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class OrderedHitsGenerator;
namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits {
public:

  //ctor,  ParameterSet is passed temporary!!!!
  SeedGeneratorFromRegionHits(OrderedHitsGenerator *, const edm::ParameterSet &);

  //dtor
  ~SeedGeneratorFromRegionHits();

  // make job
  void run(TrajectorySeedCollection & seedCollection, const TrackingRegion & region, 
      const edm::Event& ev, const edm::EventSetup& es);
 
private:
  OrderedHitsGenerator * theHitsGenerator;
  edm::ParameterSet theConfig; //  temporary 

};
#endif 
