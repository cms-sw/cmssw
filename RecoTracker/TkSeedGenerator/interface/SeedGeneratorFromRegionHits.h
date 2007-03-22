#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegionProducer;
class OrderedHitsGenerator;
namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits {
public:

  //ctor,  ParameterSet is passed temporary!!!!
  SeedGeneratorFromRegionHits(TrackingRegionProducer* , OrderedHitsGenerator *, const edm::ParameterSet &);

  //dtor
  ~SeedGeneratorFromRegionHits();

  // make job
  void run(TrajectorySeedCollection & seedCollection, edm::Event& ev, const edm::EventSetup& es);
 
private:
  TrackingRegionProducer * theRegionProducer;
  OrderedHitsGenerator * theHitsGenerator;
  edm::ParameterSet theConfig; //  temporary 

};
#endif 
