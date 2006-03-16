#ifndef CombinatorialSeedGeneratorFromPixel_H
#define CombinatorialSeedGeneratorFromPixel_H

/** \class CombinatorialSeedGeneratorFromPixel
 *  A concrete regional seed generator providing seeds constructed 
 *  from combinations of hits in pairs of pixel layers 
 */
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"    
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

class PixelSeedLayerPairs;

class CombinatorialSeedGeneratorFromPixel : public SeedGeneratorFromLayerPairs {
 public:
  
  CombinatorialSeedGeneratorFromPixel(const edm::ParameterSet& conf);

  void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& c);
void  run(TrajectorySeedCollection &,const edm::EventSetup& c);
 private:
  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
 
};
#endif


