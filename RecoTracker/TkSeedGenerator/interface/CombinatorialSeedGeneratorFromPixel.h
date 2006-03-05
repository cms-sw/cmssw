#ifndef CombinatorialSeedGeneratorFromPixel_H
#define CombinatorialSeedGeneratorFromPixel_H

/** \class CombinatorialSeedGeneratorFromPixel
 *  A concrete regional seed generator providing seeds constructed 
 *  from combinations of hits in pairs of pixel layers 
 */
    
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
class PixelSeedLayerPairs;

class CombinatorialSeedGeneratorFromPixel : public SeedGeneratorFromLayerPairs {
 public:
  
  CombinatorialSeedGeneratorFromPixel(const edm::ParameterSet& conf);

  void init(const edm::EventSetup& c);
  vector<TrajectorySeed>  run(const edm::EventSetup& c);
 private:
  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
  // PixelSeedLayerPairs pixellayers;
};
#endif


