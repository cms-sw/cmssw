#ifndef CombinatorialRegionalSeedGeneratorFromPixel_H
#define CombinatorialRegionalSeedGeneratorFromPixel_H

/** \class CombinatorialRegionalSeedGeneratorFromPixel
 *  A regional  seed generator providing seeds constructed 
 *  from combinations of hits in pairs of pixel layers 
 */
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"    
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

class PixelSeedLayerPairs;

class CombinatorialRegionalSeedGeneratorFromPixel : public SeedGeneratorFromLayerPairs {
 public:
  
  CombinatorialRegionalSeedGeneratorFromPixel(const edm::ParameterSet& conf);
  ~CombinatorialRegionalSeedGeneratorFromPixel(){delete pixelLayers;} 
  
  void init(const SiPixelRecHitCollection &coll, const edm::EventSetup& c);
  void  run(RectangularEtaPhiTrackingRegion& etaphiRegion, TrajectorySeedCollection &, const edm::EventSetup& c);
 private:
  //  edm::ParameterSet conf_;
  RectangularEtaPhiTrackingRegion region;
  PixelSeedLayerPairs* pixelLayers;
};
#endif


