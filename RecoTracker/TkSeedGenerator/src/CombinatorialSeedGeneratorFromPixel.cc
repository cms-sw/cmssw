#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixel.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"

void CombinatorialSeedGeneratorFromPixel::init(const edm::EventSetup& iSetup)
{
  PixelSeedLayerPairs pixellayers;
  pixellayers.init(iSetup);
  initPairGenerator(&pixellayers,iSetup);
}

CombinatorialSeedGeneratorFromPixel::CombinatorialSeedGeneratorFromPixel(edm::ParameterSet const& conf): 
  conf_(conf)
{  
  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);
}

 vector <TrajectorySeed>  CombinatorialSeedGeneratorFromPixel::run(const edm::EventSetup& iSetup){
   return seeds(iSetup,region);
}

// CombinatorialSeedGeneratorFromPixel::CombinatorialSeedGeneratorFromPixel(
//   const TrackingRegionFactory& regionFactory)
//   : SeedGeneratorFromLayerPairs(0,regionFactory)
// { init(); }

// CombinatorialSeedGeneratorFromPixel::CombinatorialSeedGeneratorFromPixel(
//   const TrackingRegion& region)
//   : SeedGeneratorFromLayerPairs(0, region)
// { 
// init(); }

// CombinatorialSeedGeneratorFromPixel::CombinatorialSeedGeneratorFromPixel(
//   float ptMin, float originRadius, float originHalfLength, float originZPos) 
//   : SeedGeneratorFromLayerPairs(0, 
//       ptMin, originRadius, originHalfLength, originZPos)
// { init(); }

