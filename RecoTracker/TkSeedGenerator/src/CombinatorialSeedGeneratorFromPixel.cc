#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixel.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

void CombinatorialSeedGeneratorFromPixel::init(const edm::EventSetup& iSetup)
{
 
  const  PixelSeedLayerPairs *pixellayers;

  //  pixellayers->init(iSetup);
  // PixelSeedLayerPairs pixel;
  //  initPairGenerator(&pixel);
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

void CombinatorialSeedGeneratorFromPixel::run(){
  cout<<"fg"<<endl;
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

