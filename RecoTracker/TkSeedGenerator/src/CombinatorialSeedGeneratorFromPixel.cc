#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixel.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CombinatorialSeedGeneratorFromPixel::init(const SiPixelRecHitCollection &coll ,
					       const edm::EventSetup& iSetup)
{
  pixelLayers->init(coll,iSetup);
  initPairGenerator(pixelLayers,iSetup);
}

CombinatorialSeedGeneratorFromPixel::CombinatorialSeedGeneratorFromPixel(edm::ParameterSet const& conf)
  : SeedGeneratorFromLayerPairs(conf)
{  
  edm::ParameterSet conf_ = pSet();
  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);

  edm::LogInfo("CombinatorialSeedGeneratorFromPixel")<<" PtMin of track is "<<ptmin<< 
    " The Radius of the cylinder for seeds is "<<originradius <<"cm" ;

  pixelLayers = new PixelSeedLayerPairs();
}

void CombinatorialSeedGeneratorFromPixel::run(TrajectorySeedCollection &output,
    const edm::EventSetup& iSetup){
  seeds(output,iSetup,region);
}
