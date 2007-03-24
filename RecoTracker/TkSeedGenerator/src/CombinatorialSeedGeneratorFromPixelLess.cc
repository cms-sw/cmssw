#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixelLess.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/PixelLessSeedLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void 
CombinatorialSeedGeneratorFromPixelLess::init(const SiStripMatchedRecHit2DCollection &collmatch,
					      const SiStripRecHit2DCollection &collstereo ,
					      const SiStripRecHit2DCollection &collrphi ,
					      const edm::EventSetup& iSetup)
{
  stripLayers->init(collmatch,collstereo,collrphi,iSetup);
  initPairGenerator(stripLayers,iSetup);
}

CombinatorialSeedGeneratorFromPixelLess::CombinatorialSeedGeneratorFromPixelLess(edm::ParameterSet const& conf): SeedGeneratorFromLayerPairs(conf)
{  

  edm::ParameterSet conf_ = pSet();
  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);

  edm::LogInfo("CombinatorialSeedGeneratorFromPixelLess")<<" PtMin of track is "<<ptmin<< 
    " The Radius of the cylinder for seeds is "<<originradius <<"cm" ;

  stripLayers = new PixelLessSeedLayerPairs();
}

void CombinatorialSeedGeneratorFromPixelLess::run(TrajectorySeedCollection &output, const edm::EventSetup& iSetup){
  seeds(output,iSetup,region);
}
