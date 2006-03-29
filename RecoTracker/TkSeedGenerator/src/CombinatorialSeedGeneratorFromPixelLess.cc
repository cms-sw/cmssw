#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixelLess.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/PixelLessSeedLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
void 
CombinatorialSeedGeneratorFromPixelLess::init(const SiStripRecHit2DMatchedLocalPosCollection &collmatch,
					  const SiStripRecHit2DLocalPosCollection &collrphi ,
					  const edm::EventSetup& iSetup)
{

   PixelLessSeedLayerPairs pixellesslayers;
   pixellesslayers.init(collmatch,collrphi,iSetup);
   initPairGenerator(&pixellesslayers,iSetup);
}

CombinatorialSeedGeneratorFromPixelLess::CombinatorialSeedGeneratorFromPixelLess(edm::ParameterSet const& conf): 
  conf_(conf)
{  

  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);

  edm::LogInfo("CombinatorialSeedGeneratorFromPixelLess")<<" PtMin of track is "<<ptmin<< 
    " The Radius of the cylinder for seeds is "<<originradius <<"cm" ;

}

void CombinatorialSeedGeneratorFromPixelLess::run(TrajectorySeedCollection &output,const edm::EventSetup& iSetup){
  seeds(output,iSetup,region);
}
