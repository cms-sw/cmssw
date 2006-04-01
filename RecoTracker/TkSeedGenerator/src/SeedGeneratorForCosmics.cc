#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorForCosmics.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
void 
SeedGeneratorForCosmics::init(const SiStripRecHit2DLocalPosCollection &collstereo,
			      const SiStripRecHit2DLocalPosCollection &collrphi ,
			      const edm::EventSetup& iSetup)
{

  CosmicLayerPairs cosmiclayers;
  cosmiclayers.init(collstereo,collrphi,iSetup);
  thePairGenerator=new CosmicHitPairGenerator(cosmiclayers,iSetup);
}

SeedGeneratorForCosmics::SeedGeneratorForCosmics(edm::ParameterSet const& conf): 
  conf_(conf)
{  

  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);

  edm::LogInfo("SeedGeneratorForCosmics")<<" PtMin of track is "<<ptmin<< 
    " The Radius of the cylinder for seeds is "<<originradius <<"cm" ;

}

void SeedGeneratorForCosmics::run(TrajectorySeedCollection &output,const edm::EventSetup& iSetup){
  seeds(output,iSetup,region);
}
void SeedGeneratorForCosmics::seeds(TrajectorySeedCollection &output,
				    const edm::EventSetup& iSetup,
				    const TrackingRegion& region){

  OrderedHitPairs HitPairs;
  thePairGenerator->hitPairs(region,HitPairs,iSetup);
  //MP
  GlobalError vtxerr( 100,0,100,0,0,100);

  SeedHitPairs::const_iterator ip;

  for (ip = HitPairs.begin(); ip != HitPairs.end(); ip++) {
 
    SeedFromConsecutiveHits *seedfromhits=
      new SeedFromConsecutiveHits( ip->outer(), ip->inner(),
				   region.origin(), vtxerr,iSetup);
    output.push_back(*(seedfromhits->TrajSeed()) );
  }

}
