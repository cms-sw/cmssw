#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorForCosmics.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
void 
SeedGeneratorForCosmics::init(const SiStripRecHit2DLocalPosCollection &collstereo,
			      const SiStripRecHit2DLocalPosCollection &collrphi ,
			      const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
			      const edm::EventSetup& iSetup)
{

  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  thePropagatorAl=    new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
  thePropagatorOp=    new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );
  theUpdator=       new KFUpdator();
  
  //
  // get the transient builder
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  iSetup.get<TrackingComponentsRecord>().get(builderName,theBuilder);
  TTTRHBuilder = theBuilder.product();

 CosmicLayerPairs cosmiclayers;
 cosmiclayers.init(collstereo,collrphi,collmatched,iSetup);
 thePairGenerator=new CosmicHitPairGenerator(cosmiclayers,iSetup);

}

SeedGeneratorForCosmics::SeedGeneratorForCosmics(edm::ParameterSet const& conf): SeedGeneratorFromTrackingRegion(conf),
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

  if(HitPairs.size()>0){


    stable_sort(HitPairs.begin(),HitPairs.end(),CompareHitPairsY(iSetup));
    
    
    GlobalPoint inner = tracker->idToDet(HitPairs[0].inner()->geographicalId())->surface().toGlobal(HitPairs[0].inner()->localPosition());
    GlobalPoint outer = tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface().toGlobal(HitPairs[0].outer()->localPosition());
  
  

    //    const TransientTrackingRecHit* outrhit=TTTRHBuilder->build(HitPairs[0].outer());
    const TransientTrackingRecHit* intrhit =TTTRHBuilder->build(HitPairs[0].inner());

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[0].outer()->clone());
    hits.push_back(HitPairs[0].inner()->clone());
    

 
    
    //Definition of the local trajectory parameters of the seed for cosmics
    GlobalTrajectoryParameters Gtp(outer,
				   inner-outer,
				   //				   10000,
				   1, &(*magfield));
    FreeTrajectoryState CosmicSeed(Gtp,
				   CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
    
 
    const TSOS innerState = 
      thePropagatorOp->propagate(CosmicSeed,
				 tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface());

    if (innerState.isValid()){
      const TSOS innerUpdated= theUpdator->update( innerState,*intrhit);
      if (innerUpdated.isValid()){
	
	
	PTrajectoryStateOnDet *PTraj=  
	  transformer.persistentState(innerUpdated, HitPairs[0].outer()->geographicalId().rawId());
	
	TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,oppositeToMomentum);
	output.push_back(*trSeed);

      }
    }
  }
}
