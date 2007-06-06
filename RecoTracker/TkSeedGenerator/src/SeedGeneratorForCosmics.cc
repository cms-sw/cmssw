#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorForCosmics.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CosmicLayerTriplets.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
void 
SeedGeneratorForCosmics::init(const SiStripRecHit2DCollection &collstereo,
			      const SiStripRecHit2DCollection &collrphi ,
			      const SiStripMatchedRecHit2DCollection &collmatched,
			      const edm::EventSetup& iSetup)
{
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  thePropagatorAl=    new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
  thePropagatorOp=    new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );
  theUpdator=       new KFUpdator();
  
  // get the transient builder
  //

  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;

  iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);
  TTTRHBuilder = theBuilder.product();
  LogDebug("CosmicSeedFinder")<<" Hits built with  "<<hitsforseeds<<" hits";
 
    CosmicLayerPairs cosmiclayers(geometry);

    cosmiclayers.init(collstereo,collrphi,collmatched,iSetup);
    thePairGenerator=new CosmicHitPairGenerator(cosmiclayers,iSetup);
    HitPairs.clear();
    if ((hitsforseeds=="pairs")||(hitsforseeds=="pairsandtriplets")){
      thePairGenerator->hitPairs(region,HitPairs,iSetup);
  }

    CosmicLayerTriplets cosmiclayers2;
    cosmiclayers2.init(collstereo,collrphi,collmatched,geometry,iSetup);
    theTripletGenerator=new CosmicHitTripletGenerator(cosmiclayers2,iSetup);
    HitTriplets.clear();
    if ((hitsforseeds=="triplets")||(hitsforseeds=="pairsandtriplets")){
      theTripletGenerator->hitTriplets(region,HitTriplets,iSetup);
    }
}

SeedGeneratorForCosmics::SeedGeneratorForCosmics(edm::ParameterSet const& conf): SeedGeneratorFromTrackingRegion(conf),
  conf_(conf)
{  

  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);
  hitsforseeds=conf_.getUntrackedParameter<std::string>("HitsForSeeds","pairs");
  edm::LogInfo("SeedGeneratorForCosmics")<<" PtMin of track is "<<ptmin<< 
    " The Radius of the cylinder for seeds is "<<originradius <<"cm" ;



}

void SeedGeneratorForCosmics::run(TrajectorySeedCollection &output,const edm::EventSetup& iSetup){
  seeds(output,iSetup,region);
  delete thePairGenerator;
}
void SeedGeneratorForCosmics::seeds(TrajectorySeedCollection &output,
				    const edm::EventSetup& iSetup,
				    const TrackingRegion& region){
  LogDebug("CosmicSeedFinder")<<"Number of triplets "<<HitTriplets.size();
  LogDebug("CosmicSeedFinder")<<"Number of pairs "<<HitPairs.size();

  for (uint it=0;it<HitTriplets.size();it++){
    GlobalPoint inner = tracker->idToDet(HitTriplets[it].inner().RecHit()->
					 geographicalId())->surface().
      toGlobal(HitTriplets[it].inner().RecHit()->localPosition());
    GlobalPoint middle = tracker->idToDet(HitTriplets[it].middle().RecHit()->
					  geographicalId())->surface().
      toGlobal(HitTriplets[it].middle().RecHit()->localPosition());
    GlobalPoint outer = tracker->idToDet(HitTriplets[it].outer().RecHit()->
					 geographicalId())->surface().
      toGlobal(HitTriplets[it].outer().RecHit()->localPosition());   

    TransientTrackingRecHit::ConstRecHitPointer outrhit=TTTRHBuilder->build(HitTriplets[it].outer().RecHit());
    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitTriplets[it].outer().RecHit()->clone());
    FastHelix helix(inner, middle, outer,iSetup);
    GlobalVector gv=helix.stateAtVertex().parameters().momentum();
    float ch=helix.stateAtVertex().parameters().charge();
    if (gv.y()>0){
      gv=-1.*gv;
      ch=-1.*ch;
    }

    GlobalTrajectoryParameters Gtp(outer,
				   gv,int(ch), 
				   &(*magfield));
    FreeTrajectoryState CosmicSeed(Gtp,
				   CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));  
    if((outer.y()-inner.y())>0){
      const TSOS outerState =
	thePropagatorAl->propagate(CosmicSeed,
				   tracker->idToDet(HitTriplets[it].outer().RecHit()->geographicalId())->surface());
      if ( outerState.isValid()) {
	LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	if ( outerUpdated.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	  
	  PTrajectoryStateOnDet *PTraj=  
	    transformer.persistentState(outerUpdated, HitTriplets[it].outer().RecHit()->geographicalId().rawId());
	  
	  TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,alongMomentum);
	  output.push_back(*trSeed);
	}
      }
    } else {
      const TSOS outerState =
	thePropagatorOp->propagate(CosmicSeed,
				   tracker->idToDet(HitTriplets[it].outer().RecHit()->geographicalId())->surface());
      if ( outerState.isValid()) {
	LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	if ( outerUpdated.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	  
	  PTrajectoryStateOnDet *PTraj=  
	    transformer.persistentState(outerUpdated, HitTriplets[it].outer().RecHit()->geographicalId().rawId());
	  
	  TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,oppositeToMomentum);
	  output.push_back(*trSeed);
	}
      }
    }
  }
  

  for(uint is=0;is<HitPairs.size();is++){

    
    GlobalPoint inner = tracker->idToDet(HitPairs[is].inner().RecHit()->geographicalId())->surface().toGlobal(HitPairs[is].inner().RecHit()->localPosition());
    GlobalPoint outer = tracker->idToDet(HitPairs[is].outer().RecHit()->geographicalId())->surface().toGlobal(HitPairs[is].outer().RecHit()->localPosition());
    
    LogDebug("CosmicSeedFinder") <<"inner point of the seed "<<inner <<" outer point of the seed "<<outer; 
    //RC const TransientTrackingRecHit* outrhit=TTTRHBuilder->build(HitPairs[is].outer().RecHit());  
    TransientTrackingRecHit::ConstRecHitPointer outrhit=TTTRHBuilder->build(HitPairs[is].outer().RecHit());

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[is].outer().RecHit()->clone());
    //    hits.push_back(HitPairs[is].inner()->clone());

    for (int i=0;i<2;i++){
      //FIRST STATE IS CALCULATED CONSIDERING THAT THE CHARGE CAN BE POSITIVE OR NEGATIVE
      int predsign=(2*i)-1;
      if((outer.y()-inner.y())>0){
	GlobalTrajectoryParameters Gtp(outer,
				       inner-outer,
				       predsign, 
				       &(*magfield));
	
	FreeTrajectoryState CosmicSeed(Gtp,
				       CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
	
	
	LogDebug("CosmicSeedFinder") << " FirstTSOS "<<CosmicSeed;
	//First propagation
	const TSOS outerState =
	  thePropagatorAl->propagate(CosmicSeed,
				     tracker->idToDet(HitPairs[is].outer().RecHit()->geographicalId())->surface());
	if ( outerState.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	  const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	  if ( outerUpdated.isValid()) {
	    LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	    
	    PTrajectoryStateOnDet *PTraj=  
	      transformer.persistentState(outerUpdated, HitPairs[is].outer().RecHit()->geographicalId().rawId());
	    
	    TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,alongMomentum);
	    output.push_back(*trSeed);
	    
	  }else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
	}else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";
      
      
      }
      else{
	GlobalTrajectoryParameters Gtp(outer,
				       outer-inner,
				       predsign, 
				       &(*magfield));
	FreeTrajectoryState CosmicSeed(Gtp,
				       CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
	LogDebug("CosmicSeedFinder") << " FirstTSOS "<<CosmicSeed;
	//First propagation
	const TSOS outerState =
	  thePropagatorAl->propagate(CosmicSeed,
				     tracker->idToDet(HitPairs[is].outer().RecHit()->geographicalId())->surface());
	if ( outerState.isValid()) {
	  
	  LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	  const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	  if ( outerUpdated.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	  PTrajectoryStateOnDet *PTraj=  
	    transformer.persistentState(outerUpdated, HitPairs[is].outer().RecHit()->geographicalId().rawId());
	  
	  TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,oppositeToMomentum);
	  output.push_back(*trSeed);
	
	  }else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
	}else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";
      }
      
    }
  }
}
