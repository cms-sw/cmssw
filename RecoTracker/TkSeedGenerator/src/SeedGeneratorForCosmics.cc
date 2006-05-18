#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorForCosmics.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
void 
SeedGeneratorForCosmics::init(const SiStripRecHit2DLocalPosCollection &collstereo,
			      const SiStripRecHit2DLocalPosCollection &collrphi ,
			      const edm::EventSetup& iSetup)
{
 
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  thePropagatorAl=    new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
  thePropagatorOp=    new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );
  theUpdator=       new KFUpdator();
  TTTRHBuilder= new TkTransientTrackingRecHitBuilder((tracker.product()));
  
 
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

  if(HitPairs.size()>0){


    stable_sort(HitPairs.begin(),HitPairs.end(),CompareHitPairsY(iSetup));
    
    
    GlobalPoint inner = tracker->idToDet(HitPairs[0].inner()->geographicalId())->surface().toGlobal(HitPairs[0].inner()->localPosition());
    GlobalPoint outer = tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface().toGlobal(HitPairs[0].outer()->localPosition());
  
  

    const TransientTrackingRecHit* outrhit=TTTRHBuilder->build(HitPairs[0].outer());
    const TransientTrackingRecHit* intrhit =TTTRHBuilder->build(HitPairs[0].inner());

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[0].outer()->clone());
    hits.push_back(HitPairs[0].inner()->clone());
    


    LocalTrajectoryError LocErr(30*AlgebraicSymMatrix(5,1));
  
 
   
    //After building the first state
    //two options are considered 
    //1)if the hits are on the top of the tracker --> propagationDirection= alongMomentum
    //2)if the hits are on the bottom of the tracker --> propagationDirection= oppositeToMomentum
    if(outer.y()>0){
      
      //Definition of the local trajectory parameters of the seed for cosmics
      GlobalTrajectoryParameters Gtp(outer,
				     inner-outer,
				     10000,
				     1, &(*magfield));
      FreeTrajectoryState CosmicSeed(Gtp,
				     CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
 
   //    LocalVector lp=tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface().toLocal(inner-outer);
      
//       LocalTrajectoryParameters LocPar(0.,
// 				       lp.x()/lp.z(),
// 				       lp.y()/lp.z(),
// 				       HitPairs[0].outer()->localPosition().x(),
// 				       HitPairs[0].outer()->localPosition().y(),
// 				       lp.z()/abs(lp.z()));
//       TSOS CosmicSeed( LocPar,
// 		       LocErr,
// 		       tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface(),
// 		       &(*magfield));

     //First propagation
      const TSOS outerState =
	thePropagatorAl->propagate(CosmicSeed,
				   tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface());
      if ( !outerState.isValid()) 
	edm::LogError("Propagation") << " SeedForCosmics first propagation failed ";
      
      const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
      //Second propagation
      const TSOS innerState = 
	thePropagatorAl->propagate(outerUpdated,
				   tracker->idToDet(HitPairs[0].inner()->geographicalId())->surface());
      if ( !innerState.isValid()) 
	edm::LogError("Propagation") << " SeedForCosmics first propagation failed ";
      const TSOS innerUpdated= theUpdator->update( outerState,*intrhit);
   
      PTrajectoryStateOnDet *PTraj=  
	transformer.persistentState(innerUpdated, HitPairs[0].outer()->geographicalId().rawId());
      
      TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,alongMomentum);
      output.push_back(*trSeed);
    }
    else{
      //Definition of the local trajectory parameters of the seed for cosmics
      LocalVector lp=tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface().toLocal(outer-inner);
      
      LocalTrajectoryParameters LocPar(0.,
				       lp.x()/lp.z(),
				       lp.y()/lp.z(),
				       HitPairs[0].outer()->localPosition().x(),
				       HitPairs[0].outer()->localPosition().y(),
				       lp.z()/abs(lp.z()));
      TSOS CosmicSeed( LocPar,
		       LocErr,
		       tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface(),
		       &(*magfield));

  //     GlobalTrajectoryParameters Gtp(outer,
// 				     outer-inner,
// 				     1, &(*magfield));
//       FreeTrajectoryState CosmicSeed(Gtp,
// 				     CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
      //First propagation
      const TSOS outerState =
	thePropagatorOp->propagate(CosmicSeed,
				   tracker->idToDet(HitPairs[0].outer()->geographicalId())->surface());
    
       if ( !outerState.isValid()) 
	 edm::LogError("Propagation") << " SeedForCosmics first propagation failed ";
  
       const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
       //Second propagation
       const TSOS innerState = 
	 thePropagatorOp->propagate(outerUpdated,
				   tracker->idToDet(HitPairs[0].inner()->geographicalId())->surface());
       if ( !innerState.isValid()) 
	 edm::LogError("Propagation") << " SeedForCosmics first propagation failed ";

       const TSOS innerUpdated= theUpdator->update( outerState,*intrhit);

       PTrajectoryStateOnDet *PTraj=  
	 transformer.persistentState(innerUpdated, HitPairs[0].outer()->geographicalId().rawId());
       
       TrajectorySeed *trSeed=new TrajectorySeed(*PTraj,hits,oppositeToMomentum);
       output.push_back(*trSeed);  
    }


 
  }

}
