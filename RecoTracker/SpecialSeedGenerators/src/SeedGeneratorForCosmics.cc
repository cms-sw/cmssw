#include "RecoTracker/SpecialSeedGenerators/interface/SeedGeneratorForCosmics.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CosmicLayerTriplets.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

SeedGeneratorForCosmics::SeedGeneratorForCosmics(edm::ParameterSet const& conf):
  conf_(conf),
  maxSeeds_(conf.getParameter<int32_t>("maxSeeds"))
{  

  float ptmin=conf_.getParameter<double>("ptMin");
  float originradius=conf_.getParameter<double>("originRadius");
  float halflength=conf_.getParameter<double>("originHalfLength");
  float originz=conf_.getParameter<double>("originZPosition");
  seedpt = conf_.getParameter<double>("SeedPt");

  builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");
  region=GlobalTrackingRegion(ptmin,originradius,
 			      halflength,originz);
  hitsforseeds=conf_.getUntrackedParameter<std::string>("HitsForSeeds","pairs");
  edm::LogInfo("SeedGeneratorForCosmics")<<" PtMin of track is "<<ptmin<< 
    " The Radius of the cylinder for seeds is "<<originradius <<"cm"  << " The set Seed Momentum" <<  seedpt;

  //***top-bottom
  positiveYOnly=conf_.getParameter<bool>("PositiveYOnly");
  negativeYOnly=conf_.getParameter<bool>("NegativeYOnly");
  //***


}

void SeedGeneratorForCosmics::run(TrajectorySeedCollection &output,const edm::EventSetup& iSetup){
  seeds(output,iSetup,region);
  delete thePairGenerator;
  delete theTripletGenerator;
  delete thePropagatorAl;
  delete thePropagatorOp;
  delete theUpdator;
}
bool SeedGeneratorForCosmics::seeds(TrajectorySeedCollection &output,
				    const edm::EventSetup& iSetup,
				    const TrackingRegion& region){
  LogDebug("CosmicSeedFinder")<<"Number of triplets "<<HitTriplets.size();
  LogDebug("CosmicSeedFinder")<<"Number of pairs "<<HitPairs.size();

  for (unsigned int it=0;it<HitTriplets.size();it++){
    
    //const TrackingRecHit *hit = &(
    //    const TrackingRecHit* hit = it->hits();

  //   GlobalPoint inner = tracker->idToDet(HitTriplets[it].inner().RecHit()->
// 					 geographicalId())->surface().
//       toGlobal(HitTriplets[it].inner().RecHit()->localPosition());
    //const TrackingRecHit  *innerhit =  &(*HitTriplets[it].inner());
    //const TrackingRecHit  *middlehit =  &(*HitTriplets[it].middle());

    GlobalPoint inner = tracker->idToDet((*(HitTriplets[it].inner())).geographicalId())->surface().
      toGlobal((*(HitTriplets[it].inner())).localPosition());

    GlobalPoint middle = tracker->idToDet((*(HitTriplets[it].middle())).geographicalId())->surface().
      toGlobal((*(HitTriplets[it].middle())).localPosition());

    GlobalPoint outer = tracker->idToDet((*(HitTriplets[it].outer())).geographicalId())->surface().
      toGlobal((*(HitTriplets[it].outer())).localPosition());   

    // SeedingHitSet::ConstRecHitPointer outrhit=TTTRHBuilder->build(HitPairs[is].outer())

    SeedingHitSet::ConstRecHitPointer outrhit= HitTriplets[it].outer();
    //***top-bottom
    SeedingHitSet::ConstRecHitPointer innrhit = HitTriplets[it].inner();
    if (positiveYOnly && (outrhit->globalPosition().y()<0 || innrhit->globalPosition().y()<0
			  || outrhit->globalPosition().y() < innrhit->globalPosition().y()
			  ) ) continue;
    if (negativeYOnly && (outrhit->globalPosition().y()>0 || innrhit->globalPosition().y()>0
			  || outrhit->globalPosition().y() > innrhit->globalPosition().y()
			  ) ) continue;
    //***

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitTriplets[it].outer()->hit()->clone());
    FastHelix helix(inner, middle, outer, magfield->nominalValue(), &(*magfield));
    GlobalVector gv=helix.stateAtVertex().momentum();
    float ch=helix.stateAtVertex().charge();
    float Mom = sqrt( gv.x()*gv.x() + gv.y()*gv.y() + gv.z()*gv.z() ); 
    if(Mom > 1000 || std::isnan(Mom))  continue;   // ChangedByDaniele 

    if (gv.y()>0){
      gv=-1.*gv;
      ch=-1.*ch;
    }

    GlobalTrajectoryParameters Gtp(outer,
				   gv,int(ch), 
				   &(*magfield));
    FreeTrajectoryState CosmicSeed(Gtp,
				   CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));  
    if((outer.y()-inner.y())>0){
      const TSOS outerState =
	thePropagatorAl->propagate(CosmicSeed,
				   tracker->idToDet((*(HitTriplets[it].outer())).geographicalId())->surface());
      if ( outerState.isValid()) {
	LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	if ( outerUpdated.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	  
         output.push_back(TrajectorySeed(trajectoryStateTransform::persistentState(outerUpdated,(*(HitTriplets[it].outer())).geographicalId().rawId())
         ,hits,alongMomentum));

          if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
            edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
            output.clear(); 
            return false;
          }
	}
      }
    } else {
      const TSOS outerState =
	thePropagatorOp->propagate(CosmicSeed,
				   tracker->idToDet((*(HitTriplets[it].outer())).geographicalId())->surface());
      if ( outerState.isValid()) {
	LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	if ( outerUpdated.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	  
	  output.push_back(TrajectorySeed(trajectoryStateTransform::persistentState(outerUpdated, 
                            (*(HitTriplets[it].outer())).geographicalId().rawId()),hits,oppositeToMomentum));

          if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
            edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
            output.clear(); 
            return false;
          }
	}
      }
    }
  }
  

  for(unsigned int is=0;is<HitPairs.size();is++){

    

    GlobalPoint inner = tracker->idToDet((*(HitPairs[is].inner())).geographicalId())->surface().toGlobal((*(HitPairs[is].inner())).localPosition());
    GlobalPoint outer = tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface().toGlobal((*(HitPairs[is].outer())).localPosition());
    
    LogDebug("CosmicSeedFinder") <<"inner point of the seed "<<inner <<" outer point of the seed "<<outer; 
    //RC const TransientTrackingRecHit* outrhit=TTTRHBuilder->build(HitPairs[is].outer().RecHit());  
    SeedingHitSet::ConstRecHitPointer outrhit = HitPairs[is].outer();
    //***top-bottom
    SeedingHitSet::ConstRecHitPointer innrhit = HitPairs[is].inner();
    if (positiveYOnly && (outrhit->globalPosition().y()<0 || innrhit->globalPosition().y()<0
			  || outrhit->globalPosition().y() < innrhit->globalPosition().y()
			  ) ) continue;
    if (negativeYOnly && (outrhit->globalPosition().y()>0 || innrhit->globalPosition().y()>0
			  || outrhit->globalPosition().y() > innrhit->globalPosition().y()
			  ) ) continue;
    //***

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[is].outer()->hit()->clone());
    //    hits.push_back(HitPairs[is].inner()->clone());

    for (int i=0;i<2;i++){
      //FIRST STATE IS CALCULATED CONSIDERING THAT THE CHARGE CAN BE POSITIVE OR NEGATIVE
      int predsign=(2*i)-1;
      if((outer.y()-inner.y())>0){
	GlobalTrajectoryParameters Gtp(outer,
				       (inner-outer)*(seedpt/(inner-outer).mag()),
				       predsign, 
				       &(*magfield));
	
	FreeTrajectoryState CosmicSeed(Gtp,
				       CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
	
	
	LogDebug("CosmicSeedFinder") << " FirstTSOS "<<CosmicSeed;
	//First propagation
	const TSOS outerState =
	  thePropagatorAl->propagate(CosmicSeed,
				     tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface());
	if ( outerState.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	  const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	  if ( outerUpdated.isValid()) {
	    LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;
	    
            PTrajectoryStateOnDet const &  PTraj =
	      trajectoryStateTransform::persistentState(outerUpdated, (*(HitPairs[is].outer())).geographicalId().rawId());
	    
	    output.push_back( TrajectorySeed(PTraj,hits,alongMomentum));

            if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
              edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
              output.clear(); 
              return false;
            }
	    
	  }else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
	}else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";
      
      
      }
      else{
	GlobalTrajectoryParameters Gtp(outer,
				       (outer-inner)*(seedpt/(outer-inner).mag()),
				       predsign, 
				       &(*magfield));
	FreeTrajectoryState CosmicSeed(Gtp,
				       CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
	LogDebug("CosmicSeedFinder") << " FirstTSOS "<<CosmicSeed;
	//First propagation
	const TSOS outerState =
	  thePropagatorAl->propagate(CosmicSeed,
				     tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface());
	if ( outerState.isValid()) {
	  
	  LogDebug("CosmicSeedFinder") <<"outerState "<<outerState;
	  const TSOS outerUpdated= theUpdator->update( outerState,*outrhit);
	  if ( outerUpdated.isValid()) {
	  LogDebug("CosmicSeedFinder") <<"outerUpdated "<<outerUpdated;

	  PTrajectoryStateOnDet const &  PTraj = 
	    trajectoryStateTransform::persistentState(outerUpdated,(*(HitPairs[is].outer())).geographicalId().rawId());
          
	  output.push_back(TrajectorySeed(PTraj,hits,oppositeToMomentum));

          if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
            edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
            output.clear(); 
            return false;
          }

	  }else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
	}else      edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";
      }
      
    }
  }
  return true;
}
