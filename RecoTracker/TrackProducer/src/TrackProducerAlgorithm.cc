#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

//#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "Utilities/General/interface/CMSexception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit2DPosConstraint.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit1DMomConstraint.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

template <> bool
TrackProducerAlgorithm<reco::Track>::buildTrack (const TrajectoryFitter * theFitter,
						 const Propagator * thePropagator,
						 AlgoProductCollection& algoResults,
						 TransientTrackingRecHit::RecHitContainer& hits,
						 TrajectoryStateOnSurface& theTSOS,
						 const TrajectorySeed& seed,
						 float ndof,
						 const reco::BeamSpot& bs,
						 SeedRef seedRef)
{
  //variable declarations
  std::vector<Trajectory> trajVec;
  reco::Track * theTrack;
  Trajectory * theTraj; 
  PropagationDirection seedDir = seed.direction();
      
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  trajVec = theFitter->fit(seed, hits, theTSOS);
  
  LogDebug("TrackProducer") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  TrajectoryStateOnSurface innertsos;
  
  if (trajVec.size() != 0){

    theTraj = new Trajectory( trajVec.front() );
    theTraj->setSeedRef(seedRef);
    
    if (theTraj->direction() == alongMomentum) {
      innertsos = theTraj->firstMeasurement().updatedState();
    } else { 
      innertsos = theTraj->lastMeasurement().updatedState();
    }
    
    ndof = 0;
    TransientTrackingRecHit::RecHitContainer validHits;
    theTraj->validRecHits(validHits);
    for (TransientTrackingRecHit::RecHitContainer::iterator h=validHits.begin();h!=validHits.end();++h)
      ndof = ndof + ((*h)->dimension())*((*h)->weight());
    ndof = ndof - 5;
    
    TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
    LogDebug("TrackProducer") << "innertsos=" << innertsos ;

    TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

    if (tscbl.isValid()==false) {
        delete theTraj;
        return false;
    }

    GlobalPoint v = tscbl.trackStateAtPCA().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = tscbl.trackStateAtPCA().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    LogDebug("TrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();

    algoName_ = conf_.getParameter<std::string>( "AlgorithmName" );  
    //std::string theTrackAlgo = algoName_;
    reco::TrackBase::TrackAlgorithm Algo = reco::TrackBase::undefAlgorithm;
    if (algoName_ == "ctf")
      Algo = reco::TrackBase::ctf;
    else if (algoName_ == "rs")
      Algo = reco::TrackBase::rs;
    else if (algoName_ == "beamhalo")
      Algo = reco::TrackBase::beamhalo;
    else if (algoName_ == "cosmics")
      Algo = reco::TrackBase::cosmics;
    else if (algoName_ == "iter1")
      Algo = reco::TrackBase::iter1;
    else if (algoName_ == "iter2")
      Algo = reco::TrackBase::iter2;
    else if (algoName_ == "iter3")
      Algo = reco::TrackBase::iter3;

    theTrack = new reco::Track(theTraj->chiSquared(),
			       int(ndof),//FIXME fix weight() in TrackingRecHit
			       pos, mom, tscbl.trackStateAtPCA().charge(), 
			       tscbl.trackStateAtPCA().curvilinearError(),
			       Algo);
    
    LogDebug("TrackProducer") << "theTrack->pt()=" << theTrack->pt();

    LogDebug("TrackProducer") <<"track done\n";

    AlgoProduct aProduct(theTraj,std::make_pair(theTrack,seedDir));
    algoResults.push_back(aProduct);
    
    return true;
  } 
  else  return false;
}

template <> bool
TrackProducerAlgorithm<reco::GsfTrack>::buildTrack (const TrajectoryFitter * theFitter,
						    const Propagator * thePropagator,
						    AlgoProductCollection& algoResults,
						    TransientTrackingRecHit::RecHitContainer& hits,
						    TrajectoryStateOnSurface& theTSOS,
						    const TrajectorySeed& seed,
						    float ndof,
						    const reco::BeamSpot& bs,
						    SeedRef seedRef)
{
  //variable declarations
  std::vector<Trajectory> trajVec;
  reco::GsfTrack * theTrack;
  Trajectory * theTraj; 
  PropagationDirection seedDir = seed.direction();
      
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  trajVec = theFitter->fit(seed, hits, theTSOS);
  
  LogDebug("GsfTrackProducer") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  
  TrajectoryStateOnSurface innertsos;
  TrajectoryStateOnSurface outertsos;
  
  if (trajVec.size() != 0){

    theTraj = new Trajectory( trajVec.front() );
    theTraj->setSeedRef(seedRef);
    
    if (theTraj->direction() == alongMomentum) {
      innertsos = theTraj->firstMeasurement().updatedState();
      outertsos = theTraj->lastMeasurement().updatedState();
    } else { 
      innertsos = theTraj->lastMeasurement().updatedState();
      outertsos = theTraj->firstMeasurement().updatedState();
    }
//     std::cout
//       << "Nr. of first / last states = "
//       << innertsos.components().size() << " "
//       << outertsos.components().size() << std::endl;
//     std::vector<TrajectoryStateOnSurface> components = 
//       innertsos.components();
//     double sinTheta = 
//       sin(innertsos.globalMomentum().theta());
//     for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
// 	  ic!=components.end(); ic++ ) {
//       std::cout << " comp " << ic-components.begin() << " "
// 		<< (*ic).weight() << " "
// 		<< (*ic).localParameters().vector()[0]/sinTheta << " "
// 		<< sqrt((*ic).localError().matrix()[0][0])/sinTheta << std::endl;
//     }
    
    ndof = 0;
    TransientTrackingRecHit::RecHitContainer validHits;
    theTraj->validRecHits(validHits);
    for (TransientTrackingRecHit::RecHitContainer::iterator h=validHits.begin();h!=validHits.end();++h)
      ndof = ndof + ((*h)->dimension())*((*h)->weight());
    ndof = ndof - 5;
   
    TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
    LogDebug("TrackProducer") << "innertsos=" << innertsos ;

    TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

    if (tscbl.isValid()==false) {
        delete theTraj;
        return false;
    }

    GlobalPoint v = tscbl.trackStateAtPCA().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = tscbl.trackStateAtPCA().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    LogDebug("TrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();

    theTrack = new reco::GsfTrack(theTraj->chiSquared(),
				  int(ndof),//FIXME fix weight() in TrackingRecHit
				  //			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
				  //			       0, //FIXME no corresponding method in trajectory.h
				  //			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
				  pos, mom, tscbl.trackStateAtPCA().charge(), tscbl.trackStateAtPCA().curvilinearError());    

    LogDebug("GsfTrackProducer") <<"track done\n";

    AlgoProduct aProduct(theTraj,std::make_pair(theTrack,seedDir));
    LogDebug("GsfTrackProducer") <<"track done1\n";
    algoResults.push_back(aProduct);
    LogDebug("GsfTrackProducer") <<"track done2\n";
    
    return true;
  } 
  else  return false;
}
