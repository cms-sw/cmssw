#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

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

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "Utilities/General/interface/CMSexception.h"

void TrackProducerAlgorithm::runWithCandidate(const TrackingGeometry * theG,
					      const MagneticField * theMF,
					      const TrackCandidateCollection& theTCCollection,
					      const TrajectoryFitter * theFitter,
					      const Propagator * thePropagator,
					      const TransientTrackingRecHitBuilder* builder,
					      AlgoProductCollection& algoResults)
{
  edm::LogInfo("TrackProducer") << "Number of TrackCandidates: " << theTCCollection.size() << "\n";

  int cont = 0;
  for (TrackCandidateCollection::const_iterator i=theTCCollection.begin(); i!=theTCCollection.end();i++)
    {
      
      const TrackCandidate * theTC = &(*i);
      PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
      const TrackCandidate::range& recHitVec=theTC->recHits();
      const TrajectorySeed& seed = theTC->seed();

      //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
      TrajectoryStateTransform transformer;
  
      DetId  detId(state.detId());
      TrajectoryStateOnSurface theTSOS = transformer.transientState( state,
								     &(theG->idToDet(detId)->surface()), 
								     theMF);

      LogDebug("TrackProducer") << "Initial TSOS\n" << theTSOS << "\n";
      
      //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
      //meanwhile computes the number of degrees of freedom
      TransientTrackingRecHit::RecHitContainer hits;
      
      float ndof=0;
      
      for (edm::OwnVector<TrackingRecHit>::const_iterator i=recHitVec.first;
	   i!=recHitVec.second; i++){
	hits.push_back(builder->build(&(*i) ));
	if ((*i).isValid()){
	  ndof = ndof + (i->dimension())*(i->weight());
	}
      }
      
      
      ndof = ndof - 5;
      
      //build Track
      LogDebug("TrackProducer") << "going to buildTrack"<< "\n";
      bool ok = buildTrack(theFitter,thePropagator,algoResults, hits, theTSOS, seed, ndof);
      LogDebug("TrackProducer") << "buildTrack result: " << ok << "\n";
      if(ok) cont++;
    }
  edm::LogInfo("TrackProducer") << "Number of Tracks found: " << cont << "\n";
}


void TrackProducerAlgorithm::runWithTrack(const TrackingGeometry * theG,
					  const MagneticField * theMF,
					  const reco::TrackCollection& theTCollection,
					  const TrajectoryFitter * theFitter,
					  const Propagator * thePropagator,
					  const TransientTrackingRecHitBuilder* builder,
					  AlgoProductCollection& algoResults)
{
  edm::LogInfo("TrackProducer") << "Number of input Tracks: " << theTCollection.size() << "\n";
  
  int cont = 0;
  for (reco::TrackCollection::const_iterator i=theTCollection.begin(); i!=theTCollection.end();i++)
    {
      try{
	const reco::Track * theT = &(*i);
	
	//convert the TrackingRecHit vector to a TransientTrackingRecHit vector
	//meanwhile computes the number of degrees of freedom

	TransientTrackingRecHit::RecHitContainer hits;
	
	float ndof=0;
	
	//========  We keep hits sorted as they were in the previously reconstructed track
	for (trackingRecHit_iterator i=theT->recHitsBegin();
	     i!=theT->recHitsEnd(); i++){
	  hits.push_back(builder->build(&**i ));
		    if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
	}
	
	
	ndof = ndof - 5;

	bool isFirstFound(false);
	// just look for the first and second *valid* hits.Don't care about ordering. See comment above.
	TransientTrackingRecHit::ConstRecHitPointer firstHit(0),secondHit(0);
	for (TransientTrackingRecHit::RecHitContainer::const_iterator it=hits.begin(); it!=hits.end();it++){
	  if(((**it).isValid()) ) {
	    if(!isFirstFound){
	      isFirstFound = true;
	      firstHit = *it;
	      //LogDebug("TrackProducer") << "firstHit->globalPosition(): " << firstHit->globalPosition() << std::endl;
	      continue;
	    }
	    secondHit = *it;
	    //LogDebug("TrackProducer") << "secondHit->globalPosition(): " << secondHit->globalPosition() << std::endl;
	    break;
	  }else LogDebug("TrackProducer") << "==== debug:this hit of a reco::Track is not valid!! =======";
	}


	//======== this is a crap :( Just a temporary "most general" solution before the sorting of hits is made 
	//         consistent in several part of the tracking code. 
	//         it assumes the hits at least follow the track path.No matter the direction.
	TrajectoryStateTransform transformer;
	//	  avoiding to use transientTrack, it should be faster;
	TrajectoryStateOnSurface innerStateFromTrack=transformer.innerStateOnSurface(*theT,*theG,theMF);
	TrajectoryStateOnSurface outerStateFromTrack=transformer.outerStateOnSurface(*theT,*theG,theMF);
	TrajectoryStateOnSurface initialStateFromTrack = 
	  ( (innerStateFromTrack.globalPosition()-firstHit->globalPosition()).mag2() <
	    (outerStateFromTrack.globalPosition()-firstHit->globalPosition()).mag2() ) ? 
	  innerStateFromTrack: outerStateFromTrack;       
	//std::cout << "initialStateFromTrack->globalPosition: " << initialStateFromTrack.globalPosition() << std::endl;
	// =========================

	// error is rescaled, but correlation are kept.
	initialStateFromTrack.rescaleError(100);
	TrajectoryStateOnSurface theInitialStateForRefitting( initialStateFromTrack.localParameters(),
							      initialStateFromTrack.localError(), 		      
							      initialStateFromTrack.surface(),
							      thePropagator->magneticField()); 
		
	// ====== Another crap: investigate how the hits are sorted: either alongMomentum or opposite.
	//        In the next release we should have a method in the reco::Track to know that. 
	//        Or adapt a general convention.
	GlobalVector delta = secondHit->globalPosition() - firstHit->globalPosition() ;
	PropagationDirection seedDirection = 
	  (theInitialStateForRefitting.globalDirection()*delta>0)? alongMomentum : oppositeToMomentum;
	// the seed has dummy state and hits.What matters for the fitting is the seedDirection;
	const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(),
						   BasicTrajectorySeed::recHitContainer(), seedDirection);
	// =========================

	//=====  the hits are in the same order as they were in the track::extra.        
	bool ok = buildTrack(theFitter,thePropagator,algoResults, hits, theInitialStateForRefitting, seed, ndof);
	if(ok) cont++;
      }catch ( CMSexception & e){
	edm::LogError("TrackProducer") << "Genexception1: " << e.explainSelf() <<"\n";      
      }catch ( std::exception & e){
	edm::LogError("TrackProducer") << "Genexception2: " << e.what() <<"\n";      
      }catch (...){
	edm::LogError("TrackProducer") << "Genexception: \n";
      }
    }
  edm::LogInfo("TrackProducer") << "Number of Tracks found: " << cont << "\n";
  
}

bool TrackProducerAlgorithm::buildTrack (const TrajectoryFitter * theFitter,
					 const Propagator * thePropagator,
					 AlgoProductCollection& algoResults,
					 TransientTrackingRecHit::RecHitContainer& hits,
					 TrajectoryStateOnSurface& theTSOS,
					 const TrajectorySeed& seed,
					 float ndof)
{
  //variable declarations
  std::vector<Trajectory> trajVec;
  reco::Track * theTrack;
  Trajectory * theTraj; 
      
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  trajVec = theFitter->fit(seed, hits, theTSOS);
  
  LogDebug("TrackProducer") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  
  TrajectoryStateOnSurface innertsos;
  
  if (trajVec.size() != 0){

    theTraj = new Trajectory( trajVec.front() );
    
    if (theTraj->direction() == alongMomentum) {
      innertsos = theTraj->firstMeasurement().updatedState();
    } else { 
      innertsos = theTraj->lastMeasurement().updatedState();
    }
    
    
    TSCPBuilderNoMaterial tscpBuilder;

    TrajectoryStateClosestToPoint tscp = tscpBuilder(*(innertsos.freeState()),
						     GlobalPoint(0,0,0) );//FIXME Correct?
    
    GlobalPoint v = tscp.theState().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = tscp.theState().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    LogDebug("TrackProducer") <<v<<p<<std::endl;
//      PerigeeTrajectoryParameters::ParameterVector param = tscp.perigeeParameters();
//  
//      PerigeeTrajectoryError::CovarianceMatrix covar = tscp.perigeeError();

    theTrack = new reco::Track(theTraj->chiSquared(),
			       int(ndof),//FIXME fix weight() in TrackingRecHit
			       //			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
			       //			       0, //FIXME no corresponding method in trajectory.h
			       //			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
			       pos, mom, tscp.charge(), tscp.theState().curvilinearError());


    LogDebug("TrackProducer") <<"track done\n";

    AlgoProduct aProduct(theTraj,theTrack);
    LogDebug("TrackProducer") <<"track done1\n";
    algoResults.push_back(aProduct);
    LogDebug("TrackProducer") <<"track done2\n";
    
    return true;
  } 
  else  return false;
}
