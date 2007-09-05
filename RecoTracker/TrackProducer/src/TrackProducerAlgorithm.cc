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

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "Utilities/General/interface/CMSexception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit2DPosConstraint.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit1DMomConstraint.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"

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
	float ndof=0;
	PropagationDirection seedDir = theT->seedDirection();
	//LogDebug("TrackProducer") << "theT->seedDirection()=" << seedDir;

	TransientTrackingRecHit::RecHitContainer hits = getHitVector(theT,seedDir,ndof,builder);

	TrajectoryStateOnSurface theInitialStateForRefitting = getInitialState(theT,hits,theG,theMF);

	// the seed has dummy state and hits.What matters for the fitting is the seedDirection;
	const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(),
						   TrajectorySeed::recHitContainer(), seedDir);
	// =========================
	//LogDebug("TrackProducer") << "seed.direction()=" << seed.direction();

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

void TrackProducerAlgorithm::runWithMomentum(const TrackingGeometry * theG,
					     const MagneticField * theMF,
					     const TrackMomConstraintAssociationCollection& theTCollectionWithConstraint,
					     const TrajectoryFitter * theFitter,
					     const Propagator * thePropagator,
					     const TransientTrackingRecHitBuilder* builder,
					     AlgoProductCollection& algoResults){

  edm::LogInfo("TrackProducer") << "Number of input Tracks: " << theTCollectionWithConstraint.size() << "\n";
  
  int cont = 0;
  for (TrackMomConstraintAssociationCollection::const_iterator i=theTCollectionWithConstraint.begin(); i!=theTCollectionWithConstraint.end();i++) {
      try{
	const reco::Track * theT = i->key.get();

	LogDebug("TrackProducer") << "Running Refitter with Momentum Constraint. p=" << i->val->first << " err=" << i->val->second;

	float ndof=0;
	PropagationDirection seedDir = theT->seedDirection();

	TransientTrackingRecHit::RecHitContainer hits = getHitVector(theT,seedDir,ndof,builder);

	TrajectoryStateOnSurface theInitialStateForRefitting = getInitialState(theT,hits,theG,theMF);

	double mom = i->val->first;//10;
	double err = i->val->second;//0.01;
	TransientTrackingRecHit::RecHitPointer testhit = 
	  TRecHit1DMomConstraint::build(((int)(theInitialStateForRefitting.charge())),
					mom,err,
					&theInitialStateForRefitting.surface());
	
	//no insert in OwnVector...
	TransientTrackingRecHit::RecHitContainer tmpHits;	
	tmpHits.push_back(testhit);
	for (TransientTrackingRecHit::RecHitContainer::const_iterator i=hits.begin(); i!=hits.end(); i++){
	  tmpHits.push_back(*i);
	}
	hits.swap(tmpHits);
	
	// the seed has dummy state and hits.What matters for the fitting is the seedDirection;
	const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(),
						   TrajectorySeed::recHitContainer(), seedDir);
	// =========================
	//LogDebug("TrackProducer") << "seed.direction()=" << seed.direction();
	
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

void TrackProducerAlgorithm::runWithVertex(const TrackingGeometry * theG,
					     const MagneticField * theMF,
					     const TrackVtxConstraintAssociationCollection& theTCollectionWithConstraint,
					     const TrajectoryFitter * theFitter,
					     const Propagator * thePropagator,
					     const TransientTrackingRecHitBuilder* builder,
					     AlgoProductCollection& algoResults){

  edm::LogInfo("TrackProducer") << "Number of input Tracks: " << theTCollectionWithConstraint.size() << "\n";
  
  int cont = 0;
  for (TrackVtxConstraintAssociationCollection::const_iterator i=theTCollectionWithConstraint.begin(); i!=theTCollectionWithConstraint.end();i++) {
      try{
	const reco::Track * theT = i->key.get();

	LogDebug("TrackProducer") << "Running Refitter with Vertex Constraint. pos=" << i->val->first << " err=" << i->val->second.matrix();

	float ndof=0;
	PropagationDirection seedDir = theT->seedDirection();

	TransientTrackingRecHit::RecHitContainer hits = getHitVector(theT,seedDir,ndof,builder);

	TrajectoryStateOnSurface theInitialStateForRefitting = getInitialState(theT,hits,theG,theMF);

	const LocalPoint testpoint(0,0,0);
	GlobalPoint pos = i->val->first;//(0,0,0);
	GlobalError err = i->val->second;//(0.01,0,0,0.01,0,0.001);

	Propagator* myPropagator = new PropagatorWithMaterial(anyDirection,0.105,theMF);
	TransverseImpactPointExtrapolator extrapolator(*myPropagator);
	TrajectoryStateOnSurface tsosAtVtx = extrapolator.extrapolate(theInitialStateForRefitting,pos);
	
	const Surface * surfAtVtx = &tsosAtVtx.surface();
	

	LocalError testerror = ErrorFrameTransformer().transform(err, *surfAtVtx);
	
	//GlobalError myerror = ErrorFrameTransformer().transform(testerror, *surfAtVtx);
	//LogTrace("TrackProducer") << "initial GlobalError:" << err.matrix();
	//LogTrace("TrackProducer") << "Surface position:\n" << surfAtVtx->position();
	//LogTrace("TrackProducer") << "Surface rotation:\n" << surfAtVtx->rotation();
	//LogTrace("TrackProducer") << "corresponding LocalError:\n" << testerror;
	//LogTrace("TrackProducer") << "corresponding GlobalError:" << myerror.matrix();
	
	TransientTrackingRecHit::RecHitPointer testhit = TRecHit2DPosConstraint::build(testpoint,testerror,surfAtVtx);
	
	//push constraining hit and sort along seed direction
	hits.push_back(testhit);
	RecHitSorter sorter = RecHitSorter();
	hits = sorter.sortHits(hits,seedDir);

	//use the state on the surface of the first hit (could be the constraint or not)
	theInitialStateForRefitting = myPropagator->propagate(*theInitialStateForRefitting.freeState(), *(hits[0]->surface()) );	  
	delete myPropagator;

	// the seed has dummy state and hits.What matters for the fitting is the seedDirection;
	const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(),
						   TrajectorySeed::recHitContainer(), seedDir);
	// =========================
	//LogDebug("TrackProducer") << "seed.direction()=" << seed.direction();

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
  PropagationDirection seedDir = seed.direction();
      
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
    LogDebug("TrackProducer") << "innertsos=" << innertsos ;
    TrajectoryStateClosestToPoint tscp = tscpBuilder(*(innertsos.freeState()),
						     GlobalPoint(0,0,0) );//FIXME Correct?   
    GlobalPoint v = tscp.theState().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = tscp.theState().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    LogDebug("TrackProducer") << "pos=" << v << " mom=" << p << " pt=" << p.perp() << " mag=" << p.mag();

    theTrack = new reco::Track(theTraj->chiSquared(),
			       int(ndof),//FIXME fix weight() in TrackingRecHit
			       //			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
			       //			       0, //FIXME no corresponding method in trajectory.h
			       //			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
			       pos, mom, tscp.charge(), tscp.theState().curvilinearError());
    
    LogDebug("TrackProducer") << "theTrack->pt()=" << theTrack->pt();

    LogDebug("TrackProducer") <<"track done\n";

    AlgoProduct aProduct(theTraj,std::make_pair(theTrack,seedDir));
    algoResults.push_back(aProduct);
    
    return true;
  } 
  else  return false;
}

TrajectoryStateOnSurface TrackProducerAlgorithm::getInitialState(const reco::Track * theT,
								 TransientTrackingRecHit::RecHitContainer& hits,
								 const TrackingGeometry * theG,
								 const MagneticField * theMF){

  TrajectoryStateOnSurface theInitialStateForRefitting;
  //the starting state is the state closest to the first hit along seedDirection.
  TrajectoryStateTransform transformer;
  //avoiding to use transientTrack, it should be faster;
  TrajectoryStateOnSurface innerStateFromTrack=transformer.innerStateOnSurface(*theT,*theG,theMF);
  TrajectoryStateOnSurface outerStateFromTrack=transformer.outerStateOnSurface(*theT,*theG,theMF);
  TrajectoryStateOnSurface initialStateFromTrack = 
    ( (innerStateFromTrack.globalPosition()-hits.front()->globalPosition()).mag2() <
      (outerStateFromTrack.globalPosition()-hits.front()->globalPosition()).mag2() ) ? 
    innerStateFromTrack: outerStateFromTrack;       
  
  // error is rescaled, but correlation are kept.
  initialStateFromTrack.rescaleError(100);
  theInitialStateForRefitting = TrajectoryStateOnSurface(initialStateFromTrack.localParameters(),
							 initialStateFromTrack.localError(), 		      
							 initialStateFromTrack.surface(),
							 theMF); 
  return theInitialStateForRefitting;
}

TransientTrackingRecHit::RecHitContainer 
TrackProducerAlgorithm::getHitVector(const reco::Track * theT,   
				     PropagationDirection& seedDir,
				     float& ndof,
				     const TransientTrackingRecHitBuilder* builder){

  TransientTrackingRecHit::RecHitContainer hits;	
  bool isFirstFound(false);
  //just look for the first and second *valid* hits.Don't care about ordering.
  TransientTrackingRecHit::ConstRecHitPointer firstHit(0),secondHit(0);
  for (trackingRecHit_iterator it=theT->recHitsBegin(); it!=theT->recHitsEnd(); it++){
    if(((**it).isValid()) ) {
      if(!isFirstFound){
	isFirstFound = true;
	firstHit = builder->build(&**it);
	//LogDebug("TrackProducer") << "firstHit->globalPosition(): " << firstHit->globalPosition() << std::endl;
	continue;
      }
      secondHit = builder->build(&**it);
      //LogDebug("TrackProducer") << "secondHit->globalPosition(): " << secondHit->globalPosition() << std::endl;
      break;
    }else LogDebug("TrackProducer") << "==== debug:this hit of a reco::Track is not valid!! =======";
  }
  GlobalVector delta = secondHit->globalPosition() - firstHit->globalPosition() ;
  PropagationDirection trackHitsSort = ( (delta.dot(GlobalVector(theT->momentum().x(),theT->momentum().y(),theT->momentum().z()))
					  > 0) ? alongMomentum : oppositeToMomentum);
  
  //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
  //meanwhile computes the number of degrees of freedom
  //========  Sorted as seed direction (refit in the same ordeer of original final fit)
  if (seedDir==anyDirection){//if anyDirection the seed direction is not stored in the root file: keep same order
    for (trackingRecHit_iterator i=theT->recHitsBegin(); i!=theT->recHitsEnd(); i++){
      hits.push_back(builder->build(&**i ));
      if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
    }
    seedDir=trackHitsSort;
  } else if (seedDir==trackHitsSort){//keep same order
    for (trackingRecHit_iterator i=theT->recHitsBegin(); i!=theT->recHitsEnd(); i++){
      hits.push_back(builder->build(&**i ));
      if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
    }
  } else{//invert hits order
    //no reverse iterator in OwnVector...
    for (TrackingRecHitRefVector::iterator i=theT->recHitsEnd()-1; i!=theT->recHitsBegin()-1; i--){
      hits.push_back(builder->build(&**i ));
      if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
    }	  
  }
  ndof = ndof - 5;
  return hits;
}
