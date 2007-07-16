#include "RecoTracker/TrackProducer/interface/GsfTrackProducerAlgorithm.h"

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

void GsfTrackProducerAlgorithm::runWithCandidate(const TrackingGeometry * theG,
					      const MagneticField * theMF,
					      const TrackCandidateCollection& theTCCollection,
					      const TrajectoryFitter * theFitter,
					      const Propagator * thePropagator,
					      const TransientTrackingRecHitBuilder* builder,
					      AlgoProductCollection& algoResults)
{
  edm::LogInfo("GsfTrackProducer") << "Number of TrackCandidates: " << theTCCollection.size() << "\n";

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

      LogDebug("GsfTrackProducer") << "Initial TSOS\n" << theTSOS << "\n";
      
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
      LogDebug("GsfTrackProducer") << "going to buildTrack"<< "\n";
      bool ok = buildTrack(theFitter,thePropagator,algoResults, hits, theTSOS, seed, ndof);
      LogDebug("GsfTrackProducer") << "buildTrack result: " << ok << "\n";
      if(ok) cont++;
    }
  edm::LogInfo("GsfTrackProducer") << "Number of Tracks found: " << cont << "\n";
}


void GsfTrackProducerAlgorithm::runWithTrack(const TrackingGeometry * theG,
					  const MagneticField * theMF,
					  const reco::TrackCollection& theTCollection,
					  const TrajectoryFitter * theFitter,
					  const Propagator * thePropagator,
					  const TransientTrackingRecHitBuilder* builder,
					  AlgoProductCollection& algoResults)
{
  edm::LogInfo("GsfTrackProducer") << "Number of input Tracks: " << theTCollection.size() << "\n";
  
  int cont = 0;
  for (reco::TrackCollection::const_iterator i=theTCollection.begin(); i!=theTCollection.end();i++)
    {
      try{
	const reco::Track * theT = &(*i);
	
	//convert the TrackingRecHit vector to a TransientTrackingRecHit vector
	//meanwhile computes the number of degrees of freedom

	TransientTrackingRecHit::RecHitContainer tmp;
	TransientTrackingRecHit::RecHitContainer hits;
	
	float ndof=0;
	
	for (trackingRecHit_iterator i=theT->recHitsBegin();
	     i!=theT->recHitsEnd(); i++){
	  // 	hits.push_back(builder->build(&**i ));
	  // 	  if ((*i)->isValid()){
	    tmp.push_back(builder->build(&**i ));
	    if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
	    //	  }
	}
	
	
	ndof = ndof - 5;

	//SORT RECHITS ALONGMOMENTUM
	TransientTrackingRecHit::ConstRecHitPointer firstHit;
	for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.begin(); it!=tmp.end();it++){
	  if ((**it).isValid()) {
	    firstHit = *it;
	    break;
	  }
	}
	TransientTrackingRecHit::ConstRecHitPointer lastHit;
	for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1; it!=tmp.begin()-1;it--){
	  if ((**it).isValid()) {
	    lastHit = *it;
	    break;
	  }
	}
	if (firstHit->globalPosition().mag2() > (lastHit->globalPosition().mag2()) ){
	//FIXME temporary should use reverse
	  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1;it!=tmp.begin()-1;it--){
	    hits.push_back(*it);
	  }
	} else hits=tmp;
	
	reco::TransientTrack theTT(*theT,thePropagator->magneticField() );
	
	//       TrajectoryStateOnSurface theTSOS=theTT.impactPointState();
	//       theTSOS.rescaleError(100);

	TrajectoryStateOnSurface firstState=thePropagator->propagate(theTT.impactPointState(), hits.front()->det()->surface());
	AlgebraicSymMatrix55 C = AlgebraicMatrixID();
	C *= 100.;
	TrajectoryStateOnSurface theTSOS( firstState.localParameters(), LocalTrajectoryError(C),
					  firstState.surface(),
					  thePropagator->magneticField()); 
	
	LogDebug("GsfTrackProducer") << "Initial TSOS\n" << theTSOS << "\n";
	
	const TrajectorySeed * seed = new TrajectorySeed();//empty seed: not needed
	//buildTrack
	bool ok = buildTrack(theFitter,thePropagator,algoResults, hits, theTSOS, *seed, ndof);
	if(ok) cont++;
      }catch ( CMSexception & e){
	edm::LogInfo("GsfTrackProducer") << "Genexception1: " << e.explainSelf() <<"\n";      
      }catch ( std::exception & e){
	edm::LogInfo("GsfTrackProducer") << "Genexception2: " << e.what() <<"\n";      
      }catch (...){
	edm::LogInfo("GsfTrackProducer") << "Genexception: \n";
      }
    }
  edm::LogInfo("GsfTrackProducer") << "Number of Tracks found: " << cont << "\n";
  
}

bool GsfTrackProducerAlgorithm::buildTrack (const TrajectoryFitter * theFitter,
					 const Propagator * thePropagator,
					 AlgoProductCollection& algoResults,
					 TransientTrackingRecHit::RecHitContainer& hits,
					 TrajectoryStateOnSurface& theTSOS,
					 const TrajectorySeed& seed,
					 float ndof)
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
    
    
    TSCPBuilderNoMaterial tscpBuilder;

    TrajectoryStateClosestToPoint tscp = tscpBuilder(*(innertsos.freeState()),
						     GlobalPoint(0,0,0) );//FIXME Correct?
    
//      PerigeeTrajectoryParameters::ParameterVector param = tscp.perigeeParameters();
//  
//      PerigeeTrajectoryError::CovarianceMatrix covar = tscp.perigeeError();

    GlobalPoint v = tscp.theState().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = tscp.theState().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    theTrack = new reco::GsfTrack(theTraj->chiSquared(),
				  int(ndof),//FIXME fix weight() in TrackingRecHit
				  //			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
				  //			       0, //FIXME no corresponding method in trajectory.h
				  //			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
				  pos, mom, tscp.charge(), tscp.theState().curvilinearError().matrix());


    LogDebug("GsfTrackProducer") <<"track done\n";

    AlgoProduct aProduct(theTraj,std::make_pair(theTrack,seedDir));
    LogDebug("GsfTrackProducer") <<"track done1\n";
    algoResults.push_back(aProduct);
    LogDebug("GsfTrackProducer") <<"track done2\n";
    
    return true;
  } 
  else  return false;
}
