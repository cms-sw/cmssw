#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "FWCore/Framework/interface/OrphanHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

void TrackProducerAlgorithm::runWithCandidate(const TrackingGeometry * theG,
					      const MagneticField * theMF,
					      const TrackCandidateCollection& theTCCollection,
					      const TrajectoryFitter * theFitter,
					      const Propagator * thePropagator,
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
  
      DetId * detId = new DetId(state.detId());
      TrajectoryStateOnSurface theTSOS = transformer.transientState( state,
								     &(theG->idToDet(*detId)->surface()), 
								     theMF);
      
      //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
      //meanwhile computes the number of degrees of freedom
      edm::OwnVector<TransientTrackingRecHit> hits;
      TransientTrackingRecHitBuilder * builder;
  
      //
      // temporary!
      //
      builder = new TkTransientTrackingRecHitBuilder( theG);
      
      float ndof=0;
      
      for (edm::OwnVector<TrackingRecHit>::const_iterator i=recHitVec.first;
	   i!=recHitVec.second; i++){
	hits.push_back(builder->build(&(*i) ));
	if ((*i).isValid()){
	  ndof = ndof + (i->dimension())*(i->weight());
	}
      }
      
      delete builder;
      
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
					  AlgoProductCollection& algoResults)
{
  edm::LogInfo("TrackProducer") << "Number of input Tracks: " << theTCollection.size() << "\n";

  int cont = 0;
  for (reco::TrackCollection::const_iterator i=theTCollection.begin(); i!=theTCollection.end();i++)
    {

      const reco::Track * theT = &(*i);

      //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
      //meanwhile computes the number of degrees of freedom
      edm::OwnVector<TransientTrackingRecHit> hits;
      TransientTrackingRecHitBuilder * builder;
  
      //
      // temporary!
      //
      builder = new TkTransientTrackingRecHitBuilder( theG);
      
      float ndof=0;
      
      for (trackingRecHit_iterator i=theT->recHitsBegin();
	   i!=theT->recHitsEnd(); i++){
	hits.push_back(builder->build(&**i ));
	if ((*i)->isValid()){
	  ndof = ndof + ((*i)->dimension())*((*i)->weight());
	}
      }
      
      delete builder;
      
      ndof = ndof - 5;

      //SORT RECHITS ALONGMOMENTUM
      hits.sort(TrackingRecHitLessFromGlobalPosition(theG,alongMomentum));

      reco::TransientTrack theTT(*theT);

      TrajectoryStateOnSurface theTSOS=theTT.impactPointState();
      theTSOS.rescaleError(100);

      const TrajectorySeed * seed = new TrajectorySeed();//empty seed: not needed
      //buildTrack
      bool ok = buildTrack(theFitter,thePropagator,algoResults, hits, theTSOS, *seed, ndof);
      if(ok) cont++;
    }
  edm::LogInfo("TrackProducer") << "Number of Tracks found: " << cont << "\n";

}

bool TrackProducerAlgorithm::buildTrack (const TrajectoryFitter * theFitter,
					 const Propagator * thePropagator,
					 AlgoProductCollection& algoResults,
					 edm::OwnVector<TransientTrackingRecHit>& hits,
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
    
    
    //extrapolate the innermost state to the point of closest approach to the beamline
    //     tsos = tipe->extrapolate(*(innertsos.freeState()), 
    // 			     GlobalPoint(0,0,0) );//FIXME Correct?
    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateClosestToPoint tscp = tscpBuilder(*(innertsos.freeState()),
						     GlobalPoint(0,0,0) );
    
//     //
//     // TB: if the tsos is not valid, stop
//     //
//     if (tscp.isValid() == false) {
//       edm::LogInfo("TrackProducer") <<" Could not extrapolate a track to (0,0,0) - skipping it.\n";
// 	  return false;//	  continue;
//     }
    
    reco::perigee::Parameters param = tscp.perigeeParameters();
 
    reco::perigee::Covariance covar = tscp.perigeeError();

    theTrack = new reco::Track(theTraj->chiSquared(),
			       int(ndof),//FIXME fix weight() in TrackingRecHit
			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
			       0, //FIXME no corresponding method in trajectory.h
			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
			       param,
			       covar);

    LogDebug("TrackProducer") <<"track done\n";

    //     //compute parameters needed to build a Track from a Trajectory    
    //     int charge = tsos.charge();
    //     const GlobalTrajectoryParameters& gp = tsos.globalParameters();
    //     GlobalPoint v = gp.position();
    //     GlobalVector p = gp.momentum();
    //     const CartesianTrajectoryError& cte = tsos.cartesianError();
    //     AlgebraicSymMatrix m = cte.matrix();
    //     math::Error<6>::type cov;
    //     for( int i = 0; i < 6; ++i )
    //       for( int j = 0; j <= i; ++j )
    // 	cov( i, j ) = m.fast( i + 1 , j + 1 );
    //     math::XYZVector mom( p.x(), p.y(), p.z() );
    //     math::XYZPoint  vtx( v.x(), v.y(), v.z() );   
    //     edm::LogInfo("TrackProducer") << " RESULT Momentum "<< p<<"\n";
    //     edm::LogInfo("TrackProducer") << " RESULT Vertex "<< v<<"\n";
    
    //     //build the Track(chiSquared, ndof, found, invalid, lost, q, vertex, momentum, covariance)
    //     theTrack = new reco::Track(theTraj->chiSquared(), 
    // 			       int(ndof),//FIXME fix weight() in TrackingRecHit 
    // 			       theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
    // 			       0, //FIXME no corresponding method in trajectory.h
    // 			       theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
    // 			       charge, 
    // 			       vtx,
    // 			       mom,
    // 			       cov);
    AlgoProduct aProduct(theTraj,theTrack);
    LogDebug("TrackProducer") <<"track done1\n";
    algoResults.push_back(aProduct);
    LogDebug("TrackProducer") <<"track done2\n";
    
    return true;
  } 
  else  return false;
}
