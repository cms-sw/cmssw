#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "FWCore/Framework/interface/OrphanHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"

void TrackProducerAlgorithm::run(const TrackingGeometry * theG,
				 const MagneticField * theMF,
				 TrackCandidateCollection& theTCCollection,
				 const KFTrajectoryFitter * theFitter,
				 std::auto_ptr<reco::TrackCollection>& tcoll,
				 std::auto_ptr<reco::TrackExtraCollection>& tecoll)
{
  int cont = 0;
  for (TrackCandidateCollection::iterator i=theTCCollection.begin(); i!=theTCCollection.end();i++)
    {
      TrackCandidate * theTC = &(*i);//maybe it is better to clone
      //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
      TrajectoryStateTransform transformer;
      PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
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

      int ndof=0;
      
      for (edm::OwnVector<TrackingRecHit>::const_iterator i=theTC->recHits().first;
	   i!=theTC->recHits().second; i++){
	hits.push_back(builder->build(&(*i) ));
	ndof = ndof + int( (i->dimension())*(i->weight()) );
      }
      
      delete builder;

      ndof = ndof - 5;
      
      //variable declarations
      std::vector<Trajectory> trajVec;
      reco::Track * theTrack;
      reco::TrackExtra * theTrackExtra;
      Trajectory * theTraj; 
      
      //perform the fit: the result's size is 1 if it succeded, 0 if fails
      trajVec = theFitter->fit(theTC->seed(), hits, theTSOS);
      
      TransverseImpactPointExtrapolator * tipe;
      TrajectoryStateOnSurface tsos, outertsos, innertsos;
      
      if (trajVec.size() != 0){
	
	tipe = new TransverseImpactPointExtrapolator(*theFitter->propagator());
	
	theTraj = &( trajVec.front() );
	
	//sets the outermost and innermost TSOSs
	if (theTraj->direction() == alongMomentum) {
	  outertsos = theTraj->lastMeasurement().updatedState();
	  innertsos = theTraj->firstMeasurement().updatedState();
	} else { 
	  outertsos = theTraj->firstMeasurement().updatedState();
	  innertsos = theTraj->lastMeasurement().updatedState();
	}
	
	//extrapolate the innermost state to the point of closest approach to the beamline
	tsos = tipe->extrapolate(*(innertsos.freeState()), 
				 GlobalPoint(0,0,0) );
	
	//compute parameters needed to build a Track from a Trajectory    
	int charge = tsos.charge();
	const GlobalTrajectoryParameters& gp = tsos.globalParameters();
	GlobalPoint v = gp.position();
	GlobalVector p = gp.momentum();
	const CartesianTrajectoryError& cte = tsos.cartesianError();
	AlgebraicSymMatrix m = cte.matrix();
	math::Error<6> cov;
	for( int i = 0; i < 6; ++i )
	  for( int j = 0; j <= i; ++j )
	    cov( i, j ) = m.fast( i + 1 , j + 1 );
	math::XYZVector mom( p.x(), p.y(), p.z() );
	math::XYZPoint  vtx( v.x(), v.y(), v.z() );   
	
	//build the Track(chiSquared, ndof, found, invalid, lost, q, vertex, momentum, covariance)
	theTrack = new reco::Track(theTraj->chiSquared(), 
				   ndof,//FIXME fix weight() in TrackingRecHit
				   theTraj->foundHits(),//FIXME to be fixed in Trajectory.h
				   0, //FIXME no corresponding method in trajectory.h
				   theTraj->lostHits(),//FIXME to be fixed in Trajectory.h
				   charge, 
				   vtx,
				   mom,
				   cov);
	
	//build the TrackExtra
	v = outertsos.globalParameters().position();
	p = outertsos.globalParameters().momentum();
	math::XYZVector outmom( p.x(), p.y(), p.z() );
	math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
	theTrackExtra = new reco::TrackExtra(outpos, outmom, true);


	//fill the TrackExtra with TrackingRecHitRef	
	int cc = 0;	
	const edm::ProductID pid(0);
	TrackingRecHitCollection trhcoll;//needed by OrphanHandle
	for (edm::OwnVector<TransientTrackingRecHit>::iterator i=theTraj->recHits().begin();
	     i!=theTraj->recHits().end(); i++){
	  trhcoll.push_back(&(*i));
	}
	edm::OrphanHandle<TrackingRecHitCollection> trhcollOH(&trhcoll,pid);//needed by TrackingRecHitRef
	for (TrackingRecHitCollection::const_iterator i=trhcollOH.product()->begin();
	     i!=trhcollOH.product()->end(); i++){
	  theTrackExtra->add(TrackingRecHitRef(trhcollOH,cc));
	  cc++;
	}

	
	//fill the TrackExtraCollection
	tecoll->push_back(*theTrackExtra);

	edm::OrphanHandle <reco::TrackExtraCollection> tecollOH(tecoll.get(),pid);

	//create a TrackExtraRef
	reco::TrackExtraRef  theTrackExtraRef(tecollOH,cont);
	
	//use the TrackExtraRef to assign the TrackExtra to the Track
	theTrack->setExtra(theTrackExtraRef);
	
	//fill the TrackExtraCollection
	tcoll->push_back(*theTrack);

	cont++;
      }
    }
}

