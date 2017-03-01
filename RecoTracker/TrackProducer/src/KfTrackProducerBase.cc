#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
// system include files
#include <memory>
// user include files
#include "DataFormats/TrackReco/interface/TrackResiduals.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalRefSetter.h"
#include "TrajectoryToResiduals.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"


void KfTrackProducerBase::putInEvt(edm::Event& evt,
				   const Propagator* prop,
				   const MeasurementTracker* measTk,
				   std::unique_ptr<TrackingRecHitCollection>& selHits,
				   std::unique_ptr<reco::TrackCollection>& selTracks,
				   std::unique_ptr<reco::TrackExtraCollection>& selTrackExtras,
				   std::unique_ptr<std::vector<Trajectory> >&   selTrajectories,
				   std::unique_ptr<std::vector<int> >& indecesInput,
				   AlgoProductCollection& algoResults,
				   TransientTrackingRecHitBuilder const * hitBuilder,
                                   const TrackerTopology *ttopo,
                                   int BeforeOrAfter)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
  edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;

  selTracks->reserve(algoResults.size());
  selTrackExtras->reserve(algoResults.size());
  if(trajectoryInEvent_) selTrajectories->reserve(algoResults.size());

  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    auto theTraj = (*i).trajectory;
    (*indecesInput).push_back((*i).indexInput);
    if(trajectoryInEvent_) {
      selTrajectories->push_back(*theTraj);
      iTjRef++;
    }


    auto theTrack = (*i).track;
    
    // Hits are going to be re-sorted along momentum few lines later. 
    // Therefore the direction stored in the TrackExtra 
    // has to be "alongMomentum" as well. Anyway, this direction can be differnt from the one of the orignal
    // seed! The name seedDirection() for the Track's method (and the corresponding data member) is
    // misleading and should be changed into something like "hitsDirection()". TO BE FIXED!

    PropagationDirection seedDir = alongMomentum;

    LogDebug("TrackProducer") << "In KfTrackProducerBase::putInEvt - seedDir=" << seedDir;


    selTracks->push_back(std::move(*theTrack));
    delete theTrack;
    iTkRef++;

    // Store indices in local map (starts at 0)
    if(trajectoryInEvent_) tjTkMap[iTjRef-1] = iTkRef-1;
    
    //sets the outermost and innermost TSOSs
    TrajectoryStateOnSurface outertsos;
    TrajectoryStateOnSurface innertsos;
    unsigned int innerId, outerId;
    
    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collision
    if (theTraj->direction() == alongMomentum) {
      outertsos = theTraj->lastMeasurement().updatedState();
      innertsos = theTraj->firstMeasurement().updatedState();
      outerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
      innerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
    } else { 
      outertsos = theTraj->firstMeasurement().updatedState();
      innertsos = theTraj->lastMeasurement().updatedState();
      outerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
      innerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
    }
    // ---
    //build the TrackExtra
    GlobalPoint v = outertsos.globalParameters().position();
    GlobalVector p = outertsos.globalParameters().momentum();
    math::XYZVector outmom( p.x(), p.y(), p.z() );
    math::XYZPoint  outpos( v.x(), v.y(), v.z() );
    v = innertsos.globalParameters().position();
    p = innertsos.globalParameters().momentum();
    math::XYZVector inmom( p.x(), p.y(), p.z() );
    math::XYZPoint  inpos( v.x(), v.y(), v.z() );

    reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );
    reco::Track & track = selTracks->back();
    track.setExtra( teref );
    //======= I want to set the second hitPattern here =============
    if (theSchool.isValid())
      {
        edm::Handle<MeasurementTrackerEvent> mte;
        evt.getByToken(mteSrc_, mte);
	// NavigationSetter setter( *theSchool );
	setSecondHitPattern(theTraj,track,prop,&*mte, ttopo);
      }
    //==============================================================
    
    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
						 outertsos.curvilinearError(), outerId,
						 innertsos.curvilinearError(), innerId,
    						 seedDir, theTraj->seedRef()));

    // FIXME will remove this obsolete config-param in a future PR 
    assert(!useSplitting);

    reco::TrackExtra & tx = selTrackExtras->back();
    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collisions
    reco::TrackExtra::TrajParams trajParams;
    reco::TrackExtra::Chi2sFive chi2s; 
    Traj2TrackHits t2t;
    auto ih = selHits->size();
    t2t(*theTraj,*selHits,trajParams,chi2s);
    auto ie = selHits->size();
    tx.setHits(rHits,ih,ie-ih);
    tx.setTrajParams(std::move(trajParams),std::move(chi2s));
    assert(tx.trajParams().size()==tx.recHitsSize());
    for (;ih<ie; ++ih) {
      auto const & hit = (*selHits)[ih];
      track.appendHitPattern(hit, *ttopo);
    }
    
    // ----
    tx.setResiduals(trajectoryToResiduals(*theTraj));

    delete theTraj;
  }

  // Now we can re-set refs to hits, as they have already been cloned
  if (rekeyClusterRefs_) {
      ClusterRemovalRefSetter refSetter(evt, clusterRemovalInfo_);
      for (TrackingRecHitCollection::iterator it = selHits->begin(), ed = selHits->end(); it != ed; ++it) {
          refSetter.reKey(&*it);
      }
  }

  LogTrace("TrackingRegressionTest") << "========== TrackProducer Info ===================";
  LogTrace("TrackingRegressionTest") << "number of finalTracks: " << selTracks->size();
  for (reco::TrackCollection::const_iterator it = selTracks->begin(); it != selTracks->end(); it++) {
    LogTrace("TrackingRegressionTest") << "track's n valid and invalid hit, chi2, pt, eta : " 
				       << it->found() << " , " 
				       << it->lost()  <<" , " 
				       << it->normalizedChi2() << " , "
				       << it->pt() << " , "
				       << it->eta() ;
  }
  LogTrace("TrackingRegressionTest") << "=================================================";
  
  selTracks->shrink_to_fit();
  selTrackExtras->shrink_to_fit();
  selHits->shrink_to_fit();
  indecesInput->shrink_to_fit();
  if(BeforeOrAfter == 1){
    rTracks_ = evt.put(std::move(selTracks), "beforeDAF" );
    evt.put(std::move(selTrackExtras), "beforeDAF");
    evt.put(std::move(indecesInput), "beforeDAF");
  } else if (BeforeOrAfter == 2){
    rTracks_ = evt.put(std::move(selTracks), "afterDAF" );
    evt.put( std::move(selTrackExtras), "afterDAF" );
    evt.put(std::move(indecesInput), "afterDAF");
  } else {
    rTracks_ = evt.put(std::move(selTracks) );
    evt.put(std::move(selTrackExtras) );
    evt.put(std::move(selHits) );
    evt.put(std::move(indecesInput));
  }


  if(trajectoryInEvent_ && BeforeOrAfter == 0) {
    selTrajectories->shrink_to_fit();
    edm::OrphanHandle<std::vector<Trajectory> > rTrajs = evt.put(std::move(selTrajectories));

    // Now Create traj<->tracks association map
    std::unique_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection(rTrajs, rTracks_) );
    for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); 
          i != tjTkMap.end(); i++ ) {
      edm::Ref<std::vector<Trajectory> > trajRef( rTrajs, (*i).first );
      edm::Ref<reco::TrackCollection>    tkRef( rTracks_, (*i).second );
      trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
                            edm::Ref<reco::TrackCollection>( rTracks_, (*i).second ) );
    }
    evt.put( std::move(trajTrackMap) );
  }
}

