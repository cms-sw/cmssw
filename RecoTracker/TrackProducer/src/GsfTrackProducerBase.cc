#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

void 
GsfTrackProducerBase::putInEvt(edm::Event& evt,
			       std::auto_ptr<TrackingRecHitCollection>& selHits,
			       std::auto_ptr<reco::GsfTrackCollection>& selTracks,
			       std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
			       std::auto_ptr<reco::GsfTrackExtraCollection>& selGsfTrackExtras,
			       std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
			       AlgoProductCollection& algoResults)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();
  reco::GsfTrackExtraRefProd rGsfTrackExtras = evt.getRefBeforePut<reco::GsfTrackExtraCollection>();
  reco::GsfTrackRefProd rTracks = evt.getRefBeforePut<reco::GsfTrackCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  edm::Ref<reco::GsfTrackExtraCollection>::key_type idxGsf = 0;
//   edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
//   edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
//   std::map<unsigned int, unsigned int> tjTkMap;

  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    if(trajectoryInEvent_) {
      selTrajectories->push_back(*theTraj);
//       iTjRef++;
    }
    const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();

    reco::GsfTrack * theTrack = (*i).second.first;
    PropagationDirection seedDir = (*i).second.second;  
    
    LogDebug("TrackProducer") << "In GsfTrackProducerBase::putInEvt - seedDir=" << seedDir;

    reco::GsfTrack t = * theTrack;
    selTracks->push_back( t );
//     iTkRef++;

//     // Store indices in local map (starts at 0)
//     if(trajectoryInEvent_) tjTkMap[iTjRef-1] = iTkRef-1;

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
    reco::GsfTrack & track = selTracks->back();
    track.setExtra( teref );


    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
						 outertsos.curvilinearError(), outerId,
						 innertsos.curvilinearError(), innerId,
						 seedDir,theTraj->seedRef()));


    reco::TrackExtra & tx = selTrackExtras->back();


    size_t i = 0;
    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collisions
    if (theTraj->direction() == alongMomentum) {
      for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
	   j != transHits.end(); j ++ ) {
	if ((**j).hit()!=0){
	  TrackingRecHit * hit = (**j).hit()->clone();
	  track.setHitPattern( * hit, i ++ );
	  selHits->push_back( hit );
	  tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
	}
      }
    }else{
      for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.end()-1;
	   j != transHits.begin()-1; --j ) {
	if ((**j).hit()!=0){
	  TrackingRecHit * hit = (**j).hit()->clone();
	  track.setHitPattern( * hit, i ++ );
	  selHits->push_back( hit );
	tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
	}
      }
    }
    // ----


    //build the GsfTrackExtra
    std::vector<reco::GsfComponent5D> outerStates;
    outerStates.reserve(outertsos.components().size());
    fillStates(outertsos,outerStates);
    std::vector<reco::GsfComponent5D> innerStates;
    innerStates.reserve(innertsos.components().size());
    fillStates(innertsos,innerStates);

    reco::GsfTrackExtraRef terefGsf = reco::GsfTrackExtraRef ( rGsfTrackExtras, idxGsf ++ );
    track.setGsfExtra( terefGsf );
    selGsfTrackExtras->push_back( reco::GsfTrackExtra (outerStates, outertsos.localParameters().pzSign(),
						       innerStates, innertsos.localParameters().pzSign()));

    delete theTrack;
    delete theTraj;
  }

  LogTrace("TrackingRegressionTest") << "========== TrackProducer Info ===================";
  LogTrace("TrackingRegressionTest") << "number of finalGsfTracks: " << selTracks->size();
  for (reco::GsfTrackCollection::const_iterator it = selTracks->begin(); it != selTracks->end(); it++) {
    LogTrace("TrackingRegressionTest") << "track's n valid and invalid hit, chi2, pt : " 
				       << it->found() << " , " 
				       << it->lost()  <<" , " 
				       << it->normalizedChi2() << " , "
				       << it->pt();
  }
  LogTrace("TrackingRegressionTest") << "=================================================";
  

  rTracks_ = evt.put( selTracks );
  evt.put( selTrackExtras );
  evt.put( selGsfTrackExtras );
  evt.put( selHits );

  if(trajectoryInEvent_) {
    edm::OrphanHandle<std::vector<Trajectory> > rTrajs = evt.put(selTrajectories);

//     // Now Create traj<->tracks association map
//     std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
//     for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); 
//           i != tjTkMap.end(); i++ ) {
//       edm::Ref<std::vector<Trajectory> > trajRef( rTrajs, (*i).first );
//       edm::Ref<reco::GsfTrackCollection>    tkRef( rTracks_, (*i).second );
//       trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
//                             edm::Ref<reco::GsfTrackCollection>( rTracks_, (*i).second ) );
//     }
//     evt.put( trajTrackMap );
  }
}

void
GsfTrackProducerBase::fillStates (TrajectoryStateOnSurface tsos,
				  std::vector<reco::GsfComponent5D>& states) const
{
  //   std::cout << "in fill states" << std::endl;
  //   if ( !tsos.isValid() ) {
  //     std::cout << std::endl << std::endl << "invalid tsos" << std::endl;
  //     return;
  //   }
  reco::GsfComponent5D::ParameterVector pLocS;
  reco::GsfComponent5D::CovarianceMatrix cLocS;
  std::vector<TrajectoryStateOnSurface> components(tsos.components());
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator i=components.begin();
	i!=components.end(); ++i ) {
    //     if ( !(*i).isValid() ) {
    //       std::cout << std::endl << "invalid component" << std::endl;
    //       continue;
    //     }
    // Unneeded hack ... now we have SMatrix in tracking too
    // const AlgebraicVector& pLoc = i->localParameters().vector();
    // for ( int j=0; j<reco::GsfTrackExtra::dimension; ++j )  pLocS(j) = pLoc[j];
    // const AlgebraicSymMatrix& cLoc = i->localError().matrix();
    // for ( int j1=0; j1<reco::GsfTrack::dimension; ++j1 )
    // for ( int j2=0; j2<=j1; ++j2 )  cLocS(j1,j2) = cLoc[j1][j2];
    // states.push_back(reco::GsfComponent5D(i->weight(),pLocS,cLocS));
    
    states.push_back(reco::GsfComponent5D(i->weight(),i->localParameters().vector(),i->localError().matrix()));
  }
  //   std::cout << "end fill states" << std::endl;
}
