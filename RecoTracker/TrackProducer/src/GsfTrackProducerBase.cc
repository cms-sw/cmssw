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
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

void 
GsfTrackProducerBase::putInEvt(edm::Event& evt,
			       std::auto_ptr<TrackingRecHitCollection>& selHits,
			       std::auto_ptr<reco::GsfTrackCollection>& selTracks,
			       std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
			       std::auto_ptr<reco::GsfTrackExtraCollection>& selGsfTrackExtras,
			       std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
			       AlgoProductCollection& algoResults,
			       const reco::BeamSpot& bs)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();
  reco::GsfTrackExtraRefProd rGsfTrackExtras = evt.getRefBeforePut<reco::GsfTrackExtraCollection>();
  reco::GsfTrackRefProd rTracks = evt.getRefBeforePut<reco::GsfTrackCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  edm::Ref<reco::GsfTrackExtraCollection>::key_type idxGsf = 0;
  edm::Ref<reco::GsfTrackCollection>::key_type iTkRef = 0;
  edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;

  TSCBLBuilderNoMaterial tscblBuilder;
  
  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    if(trajectoryInEvent_) {
      selTrajectories->push_back(*theTraj);
      iTjRef++;
    }

    // const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits(useSplitting);  // NO: the return type in Trajectory is by VALUE
    TrajectoryFitter::RecHitContainer transHits = theTraj->recHits(useSplitting);
    reco::GsfTrack * theTrack = (*i).second.first;
    PropagationDirection seedDir = (*i).second.second;  
    
    LogDebug("TrackProducer") << "In GsfTrackProducerBase::putInEvt - seedDir=" << seedDir;

    reco::GsfTrack t = * theTrack;
    selTracks->push_back( t );
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
    
    //======= I want to set the second hitPattern here =============
    if (theSchool.isValid())
      {
	NavigationSetter setter( *theSchool );
	setSecondHitPattern(theTraj,track);
      }
    //==============================================================
    
    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
						 outertsos.curvilinearError(), outerId,
						 innertsos.curvilinearError(), innerId,
						 seedDir, theTraj->seedRef()));


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

    if ( innertsos.isValid() ) {
      GsfPropagatorAdapter gsfProp(AnalyticalPropagator(innertsos.magneticField(),anyDirection));
      TransverseImpactPointExtrapolator tipExtrapolator(gsfProp);
      fillMode(track,innertsos,gsfProp,tipExtrapolator,tscblBuilder,bs);
    }

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
				       << it->pt() << " , "
				       << it->eta() ;
  }
  LogTrace("TrackingRegressionTest") << "=================================================";
  

  rTracks_ = evt.put( selTracks );
  evt.put( selTrackExtras );
  evt.put( selGsfTrackExtras );
  evt.put( selHits );

  if(trajectoryInEvent_) {
    edm::OrphanHandle<std::vector<Trajectory> > rTrajs = evt.put(selTrajectories);

    // Now Create traj<->tracks association map
    std::auto_ptr<TrajGsfTrackAssociationCollection> trajTrackMap( new TrajGsfTrackAssociationCollection() );
    for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); 
	  i != tjTkMap.end(); i++ ) {
      edm::Ref<std::vector<Trajectory> > trajRef( rTrajs, (*i).first );
      edm::Ref<reco::GsfTrackCollection>    tkRef( rTracks_, (*i).second );
      trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
			    edm::Ref<reco::GsfTrackCollection>( rTracks_, (*i).second ) );
    }
    evt.put( trajTrackMap );
  }
}

void
GsfTrackProducerBase::fillStates (TrajectoryStateOnSurface tsos,
				  std::vector<reco::GsfComponent5D>& states) const
{
  reco::GsfComponent5D::ParameterVector pLocS;
  reco::GsfComponent5D::CovarianceMatrix cLocS;
  std::vector<TrajectoryStateOnSurface> components(tsos.components());
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator i=components.begin();
	i!=components.end(); ++i ) {
    states.push_back(reco::GsfComponent5D(i->weight(),i->localParameters().vector(),i->localError().matrix()));
  }
}

void
GsfTrackProducerBase::fillMode (reco::GsfTrack& track, const TrajectoryStateOnSurface innertsos,
				const Propagator& gsfProp,
				const TransverseImpactPointExtrapolator& tipExtrapolator,
				TrajectoryStateClosestToBeamLineBuilder& tscblBuilder,
				const reco::BeamSpot& bs) const
{
  // Get transverse impact parameter plane (from mean). This is a first approximation;
  // the mode is then extrapolated to the
  // final position closest to the beamline.
  GlobalPoint bsPos(bs.position().x()+(track.vz()-bs.position().z())*bs.dxdz(),
		    bs.position().y()+(track.vz()-bs.position().z())*bs.dydz(),
		    track.vz());
  TrajectoryStateOnSurface vtxTsos = tipExtrapolator.extrapolate(innertsos,bsPos);
  if ( !vtxTsos.isValid() )  vtxTsos = innertsos;
  // extrapolate mixture
  vtxTsos = gsfProp.propagate(innertsos,vtxTsos.surface());
  if ( !vtxTsos.isValid() )  return;              // failed (GsfTrack keeps mode = mean)
  // extract mode
  // build perigee parameters (for covariance to be stored)
  AlgebraicVector5 modeParameters;
  AlgebraicSymMatrix55 modeCovariance;
  // set parameters and variances for "mode" state (local parameters)
  for ( unsigned int iv=0; iv<5; ++iv ) {
    MultiGaussianState1D state1D = MultiGaussianStateTransform::multiState1D(vtxTsos,iv);
    GaussianSumUtilities1D utils(state1D);
    modeParameters(iv) = utils.mode().mean();
    modeCovariance(iv,iv) = utils.mode().variance();
    if ( !utils.modeIsValid() ) {
      // if mode calculation fails: use mean
      modeParameters(iv) = utils.mean();
      modeCovariance(iv,iv) = utils.variance();
    }
  }
  // complete covariance matrix
  // approximation: use correlations from mean
  const AlgebraicSymMatrix55& meanCovariance(vtxTsos.localError().matrix());
  for ( unsigned int iv1=0; iv1<5; ++iv1 ) {
    for ( unsigned int iv2=0; iv2<iv1; ++iv2 ) {
      double cov12 = meanCovariance(iv1,iv2) * 
	sqrt(modeCovariance(iv1,iv1)/meanCovariance(iv1,iv1)*
	     modeCovariance(iv2,iv2)/meanCovariance(iv2,iv2));
      modeCovariance(iv1,iv2) = modeCovariance(iv2,iv1) = cov12;
    }
  }
  TrajectoryStateOnSurface modeTsos(LocalTrajectoryParameters(modeParameters,
							      vtxTsos.localParameters().pzSign()),
				    LocalTrajectoryError(modeCovariance),
				    vtxTsos.surface(),
				    vtxTsos.magneticField(),
				    vtxTsos.surfaceSide());
  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*modeTsos.freeState(),bs);
  if ( !tscbl.isValid() )  return;            // failed (GsfTrack keeps mode = mean)
  //
  // extract state at PCA and create momentum vector and covariance matrix
  //
  FreeTrajectoryState fts = tscbl.trackStateAtPCA();
  GlobalVector tscblMom = fts.momentum();
  reco::GsfTrack::Vector mom(tscblMom.x(),tscblMom.y(),tscblMom.z());
  reco::GsfTrack::CovarianceMatrixMode cov;
  const AlgebraicSymMatrix55& tscblCov = fts.curvilinearError().matrix();
  for ( unsigned int iv1=0; iv1<reco::GsfTrack::dimensionMode; ++iv1 ) {
    for ( unsigned int iv2=0; iv2<reco::GsfTrack::dimensionMode; ++iv2 ) {
      cov(iv1,iv2) = tscblCov(iv1,iv2);
    }
  } 
  track.setMode(fts.charge(),mom,cov);
}
