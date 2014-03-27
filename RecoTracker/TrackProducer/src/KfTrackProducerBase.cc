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

#include "RecoTracker/TrackProducer/interface/ClusterRemovalRefSetter.h"
#include "TrajectoryToResiduals.h"

//TODO jaldeaar REMOVE PRINTS
void print(reco::HitPattern track_hitPattern, reco::HitPattern::HitCategory category);
#include <iostream>
void KfTrackProducerBase::putInEvt(edm::Event& evt,
				   const Propagator* prop,
				   const MeasurementTracker* measTk,
				   std::auto_ptr<TrackingRecHitCollection>& selHits,
				   std::auto_ptr<reco::TrackCollection>& selTracks,
				   std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
				   std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
				   AlgoProductCollection& algoResults)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
  edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;

  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    if(trajectoryInEvent_) {
      selTrajectories->push_back(*theTraj);
      iTjRef++;
    }

    // const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits(useSplitting);  // NO: the return type in Trajectory is by VALUE
    TrajectoryFitter::RecHitContainer transHits = theTraj->recHits(useSplitting);

    reco::Track * theTrack = (*i).second.first;
    
    // Hits are going to be re-sorted along momentum few lines later. 
    // Therefore the direction stored in the TrackExtra 
    // has to be "alongMomentum" as well. Anyway, this direction can be differnt from the one of the orignal
    // seed! The name seedDirection() for the Track's method (and the corresponding data member) is
    // misleading and should be changed into something like "hitsDirection()". TO BE FIXED!
    PropagationDirection seedDir = alongMomentum;

    LogDebug("TrackProducer") << "In KfTrackProducerBase::putInEvt - seedDir=" << seedDir;

    reco::Track t = * theTrack;
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

    std::cout << "Setting second hit pattern for track with pt=" << track.pt() << " and algo=" << track.algo() << std::endl;
    //======= I want to set the second hitPattern here =============
    if (theSchool.isValid())
      {
        edm::Handle<MeasurementTrackerEvent> mte;
        evt.getByToken(mteSrc_, mte);
	NavigationSetter setter( *theSchool );
	setSecondHitPattern(theTraj,track,prop,&*mte);
      }
    //==============================================================
    
    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
						 outertsos.curvilinearError(), outerId,
						 innertsos.curvilinearError(), innerId,
    						 seedDir, theTraj->seedRef()));


    reco::TrackExtra & tx = selTrackExtras->back();
    
    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collisions
    std::cout << "Setting first hit pattern for track with pt=" << track.pt() << " and algo=" << track.algo() << std::endl;
    if (theTraj->direction() == alongMomentum) {
        for(TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
                j != transHits.end(); j++) {
            if ((**j).hit() != 0){
                TrackingRecHit *hit = (**j).hit()->clone();
                track.appendHitPattern(*hit);
                selHits->push_back(hit);
                tx.add(TrackingRecHitRef(rHits, hidx++));
            }
        }
    }else{
        for(TrajectoryFitter::RecHitContainer::const_iterator j = transHits.end() - 1;
                j != transHits.begin() - 1; --j) {
            if ((**j).hit() != 0){
                TrackingRecHit * hit = (**j).hit()->clone();
                track.appendHitPattern(*hit);
                selHits->push_back(hit);
                tx.add(TrackingRecHitRef(rHits, hidx++));
            }
        }
    }

    std::cout << "########################### hitPattern ###########################" << std::endl;
    print(track.getHitPattern(), reco::HitPattern::TRACK_HITS);
    std::cout << "########################### trackerExpectedHitsInner ###########################" << std::endl;
    print(track.getHitPattern(), reco::HitPattern::MISSING_INNER_HITS);
    std::cout << "########################### trackerExpectedHitsOuter ###########################" << std::endl;
    print(track.getHitPattern(), reco::HitPattern::MISSING_OUTER_HITS);

    tx.setResiduals(trajectoryToResiduals(*theTraj));

    delete theTrack;
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
  
  
  rTracks_ = evt.put( selTracks );
  evt.put( selTrackExtras );
  evt.put( selHits );

  if(trajectoryInEvent_) {
    edm::OrphanHandle<std::vector<Trajectory> > rTrajs = evt.put(selTrajectories);

    // Now Create traj<->tracks association map
    std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
    for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); 
          i != tjTkMap.end(); i++ ) {
      edm::Ref<std::vector<Trajectory> > trajRef( rTrajs, (*i).first );
      edm::Ref<reco::TrackCollection>    tkRef( rTracks_, (*i).second );
      trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
                            edm::Ref<reco::TrackCollection>( rTracks_, (*i).second ) );
    }
    evt.put( trajTrackMap );
  }
}
void print(reco::HitPattern track_hitPattern, reco::HitPattern::HitCategory category)
{
    using namespace std;
    cout << "numberOfHits ";
    cout << track_hitPattern.numberOfHits(category);
    cout << endl;

    cout << "numberOfValidHits ";
    cout << track_hitPattern.numberOfValidHits(category) << ' ';
    cout << track_hitPattern.numberOfValidTrackerHits(category) << ' ';
    cout << track_hitPattern.numberOfValidMuonHits(category) << ' ';
    cout << track_hitPattern.numberOfValidPixelHits(category) << ' ';
    cout << track_hitPattern.numberOfValidPixelBarrelHits(category) << ' ';
    cout << track_hitPattern.numberOfValidPixelEndcapHits(category) << ' ';
    cout << track_hitPattern.numberOfValidStripHits(category) << ' ';
    cout << track_hitPattern.numberOfValidStripTIBHits(category) << ' ';
    cout << track_hitPattern.numberOfValidStripTIDHits(category) << ' ';
    cout << track_hitPattern.numberOfValidStripTOBHits(category) << ' ';
    cout << track_hitPattern.numberOfValidStripTECHits(category) << ' ';
    cout << track_hitPattern.numberOfValidMuonDTHits(category) << ' ';
    cout << track_hitPattern.numberOfValidMuonCSCHits(category) << ' ';
    cout << track_hitPattern.numberOfValidMuonRPCHits(category) << ' ';
    cout << endl;

    cout << "numberOfLostHits ";
    cout << track_hitPattern.numberOfLostHits(category) << ' ';
    cout << track_hitPattern.numberOfLostTrackerHits(category) << ' ';
    cout << track_hitPattern.numberOfLostMuonHits(category) << ' ';
    cout << track_hitPattern.numberOfLostPixelHits(category) << ' ';
    cout << track_hitPattern.numberOfLostPixelBarrelHits(category) << ' ';
    cout << track_hitPattern.numberOfLostPixelEndcapHits(category) << ' ';
    cout << track_hitPattern.numberOfLostStripHits(category) << ' ';
    cout << track_hitPattern.numberOfLostStripTIBHits(category) << ' ';
    cout << track_hitPattern.numberOfLostStripTIDHits(category) << ' ';
    cout << track_hitPattern.numberOfLostStripTOBHits(category) << ' ';
    cout << track_hitPattern.numberOfLostStripTECHits(category) << ' ';
    cout << track_hitPattern.numberOfLostMuonDTHits(category) << ' ';
    cout << track_hitPattern.numberOfLostMuonCSCHits(category) << ' ';
    cout << track_hitPattern.numberOfLostMuonRPCHits(category) << ' ';
    cout << endl;

    cout << "numberOfBadHits ";
    cout << track_hitPattern.numberOfBadHits(category) << ' ';
    cout << track_hitPattern.numberOfBadMuonHits(category) << ' ';
    cout << track_hitPattern.numberOfBadMuonDTHits(category) << ' ';
    cout << track_hitPattern.numberOfBadMuonCSCHits(category) << ' ';
    cout << track_hitPattern.numberOfBadMuonRPCHits(category) << ' ';
    cout << track_hitPattern.numberOfInactiveHits(category) << ' ';
    cout << track_hitPattern.numberOfInactiveTrackerHits(category) << ' ';
    cout << track_hitPattern.numberOfValidStripLayersWithMonoAndStereo(category);
    cout << endl;

    cout << "layersWithMeasurements ";
    cout << track_hitPattern.trackerLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.pixelLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.stripLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.pixelBarrelLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.pixelEndcapLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.stripTIBLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.stripTIDLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.stripTOBLayersWithMeasurement(category) << ' ';
    cout << track_hitPattern.stripTECLayersWithMeasurement(category) << ' ';
    cout << endl;

    cout << "WithoutMeasurements ";
    cout << track_hitPattern.trackerLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.pixelLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.stripLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.pixelBarrelLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.pixelEndcapLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.stripTIBLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.stripTIDLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.stripTOBLayersWithoutMeasurement(category) << ' ';
    cout << track_hitPattern.stripTECLayersWithoutMeasurement(category) << ' ';
    cout << endl;

    cout << "LayersTotallyOffOrBad ";
    cout << track_hitPattern.trackerLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.pixelLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.stripLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.pixelBarrelLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.pixelEndcapLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.stripTIBLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.stripTIDLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.stripTOBLayersTotallyOffOrBad(category) << ' ';
    cout << track_hitPattern.stripTECLayersTotallyOffOrBad(category) << ' ';
    cout << endl;

    cout << "LayersNull ";
    cout << track_hitPattern.trackerLayersNull(category) << ' ';
    cout << track_hitPattern.pixelLayersNull(category) << ' ';
    cout << track_hitPattern.stripLayersNull(category) << ' ';
    cout << track_hitPattern.pixelBarrelLayersNull(category) << ' ';
    cout << track_hitPattern.pixelEndcapLayersNull(category) << ' ';
    cout << track_hitPattern.stripTIBLayersNull(category) << ' ';
    cout << track_hitPattern.stripTIDLayersNull(category) << ' ';
    cout << track_hitPattern.stripTOBLayersNull(category) << ' ';
    cout << track_hitPattern.stripTECLayersNull(category) << ' ';
    cout << endl;

    //muon stations missing
    cout << "muon stuff ";
    cout << track_hitPattern.muonStationsWithValidHits(category) << ' ';
    cout << track_hitPattern.muonStationsWithBadHits(category) << ' ';
    cout << track_hitPattern.muonStationsWithAnyHits(category) << ' ';
    cout << track_hitPattern.dtStationsWithValidHits(category) << ' ';
    cout << track_hitPattern.dtStationsWithBadHits(category) << ' ';
    cout << track_hitPattern.dtStationsWithAnyHits(category) << ' ';
    cout << track_hitPattern.cscStationsWithValidHits(category) << ' ';
    cout << track_hitPattern.cscStationsWithBadHits(category) << ' ';
    cout << track_hitPattern.cscStationsWithAnyHits(category) << ' ';
    cout << track_hitPattern.rpcStationsWithValidHits(category) << ' ';
    cout << track_hitPattern.rpcStationsWithBadHits(category) << ' ';
    cout << track_hitPattern.rpcStationsWithAnyHits(category) << ' ';
    //missing  track_hitPattern.innermostMuonStationWithHits(int hitType)
    cout << track_hitPattern.innermostMuonStationWithValidHits(category) << ' ';
    cout << track_hitPattern.innermostMuonStationWithBadHits(category) << ' ';
    cout << track_hitPattern.innermostMuonStationWithAnyHits(category) << ' ';
    //missing cout << track_hitPattern.outermostMuonStationWithHits(int hitType)
    cout << track_hitPattern.outermostMuonStationWithValidHits(category) << ' ';
    cout << track_hitPattern.outermostMuonStationWithBadHits(category) << ' ';
    cout << track_hitPattern.outermostMuonStationWithAnyHits(category) << ' ';
    cout << track_hitPattern.numberOfDTStationsWithRPhiView(category) << ' ';
    cout << track_hitPattern.numberOfDTStationsWithRZView(category) << ' ';
    cout << track_hitPattern.numberOfDTStationsWithBothViews(category) << ' ';
    cout << endl;
}

