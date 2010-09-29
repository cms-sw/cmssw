#include "RecoTracker/TrackProducer/plugins/GsfTrackProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"

GsfTrackProducer::GsfTrackProducer(const edm::ParameterSet& iConfig):
  GsfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
		       iConfig.getParameter<bool>("useHitsSplitting")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<edm::InputTag>( "src" ), iConfig.getParameter<edm::InputTag>( "beamSpot" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
//   string a = alias_;
//   a.erase(a.size()-6,a.size());
  //register your products
  produces<reco::GsfTrackCollection>().setBranchAlias( alias_ + "GsfTracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<reco::GsfTrackExtraCollection>().setBranchAlias( alias_ + "GsfTrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajGsfTrackAssociationCollection>();

}


void GsfTrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("GsfTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::GsfTrackCollection> outputTColl(new reco::GsfTrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<reco::GsfTrackExtraCollection> outputGsfTEColl(new reco::GsfTrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker>  theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  try{  
    edm::Handle<TrackCandidateCollection> theTCCollection;
    getFromEvt(theEvent,theTCCollection,bs);
    
    //
    //run the algorithm  
    //
    LogDebug("GsfTrackProducer") << "run the algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
			     theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);
  } catch (cms::Exception &e){ edm::LogInfo("GsfTrackProducer") << "cms::Exception caught!!!" << "\n" << e << "\n"; throw; }
  //
  //put everything in the event
  putInEvt(theEvent, thePropagator.product(), theMeasTk.product(), outputRHColl, outputTColl, outputTEColl, outputGsfTEColl,
	   outputTrajectoryColl, algoResults, bs);
  LogDebug("GsfTrackProducer") << "end" << "\n";
}


// std::vector<reco::TransientTrack> GsfTrackProducer::getTransient(edm::Event& theEvent, const edm::EventSetup& setup)
// {
//   edm::LogInfo("GsfTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
//   //
//   // create empty output collections
//   //
//   std::vector<reco::TransientTrack> ttks;

//   //
//   //declare and get stuff to be retrieved from ES
//   //
//   edm::ESHandle<TrackerGeometry> theG;
//   edm::ESHandle<MagneticField> theMF;
//   edm::ESHandle<TrajectoryFitter> theFitter;
//   edm::ESHandle<Propagator> thePropagator;
//   edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
//   getFromES(setup,theG,theMF,theFitter,thePropagator,theBuilder);

//   //
//   //declare and get TrackColection to be retrieved from the event
//   //
//   AlgoProductCollection algoResults;
//   try{  
//     edm::Handle<TrackCandidateCollection> theTCCollection;
//     getFromEvt(theEvent,theTCCollection);
    
//     //
//     //run the algorithm  
//     //
//     LogDebug("GsfTrackProducer") << "run the algorithm" << "\n";
//     theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
// 			     theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
//   } catch (cms::Exception &e){ edm::LogInfo("GsfTrackProducer") << "cms::Exception caught!!!" << "\n" << e << "\n"; throw;}


//   for (AlgoProductCollection::iterator prod=algoResults.begin();prod!=algoResults.end(); prod++){
//     ttks.push_back( reco::TransientTrack(*((*prod).second),thePropagator.product()->magneticField() ));
//   }

//   LogDebug("GsfTrackProducer") << "end" << "\n";

//   return ttks;
// }


// void 
// GsfTrackProducer::putInEvt(edm::Event& evt,
// 			      std::auto_ptr<TrackingRecHitCollection>& selHits,
// 			      std::auto_ptr<reco::GsfTrackCollection>& selTracks,
// 			      std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
// 			      std::auto_ptr<reco::GsfTrackExtraCollection>& selGsfTrackExtras,
// 			      std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
// 			      AlgoProductCollection& algoResults)
// {

//   TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
//   reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();
//   reco::GsfTrackExtraRefProd rGsfTrackExtras = evt.getRefBeforePut<reco::GsfTrackExtraCollection>();
//   reco::GsfTrackRefProd rTracks = evt.getRefBeforePut<reco::GsfTrackCollection>();

//   edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
//   edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
//   edm::Ref<reco::GsfTrackExtraCollection>::key_type idxGsf = 0;
//   for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
//     Trajectory * theTraj = (*i).first;
//     if(trajectoryInEvent_) selTrajectories->push_back(*theTraj);
//     const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();

//     reco::GsfTrack * theTrack = (*i).second.first;
//     PropagationDirection seedDir = (*i).second.second;  
//     //     if( ) {
//     reco::GsfTrack t = * theTrack;
//     selTracks->push_back( t );
    
//     //sets the outermost and innermost TSOSs
//     TrajectoryStateOnSurface outertsos;
//     TrajectoryStateOnSurface innertsos;
//     unsigned int innerId, outerId;
//     if (theTraj->direction() == alongMomentum) {
//       outertsos = theTraj->lastMeasurement().updatedState();
//       innertsos = theTraj->firstMeasurement().updatedState();
//       outerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
//       innerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
//     } else { 
//       outertsos = theTraj->firstMeasurement().updatedState();
//       innertsos = theTraj->lastMeasurement().updatedState();
//       outerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
//       innerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
//    }

//     GlobalPoint v = outertsos.globalParameters().position();
//     GlobalVector p = outertsos.globalParameters().momentum();
//     math::XYZVector outmom( p.x(), p.y(), p.z() );
//     math::XYZPoint  outpos( v.x(), v.y(), v.z() );
//     v = innertsos.globalParameters().position();
//     p = innertsos.globalParameters().momentum();
//     math::XYZVector inmom( p.x(), p.y(), p.z() );
//     math::XYZPoint  inpos( v.x(), v.y(), v.z() );

//     reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );
//     reco::GsfTrack & track = selTracks->back();
//     track.setExtra( teref );
//     selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true,
// 						 inpos, inmom, true,
// 						 outertsos.curvilinearError(), outerId, 
// 						 innertsos.curvilinearError(), innerId,
// 						 seedDir,theTraj->seedRef()));

//     reco::TrackExtra & tx = selTrackExtras->back();
//     size_t i = 0;
//     for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
// 	 j != transHits.end(); j ++ ) {
//       TrackingRecHit * hit = (**j).hit()->clone();
//       track.setHitPattern( * hit, i ++ );
//       selHits->push_back( hit );
//       tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
//     }

//     //build the GsfTrackExtra
//     std::vector<reco::GsfComponent5D> outerStates;
//     outerStates.reserve(outertsos.components().size());
//     fillStates(outertsos,outerStates);
//     std::vector<reco::GsfComponent5D> innerStates;
//     innerStates.reserve(innertsos.components().size());
//     fillStates(innertsos,innerStates);

//     reco::GsfTrackExtraRef terefGsf = reco::GsfTrackExtraRef ( rGsfTrackExtras, idxGsf ++ );
//     track.setGsfExtra( terefGsf );
//     selGsfTrackExtras->push_back( reco::GsfTrackExtra (outerStates, outertsos.localParameters().pzSign(),
// 						       innerStates, innertsos.localParameters().pzSign()));

//     delete theTrack;
//     delete theTraj;
//   }
  
//   evt.put( selTracks );
//   evt.put( selTrackExtras );
//   evt.put( selGsfTrackExtras );
//   evt.put( selHits );
//   if(trajectoryInEvent_) evt.put(selTrajectories);
// }


// void
// GsfTrackProducer::fillStates (TrajectoryStateOnSurface tsos,
// 				  std::vector<reco::GsfComponent5D>& states) const
// {
// //   std::cout << "in fill states" << std::endl;
// //   if ( !tsos.isValid() ) {
// //     std::cout << std::endl << std::endl << "invalid tsos" << std::endl;
// //     return;
// //   }
//   reco::GsfComponent5D::ParameterVector pLocS;
//   reco::GsfComponent5D::CovarianceMatrix cLocS;
//   std::vector<TrajectoryStateOnSurface> components(tsos.components());
//   for ( std::vector<TrajectoryStateOnSurface>::const_iterator i=components.begin();
// 	  i!=components.end(); ++i ) {
// //     if ( !(*i).isValid() ) {
// //       std::cout << std::endl << "invalid component" << std::endl;
// //       continue;
// //     }
//     // Unneeded hack ... now we have SMatrix in tracking too
//     // const AlgebraicVector& pLoc = i->localParameters().vector();
//     // for ( int j=0; j<reco::GsfTrackExtra::dimension; ++j )  pLocS(j) = pLoc[j];
//     // const AlgebraicSymMatrix& cLoc = i->localError().matrix();
//     // for ( int j1=0; j1<reco::GsfTrack::dimension; ++j1 )
//       // for ( int j2=0; j2<=j1; ++j2 )  cLocS(j1,j2) = cLoc[j1][j2];
//     // states.push_back(reco::GsfComponent5D(i->weight(),pLocS,cLocS));
    
//     states.push_back(reco::GsfComponent5D(i->weight(),i->localParameters().vector(),i->localError().matrix()));
//   }
// //   std::cout << "end fill states" << std::endl;
// }
