#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

/// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"


//destructor
TrackProducerBase::~TrackProducerBase(){ }

// member functions
// ------------ method called to produce the data  ------------

void TrackProducerBase::getFromES(const edm::EventSetup& setup,
				  edm::ESHandle<TrackerGeometry>& theG,
				  edm::ESHandle<MagneticField>& theMF,
				  edm::ESHandle<TrajectoryFitter>& theFitter,
				  edm::ESHandle<Propagator>& thePropagator,
				  edm::ESHandle<TransientTrackingRecHitBuilder>& theBuilder)
{
  //
  //get geometry
  //
  LogDebug("TrackProducer") << "get geometry" << "\n";
  setup.get<TrackerDigiGeometryRecord>().get(theG);
  //
  //get magnetic field
  //
  LogDebug("TrackProducer") << "get magnetic field" << "\n";
  setup.get<IdealMagneticFieldRecord>().get(theMF);  
  //
  // get the fitter from the ES
  //
  LogDebug("TrackProducer") << "get the fitter from the ES" << "\n";
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  setup.get<TrackingComponentsRecord>().get(fitterName,theFitter);
  //
  // get also the propagator
  //
  LogDebug("TrackProducer") << "get also the propagator" << "\n";
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  setup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  //
  // get the builder
  //
  LogDebug("TrackProducer") << "get also the TransientTrackingRecHitBuilder" << "\n";
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  setup.get<TransientRecHitRecord>().get(builderName,theBuilder);

  

}

void TrackProducerBase::getFromEvt(edm::Event& theEvent,edm::Handle<TrackCandidateCollection>& theTCCollection)
{
  //
  //get the TrackCandidateCollection from the event
  //
  LogDebug("TrackProducer") << 
    "get the TrackCandidateCollection from the event, source is " << src_<<"\n";
  theEvent.getByLabel(src_,theTCCollection );
}

void TrackProducerBase::getFromEvt(edm::Event& theEvent,edm::Handle<reco::TrackCollection>& theTCollection)
{
  //
  //get the TrackCollection from the event
  //
  LogDebug("TrackProducer") << 
    "get the TrackCollection from the event, source is " << src_<<"\n";
  theEvent.getByLabel(src_,theTCollection );
}

void TrackProducerBase::putInEvt(edm::Event& evt,
				 std::auto_ptr<TrackingRecHitCollection>& selHits,
				 std::auto_ptr<reco::TrackCollection>& selTracks,
				 std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
				 AlgoProductCollection& algoResults)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();
  reco::TrackRefProd rTracks = evt.getRefBeforePut<reco::TrackCollection>();

  size_t idx = 0, hidx = 0;
//   for( TrackCollection::const_iterator trk = tracks->begin(); trk != tracks->end(); ++ trk ) {
  for(AlgoProductCollection::iterator i=algoResults.begin();
      i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();

    reco::Track * theTrack = (*i).second;
    
    //     if( ) {
    selTracks->push_back( *theTrack );
    
    reco::TrackExtra * theTrackExtra;
    //sets the outermost and innermost TSOSs
    TrajectoryStateOnSurface outertsos;
    TrajectoryStateOnSurface innertsos;
    if (theTraj->direction() == alongMomentum) {
      outertsos = theTraj->lastMeasurement().updatedState();
      innertsos = theTraj->firstMeasurement().updatedState();
    } else { 
      outertsos = theTraj->firstMeasurement().updatedState();
      innertsos = theTraj->lastMeasurement().updatedState();
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
    
    theTrackExtra = new reco::TrackExtra(outpos, outmom, true, inpos, inmom, true);
    reco::TrackExtraRef teref( rTrackExtras, idx ++ );
    selTracks->back().setExtra( teref );
    selTracks->back().setHitPattern(teref->recHits());
    selTrackExtras->push_back( *theTrackExtra );
    
//     reco::TrackExtra & tx = selTrackExtras->back();
    for(TrajectoryFitter::RecHitContainer::const_iterator j=transHits.begin();
	j!=transHits.end(); j++){
      selHits->push_back( ( ((**j).hit() )->clone()) );
      theTrackExtra->add( TrackingRecHitRef( rHits, hidx ++ ) );
    }
    //     }
    delete theTrackExtra;
    delete theTrack;
    delete theTraj;
  }
  
  evt.put( selTracks );
  evt.put( selTrackExtras );
  evt.put( selHits );
}

//   //
//   //first loop: create the full collection of TrackingRecHit
//   //
//   LogDebug("TrackProducer") << 
//     "first loop: create the full collection of TrackingRecHit" << "\n";
//   for(AlgoProductCollection::iterator i=algoResults.begin();
//       i!=algoResults.end();i++){
//     Trajectory * theTraj = (*i).first;
    
//     const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();
//     for(TrajectoryFitter::RecHitContainer::const_iterator j=transHits.begin();
// 	j!=transHits.end(); j++){
//       outputRHColl->push_back( ( ((**j).hit() )->clone()) );
//     }
    
//   }
//   //put the collection of TrackingRecHit in the event
//   LogDebug("TrackProducer") << 
//     "put the collection of TrackingRecHit in the event" << "\n";
  
//   edm::OrphanHandle <TrackingRecHitCollection> ohRH = theEvent.put( outputRHColl );
  
//   //
//   //second loop: create the collection of TrackExtra
//   //
//   LogDebug("TrackProducer") << 
//     "second loop: create the collection of TrackExtra" << "\n";
//   int cc = 0;	
//   for(AlgoProductCollection::iterator i=algoResults.begin();
//       i!=algoResults.end();i++){
    
//     Trajectory * theTraj = (*i).first;
    
//     reco::TrackExtra * theTrackExtra;
//     //sets the outermost and innermost TSOSs
//     TrajectoryStateOnSurface outertsos;
//     TrajectoryStateOnSurface innertsos;
//     if (theTraj->direction() == alongMomentum) {
//       outertsos = theTraj->lastMeasurement().updatedState();
//       innertsos = theTraj->firstMeasurement().updatedState();
//     } else { 
//       outertsos = theTraj->firstMeasurement().updatedState();
//       innertsos = theTraj->lastMeasurement().updatedState();
//     }
//     //build the TrackExtra
//     GlobalPoint v = outertsos.globalParameters().position();
//     GlobalVector p = outertsos.globalParameters().momentum();
//     math::XYZVector outmom( p.x(), p.y(), p.z() );
//     math::XYZPoint  outpos( v.x(), v.y(), v.z() );
//     v = innertsos.globalParameters().position();
//     p = innertsos.globalParameters().momentum();
//     math::XYZVector inmom( p.x(), p.y(), p.z() );
//     math::XYZPoint  inpos( v.x(), v.y(), v.z() );

//     theTrackExtra = new reco::TrackExtra(outpos, outmom, true, inpos, inmom, true);
    
    
//     //fill the TrackExtra with TrackingRecHitRef	
//     const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();
//     //    const edm::OwnVector<const TransientTrackingRecHit>& transHits = theTraj->recHits();
//     for(TrajectoryFitter::RecHitContainer::const_iterator j=transHits.begin();
// 	j!=transHits.end(); j++){
//       theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
//       cc++;
//     }
    
//     //fill the TrackExtraCollection
//     outputTEColl->push_back(*theTrackExtra);
//     delete theTrackExtra;
//   }
//   //put the collection of TrackExtra in the event
//   LogDebug("TrackProducer") << 
//     "put the collection of TrackExtra in the event" << "\n";
//   edm::OrphanHandle<reco::TrackExtraCollection> ohTE = theEvent.put(outputTEColl);
  
  
//   //
//   //third loop: create the collection of Tracks
//   //
//   LogDebug("RecoTracker/TrackProducer") << 
//     "third loop: create the collection of Tracks" << "\n";
//   cc = 0;
//   for(AlgoProductCollection::iterator i=algoResults.begin();
//       i!=algoResults.end();i++){
    
//     reco::Track * theTrack = (*i).second;
    
//     //create a TrackExtraRef
//     reco::TrackExtraRef  theTrackExtraRef(ohTE,cc);
//     theTrack->setHitPattern((*theTrackExtraRef).recHits());
    
//     //use the TrackExtraRef to assign the TrackExtra to the Track
//     theTrack->setExtra(theTrackExtraRef);
    
//     //fill the TrackCollection
//     outputTColl->push_back(*theTrack);
    
//     cc++;
//     delete theTrack;
//   }
//   //put the TrackCollection in the event
//   LogDebug("TrackProducer") << 
//     "put the TrackCollection in the event" << "\n";
//   theEvent.put(outputTColl);

//   for(AlgoProductCollection::iterator i=algoResults.begin();
//       i!=algoResults.end();i++){
//     Trajectory * theTraj = (*i).first;
//     Trajectory::DataContainer dc = theTraj->measurements();
// //     for (Trajectory::DataContainer::iterator j=dc.begin(); j!=dc.end(); j++) {
// //       delete j->recHit();
// //     }
//     delete theTraj;
//   }  
  
// }

