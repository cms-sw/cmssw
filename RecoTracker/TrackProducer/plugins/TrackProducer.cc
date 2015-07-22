#include "RecoTracker/TrackProducer/plugins/TrackProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

TrackProducer::TrackProducer(const edm::ParameterSet& iConfig):
  KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
		      iConfig.getParameter<bool>("useHitsSplitting")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>( "src" )), 
          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );

  if ( iConfig.exists("clusterRemovalInfo") ) {
        edm::InputTag tag = iConfig.getParameter<edm::InputTag>("clusterRemovalInfo");
        if (!(tag == edm::InputTag())) { setClusterRemovalInfo( tag ); }
  }

  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();

}


void TrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  LogDebug("TrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection>    outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTEColl(new reco::TrackExtraCollection);
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

  edm::ESHandle<TrackerTopology> httopo;
  setup.get<TrackerTopologyRcd>().get(httopo);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  edm::Handle<TrackCandidateCollection> theTCCollection;
  reco::BeamSpot bs;
  getFromEvt(theEvent,theTCCollection,bs);
  //protect against missing product  
  if (theTCCollection.failedToGet()){
    edm::LogError("TrackProducer") <<"could not get the TrackCandidateCollection.";} 
  else{
    LogDebug("TrackProducer") << "run the algorithm" << "\n";
    try{  
      theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
			       theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);
    } catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithCandidate." << "\n" << e << "\n"; throw;}
  }
  
  //put everything in the event
  putInEvt(theEvent, thePropagator.product(),theMeasTk.product(), outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults, theBuilder.product(), httopo.product());
  LogDebug("TrackProducer") << "end" << "\n";
}


std::vector<reco::TransientTrack> TrackProducer::getTransient(edm::Event& theEvent, const edm::EventSetup& setup)
{
  LogDebug("TrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::vector<reco::TransientTrack> ttks;

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
  edm::Handle<TrackCandidateCollection> theTCCollection;
  reco::BeamSpot bs;
  getFromEvt(theEvent,theTCCollection,bs);
  //protect against missing product  
  if (theTCCollection.failedToGet()){
    edm::LogError("TrackProducer") <<"could not get the TrackCandidateCollection.";}
  else{
    LogDebug("TrackProducer") << "run the algorithm" << "\n";
    try{  
      theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
			       theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);
    }
    catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithCandidate." << "\n" << e << "\n"; throw; }
  }
  ttks.reserve(algoResults.size());  
  for (AlgoProductCollection::iterator prod=algoResults.begin();prod!=algoResults.end(); prod++){
    ttks.push_back( reco::TransientTrack(*((*prod).second.first),thePropagator.product()->magneticField() ));
  }

  LogDebug("TrackProducer") << "end" << "\n";

  return ttks;
}


