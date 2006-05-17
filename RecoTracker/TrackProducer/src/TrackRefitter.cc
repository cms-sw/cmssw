#include "RecoTracker/TrackProducer/interface/TrackRefitter.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

TrackRefitter::TrackRefitter(const edm::ParameterSet& iConfig):
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<std::string>( "src" ));
  //register your products
  produces<TrackingRecHitCollection>();
  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
}

void TrackRefitter::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("TrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  getFromES(setup,theG,theMF,theFitter,thePropagator);

  //
  //declare and get TrackCollection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  try {
    edm::Handle<reco::TrackCollection> theTCollection;
    getFromEvt(theEvent,theTCollection);
    
    //
    //run the algorithm  
    //
    edm::LogInfo("TrackProducer") << "run the algorithm" << "\n";
    theAlgo.runWithTrack(theG.product(), theMF.product(), *theTCollection, 
			 theFitter.product(), thePropagator.product(), algoResults);
  } catch (cms::Exception &e){}
  //
  //put everything in th event
  putInEvt(theEvent, outputRHColl, outputTColl, outputTEColl, algoResults);
  edm::LogInfo("TrackProducer") << "end" << "\n";
}

