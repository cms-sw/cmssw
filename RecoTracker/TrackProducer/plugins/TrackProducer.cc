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

TrackProducer::TrackProducer(const edm::ParameterSet& iConfig)
    : KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
                          iConfig.getParameter<bool>("useHitsSplitting")),
      theAlgo(iConfig),
      theTTopoToken(esConsumes()) {
  initTrackProducerBase(
      iConfig, consumesCollector(), consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>("src")));
  setAlias(iConfig.getParameter<std::string>("@module_label"));

  if (iConfig.exists("clusterRemovalInfo")) {
    edm::InputTag tag = iConfig.getParameter<edm::InputTag>("clusterRemovalInfo");
    if (!(tag == edm::InputTag())) {
      setClusterRemovalInfo(tag);
    }
  }

  //register your products
  produces<reco::TrackExtraCollection>().setBranchAlias(alias_ + "TrackExtras");
  produces<TrackingRecHitCollection>().setBranchAlias(alias_ + "RecHits");
  // TrackCollection refers to TrackingRechit and TrackExtra
  // collections, need to declare its production after them to work
  // around a rare race condition in framework scheduling
  produces<reco::TrackCollection>().setBranchAlias(alias_ + "Tracks");
  produces<std::vector<Trajectory> >();
  produces<std::vector<int> >();
  produces<TrajTrackAssociationCollection>();
}

void TrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup) {
  LogDebug("TrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::unique_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::unique_ptr<std::vector<Trajectory> > outputTrajectoryColl(new std::vector<Trajectory>);
  std::unique_ptr<std::vector<int> > outputIndecesInputColl(new std::vector<int>);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup, theG, theMF, theFitter, thePropagator, theMeasTk, theBuilder);

  TrackerTopology const& ttopo = setup.getData(theTTopoToken);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  edm::Handle<TrackCandidateCollection> theTCCollection;
  reco::BeamSpot bs;
  getFromEvt(theEvent, theTCCollection, bs);
  //protect against missing product
  if (theTCCollection.failedToGet()) {
    edm::LogError("TrackProducer") << "could not get the TrackCandidateCollection.";
  } else {
    LogDebug("TrackProducer") << "run the algorithm"
                              << "\n";
    try {
      theAlgo.runWithCandidate(theG.product(),
                               theMF.product(),
                               *theTCCollection,
                               theFitter.product(),
                               thePropagator.product(),
                               theBuilder.product(),
                               bs,
                               algoResults);
    } catch (cms::Exception& e) {
      edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithCandidate."
                                     << "\n"
                                     << e << "\n";
      throw;
    }
  }

  //put everything in the event
  putInEvt(theEvent,
           thePropagator.product(),
           theMeasTk.product(),
           outputRHColl,
           outputTColl,
           outputTEColl,
           outputTrajectoryColl,
           outputIndecesInputColl,
           algoResults,
           theBuilder.product(),
           &ttopo);
  LogDebug("TrackProducer") << "end"
                            << "\n";
}

std::vector<reco::TransientTrack> TrackProducer::getTransient(edm::Event& theEvent, const edm::EventSetup& setup) {
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
  edm::ESHandle<MeasurementTracker> theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup, theG, theMF, theFitter, thePropagator, theMeasTk, theBuilder);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  edm::Handle<TrackCandidateCollection> theTCCollection;
  reco::BeamSpot bs;
  getFromEvt(theEvent, theTCCollection, bs);
  //protect against missing product
  if (theTCCollection.failedToGet()) {
    edm::LogError("TrackProducer") << "could not get the TrackCandidateCollection.";
  } else {
    LogDebug("TrackProducer") << "run the algorithm"
                              << "\n";
    try {
      theAlgo.runWithCandidate(theG.product(),
                               theMF.product(),
                               *theTCCollection,
                               theFitter.product(),
                               thePropagator.product(),
                               theBuilder.product(),
                               bs,
                               algoResults);
    } catch (cms::Exception& e) {
      edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithCandidate."
                                     << "\n"
                                     << e << "\n";
      throw;
    }
  }
  ttks.reserve(algoResults.size());
  for (auto& prod : algoResults) {
    ttks.push_back(reco::TransientTrack(*(prod.track), thePropagator.product()->magneticField()));
  }

  LogDebug("TrackProducer") << "end"
                            << "\n";

  return ttks;
}
