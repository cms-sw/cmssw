/** \class TrackRefitter
 *  Refit Tracks: Produce Tracks from TrackCollection. It performs a new final fit on a TrackCollection.
 *
 *  \author cerati
 */

// system include files
#include <memory>

// user include files
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class TrackRefitter : public KfTrackProducerBase, public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit TrackRefitter(const edm::ParameterSet &iConfig);

  /// Implementation of produce method
  void produce(edm::Event &, const edm::EventSetup &) override;

  /// fillDescriptions
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  TrackProducerAlgorithm<reco::Track> theAlgo;
  enum Constraint { none, momentum, vertex, trackParameters };
  Constraint constraint_;
  edm::EDGetToken trkconstrcoll_;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
};

TrackRefitter::TrackRefitter(const edm::ParameterSet &iConfig)
    : KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
                          iConfig.getParameter<bool>("useHitsSplitting")),
      theAlgo(iConfig),
      ttopoToken_(esConsumes()) {
  initTrackProducerBase(
      iConfig, consumesCollector(), consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("src")));
  setAlias(iConfig.getParameter<std::string>("@module_label"));
  std::string constraint_str = iConfig.getParameter<std::string>("constraint");
  edm::InputTag trkconstrcoll = iConfig.getParameter<edm::InputTag>("srcConstr");

  if (constraint_str.empty())
    constraint_ = none;
  else if (constraint_str == "momentum") {
    constraint_ = momentum;
    trkconstrcoll_ = consumes<TrackMomConstraintAssociationCollection>(trkconstrcoll);
  } else if (constraint_str == "vertex") {
    constraint_ = vertex;
    trkconstrcoll_ = consumes<TrackVtxConstraintAssociationCollection>(trkconstrcoll);
  } else if (constraint_str == "trackParameters") {
    constraint_ = trackParameters;
    trkconstrcoll_ = consumes<TrackParamConstraintAssociationCollection>(trkconstrcoll);
  } else {
    edm::LogError("TrackRefitter")
        << "constraint: " << constraint_str
        << " not understood. Set it to 'momentum', 'vertex', 'trackParameters' or leave it empty";
    throw cms::Exception("TrackRefitter")
        << "unknown type of contraint! Set it to 'momentum', 'vertex', 'trackParameters' or leave it empty";
  }

  //register your products
  produces<reco::TrackCollection>().setBranchAlias(alias_ + "Tracks");
  produces<reco::TrackExtraCollection>().setBranchAlias(alias_ + "TrackExtras");
  produces<TrackingRecHitCollection>().setBranchAlias(alias_ + "RecHits");
  produces<std::vector<Trajectory>>();
  produces<std::vector<int>>();
  produces<TrajTrackAssociationCollection>();
}

void TrackRefitter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("TrajectoryInEvent", false);
  desc.add<bool>("useHitsSplitting", false);
  desc.add<edm::InputTag>("src", edm::InputTag(""));
  desc.add<std::string>("constraint", "");
  desc.add<edm::InputTag>("srcConstr", edm::InputTag(""));
  TrackProducerAlgorithm<reco::Track>::fillPSetDescription(desc);
  KfTrackProducerBase::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void TrackRefitter::produce(edm::Event &theEvent, const edm::EventSetup &setup) {
  LogDebug("TrackRefitter") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::unique_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::unique_ptr<std::vector<Trajectory>> outputTrajectoryColl(new std::vector<Trajectory>);
  std::unique_ptr<std::vector<int>> outputIndecesInputColl(new std::vector<int>);

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

  TrackerTopology const &ttopo = setup.getData(ttopoToken_);

  //
  //declare and get TrackCollection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  switch (constraint_) {
    case none: {
      edm::Handle<edm::View<reco::Track>> theTCollection;
      getFromEvt(theEvent, theTCollection, bs);

      LogDebug("TrackRefitter") << "TrackRefitter::produce(none):Number of Trajectories:" << (*theTCollection).size();

      if (bs.position() == math::XYZPoint(0., 0., 0.) && bs.type() == reco::BeamSpot::Unknown) {
        edm::LogError("TrackRefitter") << " BeamSpot is (0,0,0), it is probably because is not valid in the event";
        break;
      }

      if (theTCollection.failedToGet()) {
        edm::EDConsumerBase::Labels labels;
        labelsForToken(src_, labels);
        edm::LogError("TrackRefitter") << "could not get the reco::TrackCollection." << labels.module;
        break;
      }
      LogDebug("TrackRefitter") << "run the algorithm"
                                << "\n";

      try {
        theAlgo.runWithTrack(theG.product(),
                             theMF.product(),
                             *theTCollection,
                             theFitter.product(),
                             thePropagator.product(),
                             theBuilder.product(),
                             bs,
                             algoResults);
      } catch (cms::Exception &e) {
        edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack."
                                       << "\n"
                                       << e << "\n";
        throw;
      }
      break;
    }
    case momentum: {
      edm::Handle<TrackMomConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(trkconstrcoll_, theTCollectionWithConstraint);

      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_, recoBeamSpotHandle);
      if (!recoBeamSpotHandle.isValid())
        break;
      bs = *recoBeamSpotHandle;
      if (theTCollectionWithConstraint.failedToGet()) {
        //edm::LogError("TrackRefitter")<<"could not get TrackMomConstraintAssociationCollection product.";
        break;
      }
      LogDebug("TrackRefitter") << "run the algorithm"
                                << "\n";
      try {
        theAlgo.runWithMomentum(theG.product(),
                                theMF.product(),
                                *theTCollectionWithConstraint,
                                theFitter.product(),
                                thePropagator.product(),
                                theBuilder.product(),
                                bs,
                                algoResults);
      } catch (cms::Exception &e) {
        edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithMomentum."
                                       << "\n"
                                       << e << "\n";
        throw;
      }
      break;
    }
    case vertex: {
      edm::Handle<TrackVtxConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(trkconstrcoll_, theTCollectionWithConstraint);
      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_, recoBeamSpotHandle);
      if (!recoBeamSpotHandle.isValid())
        break;
      bs = *recoBeamSpotHandle;
      if (theTCollectionWithConstraint.failedToGet()) {
        edm::LogError("TrackRefitter") << "could not get TrackVtxConstraintAssociationCollection product.";
        break;
      }
      LogDebug("TrackRefitter") << "run the algorithm"
                                << "\n";
      try {
        theAlgo.runWithVertex(theG.product(),
                              theMF.product(),
                              *theTCollectionWithConstraint,
                              theFitter.product(),
                              thePropagator.product(),
                              theBuilder.product(),
                              bs,
                              algoResults);
      } catch (cms::Exception &e) {
        edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithVertex."
                                       << "\n"
                                       << e << "\n";
        throw;
      }
      break;
    }
    case trackParameters: {
      edm::Handle<TrackParamConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(trkconstrcoll_, theTCollectionWithConstraint);
      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_, recoBeamSpotHandle);
      if (!recoBeamSpotHandle.isValid())
        break;
      bs = *recoBeamSpotHandle;
      if (theTCollectionWithConstraint.failedToGet()) {
        //edm::LogError("TrackRefitter")<<"could not get TrackParamConstraintAssociationCollection product.";
        break;
      }
      LogDebug("TrackRefitter") << "run the algorithm"
                                << "\n";
      try {
        theAlgo.runWithTrackParameters(theG.product(),
                                       theMF.product(),
                                       *theTCollectionWithConstraint,
                                       theFitter.product(),
                                       thePropagator.product(),
                                       theBuilder.product(),
                                       bs,
                                       algoResults);
      } catch (cms::Exception &e) {
        edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrackParameters."
                                       << "\n"
                                       << e << "\n";
        throw;
      }
    }
      //default... there cannot be any other possibility due to the check in the ctor
  }

  //put everything in th event
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
  LogDebug("TrackRefitter") << "end"
                            << "\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackRefitter);
