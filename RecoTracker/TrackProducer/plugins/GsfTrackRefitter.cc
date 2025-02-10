/** \class GsfTrackRefitter
 *  Refit GSF Tracks. Based on the TrackRefitter.
 */

// system include files
#include <memory>

// user include files
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "TrackingTools/GsfTracking/interface/GsfTrackConstraintAssociation.h"
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class GsfTrackRefitter : public GsfTrackProducerBase, public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit GsfTrackRefitter(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// fillDescriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  TrackProducerAlgorithm<reco::GsfTrack> theAlgo;
  enum Constraint {
    none,
    // 		    momentum,
    vertex
  };
  Constraint constraint_;
  edm::EDGetTokenT<GsfTrackVtxConstraintAssociationCollection> gsfTrackVtxConstraintTag_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
};

GsfTrackRefitter::GsfTrackRefitter(const edm::ParameterSet& iConfig)
    : GsfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
                           iConfig.getParameter<bool>("useHitsSplitting")),
      theAlgo(iConfig) {
  initTrackProducerBase(
      iConfig, consumesCollector(), consumes<edm::View<reco::GsfTrack>>(iConfig.getParameter<edm::InputTag>("src")));
  setAlias(iConfig.getParameter<std::string>("@module_label"));
  std::string constraint_str = iConfig.getParameter<std::string>("constraint");

  if (constraint_str.empty())
    constraint_ = none;
  //   else if (constraint_str == "momentum") constraint_ = momentum;
  else if (constraint_str == "vertex") {
    constraint_ = vertex;
    gsfTrackVtxConstraintTag_ = consumes<GsfTrackVtxConstraintAssociationCollection>(
        iConfig.getParameter<edm::InputTag>("gsfTrackVtxConstraintTag"));
  } else {
    edm::LogError("GsfTrackRefitter") << "constraint: " << constraint_str
                                      << " not understood. Set it to 'momentum', 'vertex' or leave it empty";
    throw cms::Exception("GsfTrackRefitter")
        << "unknown type of contraint! Set it to 'momentum', 'vertex' or leave it empty";
  }

  //register your products
  produces<reco::GsfTrackCollection>().setBranchAlias(alias_ + "GsfTracks");
  produces<reco::TrackExtraCollection>().setBranchAlias(alias_ + "TrackExtras");
  produces<reco::GsfTrackExtraCollection>().setBranchAlias(alias_ + "GsfTrackExtras");
  produces<TrackingRecHitCollection>().setBranchAlias(alias_ + "RecHits");
  produces<std::vector<Trajectory>>();
  produces<TrajGsfTrackAssociationCollection>();

  ttopoToken_ = esConsumes();
}

void GsfTrackRefitter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("TrajectoryInEvent", false);
  desc.add<bool>("useHitsSplitting", false);
  desc.add<edm::InputTag>("src", edm::InputTag(""));
  desc.add<std::string>("constraint", "");
  desc.add<edm::InputTag>("srcConstr", edm::InputTag(""));
  TrackProducerAlgorithm<reco::GsfTrack>::fillPSetDescription(desc);
  GsfTrackProducerBase::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void GsfTrackRefitter::produce(edm::Event& theEvent, const edm::EventSetup& setup) {
  edm::LogInfo("GsfTrackRefitter") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::unique_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::unique_ptr<reco::GsfTrackCollection> outputTColl(new reco::GsfTrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::unique_ptr<reco::GsfTrackExtraCollection> outputGsfTEColl(new reco::GsfTrackExtraCollection);
  std::unique_ptr<std::vector<Trajectory>> outputTrajectoryColl(new std::vector<Trajectory>);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  //  getFromES(setup,theG,theMF,theFitter,thePropagator);
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup, theG, theMF, theFitter, thePropagator, theMeasTk, theBuilder);

  edm::ESHandle<TrackerTopology> httopo = setup.getHandle(ttopoToken_);

  //
  //declare and get TrackCollection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  switch (constraint_) {
    case none: {
      edm::Handle<edm::View<reco::GsfTrack>> theTCollection;
      getFromEvt(theEvent, theTCollection, bs);
      if (theTCollection.failedToGet()) {
        edm::LogError("GsfTrackRefitter") << "could not get the reco::GsfTrackCollection.";
        return;
      }
      LogDebug("GsfTrackRefitter") << "run the algorithm"
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
      } catch (cms::Exception& e) {
        edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack."
                                       << "\n"
                                       << e << "\n";
        throw;
      }
      break;
    }
    case vertex: {
      edm::Handle<GsfTrackVtxConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(gsfTrackVtxConstraintTag_, theTCollectionWithConstraint);
      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_, recoBeamSpotHandle);
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
      } catch (cms::Exception& e) {
        edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack."
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
           outputGsfTEColl,
           outputTrajectoryColl,
           algoResults,
           theBuilder.product(),
           bs,
           httopo.product());
  LogDebug("GsfTrackRefitter") << "end"
                               << "\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfTrackRefitter);
