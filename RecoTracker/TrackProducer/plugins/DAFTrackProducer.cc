/** \class DAFTrackProducer
  *  EDProducer for DAFTrackProducerAlgorithm.
  *
  *  \author tropiano, genta
  *  \review in May 2014 by brondolin 
  */

#include <memory>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/TrackProducer/interface/DAFTrackProducerAlgorithm.h"
#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class DAFTrackProducer : public KfTrackProducerBase, public edm::stream::EDProducer<> {
public:
  typedef std::vector<Trajectory> TrajectoryCollection;
  //  typedef std::vector<TrajAnnealing> TrajAnnealingCollection;
  explicit DAFTrackProducer(const edm::ParameterSet& iConfig);

  // Implementation of produce method
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// fillDescriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  DAFTrackProducerAlgorithm theAlgo;
  using TrackProducerBase<reco::Track>::getFromEvt;
  void getFromEvt(edm::Event&, edm::Handle<TrajTrackAssociationCollection>&, reco::BeamSpot&);
  void putInEvtTrajAnn(edm::Event& theEvent,
                       TrajAnnealingCollection& trajannResults,
                       std::unique_ptr<TrajAnnealingCollection>& selTrajAnn);

  bool TrajAnnSaving_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> srcTT_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  edm::ESGetToken<MultiRecHitCollector, MultiRecHitRecord> measurementCollectorToken_;
  edm::ESGetToken<SiTrackerMultiRecHitUpdator, MultiRecHitRecord> updatorToken_;
};

DAFTrackProducer::DAFTrackProducer(const edm::ParameterSet& iConfig)
    : KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"), false), theAlgo(iConfig) {
  initTrackProducerBase(
      iConfig, consumesCollector(), consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>("src")));
  srcTT_ = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("src"));
  setAlias(iConfig.getParameter<std::string>("@module_label"));

  //register your products
  produces<reco::TrackCollection>().setBranchAlias(alias_ + "Tracks");
  produces<reco::TrackExtraCollection>().setBranchAlias(alias_ + "TrackExtras");
  produces<TrackingRecHitCollection>().setBranchAlias(alias_ + "RecHits");
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
  produces<TrajAnnealingCollection>().setBranchAlias(alias_ + "TrajectoryAnnealing");
  produces<reco::TrackCollection>("beforeDAF").setBranchAlias(alias_ + "TracksBeforeDAF");
  produces<reco::TrackExtraCollection>("beforeDAF").setBranchAlias(alias_ + "TrackExtrasBeforeDAF");
  produces<reco::TrackCollection>("afterDAF").setBranchAlias(alias_ + "TracksAfterDAF");
  produces<reco::TrackExtraCollection>("afterDAF").setBranchAlias(alias_ + "TrackExtrasAfterDAF");

  TrajAnnSaving_ = iConfig.getParameter<bool>("TrajAnnealingSaving");
  ttopoToken_ = esConsumes();
  std::string measurementCollectorName = getConf().getParameter<std::string>("MeasurementCollector");
  measurementCollectorToken_ = esConsumes(edm::ESInputTag("", measurementCollectorName));
  std::string updatorName = getConf().getParameter<std::string>("UpdatorName");
  updatorToken_ = esConsumes(edm::ESInputTag("", updatorName));
}

void DAFTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("TrajectoryInEvent", false);
  desc.add<edm::InputTag>("src", edm::InputTag("DAFTrackCandidateMaker"));
  desc.add<bool>("TrajAnnealingSaving", false);
  desc.add<std::string>("MeasurementCollector", "simpleMultiRecHitCollector");
  desc.add<std::string>("UpdatorName", "SiTrackerMultiRecHitUpdator");
  KfTrackProducerBase::fillPSetDescription(desc);
  DAFTrackProducerAlgorithm::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void DAFTrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup) {
  edm::LogInfo("DAFTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";

  //empty output collections
  std::unique_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::unique_ptr<std::vector<Trajectory> > outputTrajectoryColl(new std::vector<Trajectory>);
  std::unique_ptr<TrajAnnealingCollection> outputTrajAnnColl(new TrajAnnealingCollection);
  std::unique_ptr<std::vector<int> > outputIndecesInputColl(new std::vector<int>);

  //new tracks collections (changes before and after DAF)
  std::unique_ptr<TrackingRecHitCollection> outputRHCollBeforeDAF(new TrackingRecHitCollection);
  std::unique_ptr<reco::TrackCollection> outputTCollBeforeDAF(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> outputTECollBeforeDAF(new reco::TrackExtraCollection);
  std::unique_ptr<std::vector<Trajectory> > outputTrajectoryCollBeforeDAF(new std::vector<Trajectory>);
  std::unique_ptr<std::vector<int> > outputIndecesInputCollBeforeDAF(new std::vector<int>);
  //----
  std::unique_ptr<TrackingRecHitCollection> outputRHCollAfterDAF(new TrackingRecHitCollection);
  std::unique_ptr<reco::TrackCollection> outputTCollAfterDAF(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> outputTECollAfterDAF(new reco::TrackExtraCollection);
  std::unique_ptr<std::vector<Trajectory> > outputTrajectoryCollAfterDAF(new std::vector<Trajectory>);
  std::unique_ptr<std::vector<int> > outputIndecesInputCollAfterDAF(new std::vector<int>);

  //declare and get stuff to be retrieved from ES
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup, theG, theMF, theFitter, thePropagator, theMeasTk, theBuilder);

  edm::ESHandle<TrackerTopology> httopo = setup.getHandle(ttopoToken_);

  //get additional es_modules needed by the DAF
  edm::ESHandle<MultiRecHitCollector> measurementCollectorHandle = setup.getHandle(measurementCollectorToken_);
  edm::ESHandle<SiTrackerMultiRecHitUpdator> updatorHandle = setup.getHandle(updatorToken_);

  //get MeasurementTrackerEvent
  edm::Handle<MeasurementTrackerEvent> mte;
  theEvent.getByToken(mteSrc_, mte);

  //declare and get TrackCollection
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  TrajAnnealingCollection trajannResults;

  //declare and get  new tracks collections
  AlgoProductCollection algoResultsBeforeDAF;
  AlgoProductCollection algoResultsAfterDAF;
  try {
    edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
    getFromEvt(theEvent, trajTrackAssociationHandle, bs);

    //run the algorithm
    LogDebug("DAFTrackProducer") << "run the DAF algorithm"
                                 << "\n";
    theAlgo.runWithCandidate(theG.product(),
                             theMF.product(),
                             *trajTrackAssociationHandle,
                             &*mte,
                             theFitter.product(),
                             theBuilder.product(),
                             measurementCollectorHandle.product(),
                             updatorHandle.product(),
                             bs,
                             algoResults,
                             trajannResults,
                             TrajAnnSaving_,
                             algoResultsBeforeDAF,
                             algoResultsAfterDAF);

  } catch (cms::Exception& e) {
    edm::LogInfo("DAFTrackProducer") << "cms::Exception caught!!!"
                                     << "\n"
                                     << e << "\n";
    throw;
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
           httopo.product());
  putInEvtTrajAnn(theEvent, trajannResults, outputTrajAnnColl);

  //put in theEvent before and after DAF tracks collections
  putInEvt(theEvent,
           thePropagator.product(),
           theMeasTk.product(),
           outputRHCollBeforeDAF,
           outputTCollBeforeDAF,
           outputTECollBeforeDAF,
           outputTrajectoryCollBeforeDAF,
           outputIndecesInputCollBeforeDAF,
           algoResultsBeforeDAF,
           theBuilder.product(),
           httopo.product(),
           1);
  putInEvt(theEvent,
           thePropagator.product(),
           theMeasTk.product(),
           outputRHCollAfterDAF,
           outputTCollAfterDAF,
           outputTECollAfterDAF,
           outputTrajectoryCollAfterDAF,
           outputIndecesInputCollAfterDAF,
           algoResultsAfterDAF,
           theBuilder.product(),
           httopo.product(),
           2);

  LogDebug("DAFTrackProducer") << "end the DAF algorithm."
                               << "\n";
}
//----------------------------------------------------------------------------------------------------------//
void DAFTrackProducer::getFromEvt(edm::Event& theEvent,
                                  edm::Handle<TrajTrackAssociationCollection>& trajTrackAssociationHandle,
                                  reco::BeamSpot& bs) {
  //get the TrajTrackMap from the event
  //WARNING: src has always to be redefined in cfg file
  theEvent.getByToken(srcTT_, trajTrackAssociationHandle);

  //get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  theEvent.getByToken(bsSrc_, recoBeamSpotHandle);
  bs = *recoBeamSpotHandle;
}
//----------------------------------------------------------------------------------------------------------//
void DAFTrackProducer::putInEvtTrajAnn(edm::Event& theEvent,
                                       TrajAnnealingCollection& trajannResults,
                                       std::unique_ptr<TrajAnnealingCollection>& outputTrajAnnColl) {
  const int size = trajannResults.size();
  outputTrajAnnColl->reserve(size);

  for (unsigned int i = 0; i < trajannResults.size(); i++) {
    outputTrajAnnColl->push_back(trajannResults[i]);
  }

  theEvent.put(std::move(outputTrajAnnColl));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DAFTrackProducer);
