#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

class dso_hidden SeedGeneratorFromL1TTracksEDProducer : public edm::stream::EDProducer<> {
 public:
  SeedGeneratorFromL1TTracksEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromL1TTracksEDProducer() override = default;  

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void findSeedsOnLayer(const GeometricSearchDet& layer,
                        const TrajectoryStateOnSurface& tsosAtIP,
                        Propagator& propagatorAlong,
                        const TTTrack<Ref_Phase2TrackerDigi_>& l1,
                        edm::ESHandle<Chi2MeasurementEstimatorBase>& estimatorH,
                        unsigned int& numSeedsMade,
                        std::unique_ptr<std::vector<TrajectorySeed> >& out) const;

  void produce(edm::Event& ev, const edm::EventSetup& es) override;

 private:
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > theInputCollectionTag_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerTag_;

  // Minimum eta value to activate searching in the TEC
  const double theMinEtaForTEC_;

  // Maximum eta value to activate searching in the TOB
  const double theMaxEtaForTOB_;

  const double theErrorSFHitless_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> estToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;

};

SeedGeneratorFromL1TTracksEDProducer::SeedGeneratorFromL1TTracksEDProducer(const edm::ParameterSet& cfg)
   :  theInputCollectionTag_(
			     consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(cfg.getParameter<edm::InputTag>("InputCollection"))),
      theMeasurementTrackerTag_(
          consumes<MeasurementTrackerEvent>(cfg.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
      theMinEtaForTEC_(cfg.getParameter<double>("minEtaForTEC")),
      theMaxEtaForTOB_(cfg.getParameter<double>("maxEtaForTOB")),
      theErrorSFHitless_(cfg.getParameter<double>("errorSFHitless")),
      mfToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      geomToken_{esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()},
      estToken_{
	esConsumes<Chi2MeasurementEstimatorBase, TrackingComponentsRecord>(edm::ESInputTag("", cfg.getParameter<std::string>("estimator")))}, 
      propagatorAlongToken_{
	esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))},
      propagatorOppositeToken_{
	esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", cfg.getParameter<std::string>("propagator")))}
{
  produces<TrajectorySeedCollection>();
}

void SeedGeneratorFromL1TTracksEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("TTTracksFromTrackletEmulation","Level1TTTracks"));
  desc.add<std::string>("estimator", std::string(""));
  desc.add<std::string>("propagator", std::string(""));
  desc.add<edm::InputTag>("MeasurementTrackerEvent", edm::InputTag(""));
  desc.add<double>("minEtaForTEC", 0.9);
  desc.add<double>("maxEtaForTOB", 1.2);
  desc.add<double>("errorSFHitless", 1e-9);
  descriptions.add("SeedGeneratorFromL1TTracksEDProducer", desc);
}

void SeedGeneratorFromL1TTracksEDProducer::findSeedsOnLayer(const GeometricSearchDet& layer,
                                                          const TrajectoryStateOnSurface& tsosAtIP,
                                                          Propagator& propagatorAlong,
                                                          const TTTrack<Ref_Phase2TrackerDigi_>& l1,
							  edm::ESHandle<Chi2MeasurementEstimatorBase>& estimatorH,
                                                          unsigned int& numSeedsMade,
                                                          std::unique_ptr<std::vector<TrajectorySeed>>& out) const {
  std::vector<GeometricSearchDet::DetWithState> dets;
  layer.compatibleDetsV(tsosAtIP, propagatorAlong, *estimatorH, dets);

  if (!dets.empty()) {
    auto const& detOnLayer = dets.front().first;
    auto const& tsosOnLayer = dets.front().second;
    if (!tsosOnLayer.isValid()) {
      LogDebug("SeedGeneratorFromL1TTracks") << "Hitless TSOS is not valid!";
    } else {
      dets.front().second.rescaleError(theErrorSFHitless_);

      PTrajectoryStateOnDet const& ptsod =
          trajectoryStateTransform::persistentState(tsosOnLayer, detOnLayer->geographicalId().rawId());
      TrajectorySeed::RecHitContainer rHC;
      if (numSeedsMade < 1) {  // only outermost seed
	out->emplace_back(TrajectorySeed(ptsod, rHC, oppositeToMomentum));
        numSeedsMade++;
      }
    }
  }
}

void SeedGeneratorFromL1TTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  std::unique_ptr<std::vector<TrajectorySeed>> result(new std::vector<TrajectorySeed>());

  // TTrack Collection
  auto const& trks = ev.get(theInputCollectionTag_);

  // Trk Geometry
  const auto& geom = es.getData(geomToken_);

  // Mag field
  const auto& mag = es.getData(mfToken_);

  // Estimator
  //auto& estimator = es.getData(estToken_);
  auto estimatorH = es.getHandle(estToken_);

  // Get Propagators
  const auto& propagatorAlongH = es.getData(propagatorAlongToken_);
  std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(propagatorAlongH, alongMomentum);

  const auto& propagatorOppositeH = es.getData(propagatorOppositeToken_);
  std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(propagatorOppositeH, oppositeToMomentum);

  // Get vector of Detector layers
  auto const& measurementTracker = ev.get(theMeasurementTrackerTag_);
  std::vector<BarrelDetLayer const*> const& tob = measurementTracker.geometricSearchTracker()->tobLayers();
  
  std::vector<ForwardDetLayer const*> const& tecPositive =
    geom.isThere(GeomDetEnumerators::P2OTEC)
    ? measurementTracker.geometricSearchTracker()->posTidLayers()
    : measurementTracker.geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer const*> const& tecNegative =
    geom.isThere(GeomDetEnumerators::P2OTEC)
    ? measurementTracker.geometricSearchTracker()->negTidLayers()
    : measurementTracker.geometricSearchTracker()->negTecLayers();
  
  /// Surface used to make a TSOS at the PCA to the beamline
  Plane::PlanePointer dummyPlane = Plane::build(Plane::PositionType(), Plane::RotationType());

  // Loop over the L1's and make seeds for all of them:
  for (auto const& l1 : trks) {
    std::unique_ptr<std::vector<TrajectorySeed>> out(new std::vector<TrajectorySeed>());
    FreeTrajectoryState fts = trajectoryStateTransform::initialFreeStateTTrack(l1, &mag, true);
    dummyPlane->move(fts.position() - dummyPlane->position());
    TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);

    unsigned int numSeedsMade = 0;
    //BARREL
    if (std::abs(l1.momentum().eta()) < theMaxEtaForTOB_) {
      for (auto it = tob.rbegin(); it != tob.rend(); ++it) {  //This goes from outermost to innermost layer
        findSeedsOnLayer(**it, tsosAtIP, *(propagatorAlong.get()), l1, estimatorH, numSeedsMade, out);
      }
    }
    if (std::abs(l1.momentum().eta()) > theMinEtaForTEC_) {
      numSeedsMade = 0;  // reset num of seeds
    }
    //ENDCAP+
    if (l1.momentum().eta() > theMinEtaForTEC_) {
      for (auto it = tecPositive.rbegin(); it != tecPositive.rend(); ++it) {
        findSeedsOnLayer(**it, tsosAtIP, *(propagatorAlong.get()), l1, estimatorH, numSeedsMade, out);
      }
    }
    //ENDCAP-
    if (l1.momentum().eta() < -theMinEtaForTEC_) {
      for (auto it = tecNegative.rbegin(); it != tecNegative.rend(); ++it) {
        findSeedsOnLayer(**it, tsosAtIP, *(propagatorAlong.get()), l1, estimatorH, numSeedsMade, out);
      }
    }
    for (std::vector<TrajectorySeed>::iterator it = out->begin(); it != out->end(); ++it) {
      result->push_back(*it);
    }
    //std::copy(out->begin(), out->end(), std::back_inserter(result));
  }  // end loop over L1Tracks

  ev.put(std::move(result));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SeedGeneratorFromL1TTracksEDProducer);

