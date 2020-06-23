#include "SeedGeneratorFromTTracksEDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

// extra
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit5DParamConstraint.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace edm;
using namespace reco;

SeedGeneratorFromTTracksEDProducer::SeedGeneratorFromTTracksEDProducer(const ParameterSet& cfg)
    : theConfig(cfg),
      theInputCollectionTag(
          consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(cfg.getParameter<InputTag>("InputCollection"))),
      theEstimatorName(cfg.getParameter<std::string>("estimator")),
      thePropagatorName(cfg.getParameter<std::string>("propagator")),
      theMeasurementTrackerTag(
          consumes<MeasurementTrackerEvent>(cfg.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
      theMinEtaForTEC(cfg.getParameter<double>("minEtaForTEC")),
      theMaxEtaForTOB(cfg.getParameter<double>("maxEtaForTOB")),
      errorSFHitless(cfg.getParameter<double>("errorSFHitless")) {
  produces<TrajectorySeedCollection>();
}

void SeedGeneratorFromTTracksEDProducer::findSeedsOnLayer(const GeometricSearchDet& layer,
                                                          const TrajectoryStateOnSurface& tsosAtIP,
                                                          const Propagator& propagatorAlong,
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
      std::cout << "ERROR!: Hitless TSOS is not valid! \n";
    } else {
      dets.front().second.rescaleError(errorSFHitless);

      PTrajectoryStateOnDet const& ptsod =
          trajectoryStateTransform::persistentState(tsosOnLayer, detOnLayer->geographicalId().rawId());
      TrajectorySeed::RecHitContainer rHC;
      if (numSeedsMade < 1) {  // only outermost seed
        out->push_back(TrajectorySeed(ptsod, rHC, oppositeToMomentum));
        numSeedsMade++;
      }
    }
  }
}

void SeedGeneratorFromTTracksEDProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  std::unique_ptr<std::vector<TrajectorySeed>> result(new std::vector<TrajectorySeed>());

  // TTrack Collection
  Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> trks;
  ev.getByToken(theInputCollectionTag, trks);

  // Trk Geometry
  edm::ESHandle<TrackerGeometry> tmpTkGeometryH;
  es.get<TrackerDigiGeometryRecord>().get(tmpTkGeometryH);

  // Mag field
  edm::ESHandle<MagneticField> magfieldH;
  es.get<IdealMagneticFieldRecord>().get(magfieldH);

  // Estimator
  edm::ESHandle<Chi2MeasurementEstimatorBase> estimatorH;
  es.get<TrackingComponentsRecord>().get(theEstimatorName, estimatorH);

  // Get Propagators
  edm::ESHandle<Propagator> propagatorAlongH;
  es.get<TrackingComponentsRecord>().get(thePropagatorName, propagatorAlongH);
  std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(*propagatorAlongH, alongMomentum);

  edm::ESHandle<Propagator> propagatorOppositeH;
  es.get<TrackingComponentsRecord>().get(thePropagatorName, propagatorOppositeH);
  std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(*propagatorOppositeH, oppositeToMomentum);

  // Get vector of Detector layers
  edm::Handle<MeasurementTrackerEvent> measurementTrackerH;
  ev.getByToken(theMeasurementTrackerTag, measurementTrackerH);
  std::vector<BarrelDetLayer const*> const& tob = measurementTrackerH->geometricSearchTracker()->tobLayers();
  std::vector<ForwardDetLayer const*> const& tecPositive =
      tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC)
          ? measurementTrackerH->geometricSearchTracker()->posTidLayers()
          : measurementTrackerH->geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer const*> const& tecNegative =
      tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC)
          ? measurementTrackerH->geometricSearchTracker()->negTidLayers()
          : measurementTrackerH->geometricSearchTracker()->negTecLayers();

  /// Surface used to make a TSOS at the PCA to the beamline
  Plane::PlanePointer dummyPlane = Plane::build(Plane::PositionType(), Plane::RotationType());

  // Loop over the L1's and make seeds for all of them:
  std::vector<TTTrack<Ref_Phase2TrackerDigi_>>::const_iterator it;
  for (it = trks->begin(); it != trks->end(); it++) {
    const TTTrack<Ref_Phase2TrackerDigi_>& l1 = (*it);

    std::unique_ptr<std::vector<TrajectorySeed>> out(new std::vector<TrajectorySeed>());
    FreeTrajectoryState fts = trajectoryStateTransform::initialFreeStateTTrack(l1, magfieldH.product(), false);
    dummyPlane->move(fts.position() - dummyPlane->position());
    TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);

    unsigned int numSeedsMade = 0;
    //BARREL
    if (std::abs(l1.momentum().eta()) < theMaxEtaForTOB) {
      for (auto it = tob.rbegin(); it != tob.rend(); ++it) {  //This goes from outermost to innermost layer
        findSeedsOnLayer(**it, tsosAtIP, *(propagatorAlong.get()), l1, estimatorH, numSeedsMade, out);
      }
    }
    if (std::abs(l1.momentum().eta()) > theMinEtaForTEC && std::abs(l1.momentum().eta()) < theMaxEtaForTOB) {
      numSeedsMade = 0;  // reset num of seeds
    }
    //ENDCAP+
    if (l1.momentum().eta() > theMinEtaForTEC) {
      for (auto it = tecPositive.rbegin(); it != tecPositive.rend(); ++it) {
        findSeedsOnLayer(**it, tsosAtIP, *(propagatorAlong.get()), l1, estimatorH, numSeedsMade, out);
      }
    }
    //ENDCAP-
    if (l1.momentum().eta() < -theMinEtaForTEC) {
      for (auto it = tecNegative.rbegin(); it != tecNegative.rend(); ++it) {
        findSeedsOnLayer(**it, tsosAtIP, *(propagatorAlong.get()), l1, estimatorH, numSeedsMade, out);
      }
    }
    for (std::vector<TrajectorySeed>::iterator it = out->begin(); it != out->end(); ++it) {
      result->push_back(*it);
    }
  }  // end loop over L1Tracks

  auto const& seeds = *result;
  ev.put(std::move(result));
}
