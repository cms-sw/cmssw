#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

/*
  produces seeds directly from cuda produced tuples
*/
class SeedProducerFromSoA : public edm::global::EDProducer<> {
public:
  explicit SeedProducerFromSoA(const edm::ParameterSet& iConfig);
  ~SeedProducerFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenTrack_;

  int32_t minNumberOfHits_;
};

SeedProducerFromSoA::SeedProducerFromSoA(const edm::ParameterSet& iConfig)
    : tBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      tokenTrack_(consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("src"))),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits"))

{
  produces<TrajectorySeedCollection>();
}

void SeedProducerFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTrackSoA"));
  desc.add<int>("minNumberOfHits", 0);

  descriptions.addWithDefaultLabel(desc);
}

void SeedProducerFromSoA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // std::cout << "Converting gpu helix to trajectory seed" << std::endl;
  auto result = std::make_unique<TrajectorySeedCollection>();

  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  auto const& dus = tracker->detUnits();

  edm::ESHandle<Propagator> propagatorHandle;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorHandle);
  const Propagator* propagator = &(*propagatorHandle);

  edm::ESHandle<TrackerTopology> httopo;
  iSetup.get<TrackerTopologyRcd>().get(httopo);

  const auto& bsh = iEvent.get(tBeamSpot_);
  // std::cout << "beamspot " << bsh.x0() << ' ' << bsh.y0() << ' ' << bsh.z0() << std::endl;
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  const auto& tsoa = *(iEvent.get(tokenTrack_));

  auto const* quality = tsoa.qualityData();
  auto const& fit = tsoa.stateAtBS;
  auto const& detIndices = tsoa.detIndices;
  auto maxTracks = tsoa.stride();

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...

    auto q = quality[it];
    if (q != trackQuality::loose)
      continue;  // FIXME
    if (nHits < minNumberOfHits_)
      continue;
    ++nt;

    // fill hits with invalid just to hold the detId
    auto b = detIndices.begin(it);
    edm::OwnVector<TrackingRecHit> hits;
    for (int iHit = 0; iHit < nHits; ++iHit) {
      auto const* det = dus[*(b + iHit)];
      // FIXME at some point get a proper type ...
      hits.push_back(new InvalidTrackingRecHit(*det, TrackingRecHit::bad));
    }

    // mind: this values are respect the beamspot!

    float phi = tsoa.phi(it);

    Rfit::Vector5d ipar, opar;
    Rfit::Matrix5d icov, ocov;
    fit.copyToDense(ipar, icov, it);
    Rfit::transformToPerigeePlane(ipar, icov, opar, ocov);

    LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        m(i, j) = ocov(i, j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

    Plane impPointPlane(bs, rot);
    GlobalTrajectoryParameters gp(impPointPlane.toGlobal(lpar.position()),
                                  impPointPlane.toGlobal(lpar.momentum()),
                                  lpar.charge(),
                                  fieldESH.product());

    JacobianLocalToCurvilinear jl2c(impPointPlane, lpar, *fieldESH.product());

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(), m);

    FreeTrajectoryState fts(gp, CurvilinearTrajectoryError(mo));

    auto const& lastHit = hits.back();

    TrajectoryStateOnSurface outerState = propagator->propagate(fts, *lastHit.surface());

    if (!outerState.isValid()) {
      edm::LogError("SeedFromGPU") << " was trying to create a seed from:\n"
                                   << fts << "\n propagating to: " << lastHit.geographicalId().rawId();
      continue;
    }

    auto const& pTraj = trajectoryStateTransform::persistentState(outerState, lastHit.geographicalId().rawId());

    result->emplace_back(pTraj, hits, alongMomentum);
  }

  iEvent.put(std::move(result));
}

DEFINE_FWK_MODULE(SeedProducerFromSoA);
