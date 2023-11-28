#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
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
#include "RecoTracker/PixelTrackFitting/interface/FitUtils.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

/*
  produces seeds directly from cuda produced tuples
*/
template <typename TrackerTraits>
class SeedProducerFromSoAT : public edm::global::EDProducer<> {
public:
  explicit SeedProducerFromSoAT(const edm::ParameterSet& iConfig);
  ~SeedProducerFromSoAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  // Event data tokens
  const edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;
  const edm::EDGetTokenT<TrackSoAHeterogeneousHost<TrackerTraits>> tokenTrack_;
  // Event setup tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerDigiGeometryToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> trackerPropagatorToken_;
  int32_t minNumberOfHits_;
};

template <typename TrackerTraits>
SeedProducerFromSoAT<TrackerTraits>::SeedProducerFromSoAT(const edm::ParameterSet& iConfig)
    : tBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      tokenTrack_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      idealMagneticFieldToken_(esConsumes()),
      trackerDigiGeometryToken_(esConsumes()),
      trackerPropagatorToken_(esConsumes(edm::ESInputTag("PropagatorWithMaterial"))),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits"))

{
  produces<TrajectorySeedCollection>();
}

template <typename TrackerTraits>
void SeedProducerFromSoAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTrackSoA"));
  desc.add<int>("minNumberOfHits", 0);

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SeedProducerFromSoAT<TrackerTraits>::produce(edm::StreamID streamID,
                                                  edm::Event& iEvent,
                                                  const edm::EventSetup& iSetup) const {
  // std::cout << "Converting gpu helix to trajectory seed" << std::endl;
  auto result = std::make_unique<TrajectorySeedCollection>();

  using trackHelper = TracksUtilities<TrackerTraits>;

  auto const& fieldESH = iSetup.getHandle(idealMagneticFieldToken_);
  auto const& tracker = iSetup.getHandle(trackerDigiGeometryToken_);
  auto const& dus = tracker->detUnits();

  auto const& propagatorHandle = iSetup.getHandle(trackerPropagatorToken_);
  const Propagator* propagator = &(*propagatorHandle);

  const auto& bsh = iEvent.get(tBeamSpot_);
  // std::cout << "beamspot " << bsh.x0() << ' ' << bsh.y0() << ' ' << bsh.z0() << std::endl;
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  auto const& tsoa = iEvent.get(tokenTrack_);

  auto const* quality = tsoa.view().quality();
  auto const& detIndices = tsoa.view().detIndices();
  auto maxTracks = tsoa.view().metadata().size();

  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = trackHelper::nHits(tsoa.view(), it);
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...

    auto q = quality[it];
    if (q != pixelTrack::Quality::loose)
      continue;  // FIXME
    if (nHits < minNumberOfHits_)
      continue;

    // fill hits with invalid just to hold the detId
    auto b = detIndices.begin(it);
    edm::OwnVector<TrackingRecHit> hits;
    for (int iHit = 0; iHit < nHits; ++iHit) {
      auto const* det = dus[*(b + iHit)];
      // FIXME at some point get a proper type ...
      hits.push_back(new InvalidTrackingRecHit(*det, TrackingRecHit::bad));
    }

    // mind: this values are respect the beamspot!

    float phi = trackHelper::nHits(tsoa.view(), it);

    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    trackHelper::copyToDense(tsoa.view(), ipar, icov, it);
    riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);

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

using SeedProducerFromSoAPhase1 = SeedProducerFromSoAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SeedProducerFromSoAPhase1);

using SeedProducerFromSoAPhase2 = SeedProducerFromSoAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SeedProducerFromSoAPhase2);
