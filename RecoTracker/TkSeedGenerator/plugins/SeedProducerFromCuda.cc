#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"

/*
  produces seeds directly from cuda produced tuples
*/
class SeedProducerFromCuda
    : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<heterogeneous::GPUCuda, heterogeneous::CPU>> {
public:
  using Input = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;
  using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;

  explicit SeedProducerFromCuda(const edm::ParameterSet &iConfig);
  ~SeedProducerFromCuda() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<> &cudaStream) override {}
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) override;

private:
  TuplesOnCPU const *tuples_ = nullptr;

  edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;
  edm::EDGetTokenT<HeterogeneousProduct> gpuToken_;
  uint32_t minNumberOfHits_;
};

SeedProducerFromCuda::SeedProducerFromCuda(const edm::ParameterSet &iConfig)
    : HeterogeneousEDProducer(iConfig),
      tBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      gpuToken_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      minNumberOfHits_(iConfig.getParameter<unsigned int>("minNumberOfHits"))
{
    produces<TrajectorySeedCollection>();
}

void SeedProducerFromCuda::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksHitQuadruplets"));
  desc.add<unsigned int>("minNumberOfHits",4);
  HeterogeneousEDProducer::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void SeedProducerFromCuda::acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                                                const edm::EventSetup &iSetup,
                                                cuda::stream_t<> &cudaStream) {
  edm::Handle<TuplesOnCPU> gh;
  iEvent.getByToken<Input>(gpuToken_, gh);
  //auto const & gTuples = *gh;
  // std::cout << "tuples from gpu " << gTuples.nTuples << std::endl;

  tuples_ = gh.product();
}

void SeedProducerFromCuda::produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                                                const edm::EventSetup &iSetup,
                                                cuda::stream_t<> &cudaStream) {

  // std::cout << "Converting gpu helix to trajectory seed" << std::endl;
  auto result = std::make_unique<TrajectorySeedCollection>();


  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  auto const & dus = tracker->detUnits();

  edm::ESHandle<Propagator>  propagatorHandle;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",propagatorHandle);
  const Propagator*  propagator = &(*propagatorHandle);

  edm::ESHandle<TrackerTopology> httopo;
  iSetup.get<TrackerTopologyRcd>().get(httopo);


  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(tBeamSpot_, bsHandle);
  const auto &bsh = *bsHandle;
  // std::cout << "beamspot " << bsh.x0() << ' ' << bsh.y0() << ' ' << bsh.z0() << std::endl;
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  auto const & detIndices =  *tuples_->detIndices;
  for (uint32_t it = 0; it < tuples_->nTuples; ++it) {
    auto q = tuples_->quality[it];
    if (q != pixelTuplesHeterogeneousProduct::loose)
      continue;                           // FIXME
    auto nHits = detIndices.size(it);
    if (nHits<minNumberOfHits_) continue;

    // fill hits with invalid just to hold the detId
    auto b = detIndices.begin(it);
    edm::OwnVector<TrackingRecHit> hits;
    for (unsigned int iHit = 0; iHit < nHits; ++iHit) {
      auto const * det =  dus[*(b+iHit)];
      // FIXME at some point get a proper type ...
      hits.push_back(new InvalidTrackingRecHit(*det,TrackingRecHit::bad));
    }

    // mind: this values are respect to the beamspot!
    auto const &fittedTrack = tuples_->helix_fit_results[it];

    // std::cout << "tk " << it << ": " << fittedTrack.q << ' ' << fittedTrack.par[2] << ' ' << std::sqrt(fittedTrack.cov(2, 2)) << std::endl;

    // "Reference implementation" following CMSSW (ORCA!) practices
    // to be optimized, refactorized and eventually moved to GPU

    auto iCharge = fittedTrack.q;
    float phi = fittedTrack.par(0);
   
    Rfit::Vector5d opar;
    Rfit::Matrix5d ocov;
    Rfit::transformToPerigeePlane(fittedTrack.par,fittedTrack.cov,opar,ocov,iCharge);

    LocalTrajectoryParameters lpar(opar(0),opar(1),opar(2),opar(3),opar(4),1.);
    AlgebraicSymMatrix55 m;
    for(int i=0; i<5; ++i) for (int j=i; j<5; ++j) m(i,j) = ocov(i,j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(
                              sp, -cp,    0,
                               0,   0, -1.f,
                              cp,  sp,    0);

    Plane impPointPlane(bs,rot);
    GlobalTrajectoryParameters gp(impPointPlane.toGlobal(lpar.position()), 
                                  impPointPlane.toGlobal(lpar.momentum()),lpar.charge(),fieldESH.product());

    JacobianLocalToCurvilinear jl2c(impPointPlane,lpar,*fieldESH.product());

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(),m);

    FreeTrajectoryState fts(gp, CurvilinearTrajectoryError(mo));

    auto const & lastHit = hits.back();

    TrajectoryStateOnSurface outerState = propagator->propagate(fts, *lastHit.surface());

    if (!outerState.isValid()){
      edm::LogError("SeedFromGPU")<<" was trying to create a seed from:\n"<<fts<<"\n propagating to: " 
                                         << lastHit.geographicalId().rawId();
      continue;
    }

    auto const & pTraj = trajectoryStateTransform::persistentState(outerState, lastHit.geographicalId().rawId());

    result->emplace_back(pTraj, hits, alongMomentum);

  }

  iEvent.put(std::move(result));
}

void SeedProducerFromCuda::produceCPU(edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) {
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

DEFINE_FWK_MODULE(SeedProducerFromCuda);
