#include <alpaka/alpaka.hpp>

#include <TFormula.h>
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "CAHitNtupletGenerator.h"

#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

// #define GPU_DEBUG

namespace reco {
  struct CAGeometryParams {
    //Constructor from ParameterSet
    CAGeometryParams(edm::ParameterSet const& iConfig)
        : caThetaCuts_(iConfig.getParameter<std::vector<double>>("caThetaCuts")),
          caDCACuts_(iConfig.getParameter<std::vector<double>>("caDCACuts")),
          pairGraph_(iConfig.getParameter<std::vector<unsigned int>>("pairGraph")),
          startingPairs_(iConfig.getParameter<std::vector<unsigned int>>("startingPairs")),
          phiCuts_(iConfig.getParameter<std::vector<int>>("phiCuts")),
          minZ_(iConfig.getParameter<std::vector<double>>("minZ")),
          maxZ_(iConfig.getParameter<std::vector<double>>("maxZ")),
          maxR_(iConfig.getParameter<std::vector<double>>("maxR")) {
          startNoBPix1_ = false;
          for (const unsigned int& i : startingPairs_) {
            if (pairGraph_[2 * i] > 0) {
              startNoBPix1_ = true;
              break;
            }
          }
        }

    // Layers params
    const std::vector<double> caThetaCuts_;
    const std::vector<double> caDCACuts_;

    // Cells params
    const std::vector<unsigned int> pairGraph_;
    const std::vector<unsigned int> startingPairs_;
    const std::vector<int> phiCuts_;
    const std::vector<double> minZ_;
    const std::vector<double> maxZ_;
    const std::vector<double> maxR_;

    bool startNoBPix1_;

    mutable edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenGeometry_;
    mutable edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tokenTopology_;
  };

}  // namespace reco

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletAlpaka
      : public stream::EDProducer<edm::GlobalCache<::reco::CAGeometryParams>,
                                  edm::RunCache<cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>>> {
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;

    using TkSoAHost = ::reco::TracksHost;
    using TkSoADevice = reco::TracksSoACollection;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

    using CAGeometryCache = cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>;
    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig, const ::reco::CAGeometryParams* iCache);
    ~CAHitNtupletAlpaka() override = default;

    void produce(device::Event& iEvent, const device::EventSetup& es) override;

    static void globalEndJob(::reco::CAGeometryParams const*) { /* Do nothing */ };
    static void globalEndRun(edm::Run const& iRun,
                             edm::EventSetup const&,
                             RunContext const* iContext) { /* Do nothing */ };

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static std::shared_ptr<CAGeometryCache> globalBeginRun(edm::Run const& iRun,
                                                           edm::EventSetup const& iSetup,
                                                           GlobalCache const* iCache) {
      assert(iCache->minZ_.size() == iCache->maxZ_.size());
      assert(iCache->minZ_.size() == iCache->maxR_.size());
      assert(iCache->minZ_.size() == iCache->phiCuts_.size());

      assert(iCache->caThetaCuts_.size() == iCache->caDCACuts_.size());

      int n_layers = iCache->caThetaCuts_.size();
      int n_pairs = iCache->pairGraph_.size() / 2;
      int n_modules = 0;

#ifdef GPU_DEBUG
      std::cout << "No. Layers to be used = " << n_layers << std::endl;
      std::cout << "No. Pairs to be used = " << n_pairs << std::endl;
#endif

      assert(int(n_pairs) == int(iCache->minZ_.size()));
      assert(int(*std::max_element(iCache->startingPairs_.begin(), iCache->startingPairs_.end())) < n_pairs);
      assert(int(*std::max_element(iCache->pairGraph_.begin(), iCache->pairGraph_.end())) < n_layers);

      const auto& trackerGeometry = iSetup.getData(iCache->tokenGeometry_);
      const auto& trackerTopology = iSetup.getData(iCache->tokenTopology_);
      auto const& dets = trackerGeometry.dets();

#ifdef GPU_DEBUG
      auto subSystem = 1;
      auto subSystemName = GeomDetEnumerators::tkDetEnum[subSystem];
      auto subSystemOffset = trackerGeometry.offsetDU(subSystemName);
      std::cout
          << "========================================================================================================="
          << std::endl;
      std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
      subSystemName = GeomDetEnumerators::tkDetEnum[++subSystem];
      subSystemOffset = trackerGeometry.offsetDU(subSystemName);
#endif

      auto oldLayer = 0u;
      auto layerCount = 0u;

      std::vector<int> layerStarts(n_layers + 1);
      //^ why n_layers + 1? This is a cumulative sum of the number
      // of modules each layer has. And we need the  extra spot
      // at the end to hold the total number of modules.

      for (auto& det : dets) {
        DetId detid = det->geographicalId();
#ifdef GPU_DEBUG
        if (n_modules >= int(subSystemOffset)) {
          subSystemName = GeomDetEnumerators::tkDetEnum[++subSystem];
          subSystemOffset = trackerGeometry.offsetDU(subSystemName);
          std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
        }
#endif

        auto layer = trackerTopology.layer(detid);

        if (layer != oldLayer) {
          layerStarts[layerCount++] = n_modules;

          if (layerCount >= layerStarts.size())
            break;

          oldLayer = layer;
#ifdef GPU_DEBUG
          std::cout << " > New layer at module : " << n_modules << " (detId: " << detid << ")" << std::endl;
#endif
        }

        n_modules++;
      }

      reco::CAGeometryHost product{{{n_layers + 1, n_pairs, n_modules}}, cms::alpakatools::host()};

      auto layerSoA = product.view();
      auto cellSoA = product.view<::reco::CAGraphSoA>();
      auto modulesSoA = product.view<::reco::CAModulesSoA>();

      for (int i = 0; i < n_modules; ++i) {
        auto det = dets[i];
        auto vv = det->surface().position();
        auto rr = Rotation(det->surface().rotation());
        modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
      }

      for (int i = 0; i < n_layers; ++i) {
        layerSoA.layerStarts()[i] = layerStarts[i];
        layerSoA.caThetaCut()[i] = iCache->caThetaCuts_[i];
        layerSoA.caDCACut()[i] = iCache->caDCACuts_[i];
      }

      layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

      for (int i = 0; i < n_pairs; ++i) {
        cellSoA.graph()[i] = {{uint32_t(iCache->pairGraph_[2 * i]), uint32_t(iCache->pairGraph_[2 * i + 1])}};
        cellSoA.phiCuts()[i] = iCache->phiCuts_[i];
        cellSoA.minz()[i] = iCache->minZ_[i];
        cellSoA.maxz()[i] = iCache->maxZ_[i];
        cellSoA.maxr()[i] = iCache->maxR_[i];
        cellSoA.startingPair()[i] = false;
      }

      for (const unsigned int& i : iCache->startingPairs_)
        cellSoA.startingPair()[i] = true;

      return std::make_shared<CAGeometryCache>(std::move(product));
    }

    static std::unique_ptr<::reco::CAGeometryParams> initializeGlobalCache(edm::ParameterSet const& iConfig) {
      return std::make_unique<::reco::CAGeometryParams>(iConfig.getParameterSet("geometry"));
    }

  private:
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;

    const ::reco::FormulaEvaluator maxNumberOfDoublets_;
    const ::reco::FormulaEvaluator maxNumberOfTuples_;

    Algo deviceAlgo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig,
                                                        const ::reco::CAGeometryParams* iCache)
      : EDProducer(iConfig),
        tokenField_(esConsumes()),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
        maxNumberOfDoublets_(iConfig.getParameter<std::string>("maxNumberOfDoublets")),
        maxNumberOfTuples_(iConfig.getParameter<std::string>("maxNumberOfTuples")),
        deviceAlgo_(iConfig) {
    iCache->tokenGeometry_ = esConsumes<edm::Transition::BeginRun>();
    iCache->tokenTopology_ = esConsumes<edm::Transition::BeginRun>();
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto const& geometry = runCache()->get(iEvent.queue());
    auto const& hits = iEvent.get(tokenHit_);

    /// Don't bother if no hits on BPix1 and no good graph for that
    /// (so no staring pair without BPix1 as first layer).
    /// TODO: this could be extended to a more general check for
    /// no hits on any of the starting layers.

    if (globalCache()->startNoBPix1_ or hits.offsetBPIX2() > 0) {
      std::array<double, 1> nHitsV = {{double(hits.nHits())}};
      std::array<double, 1> emptyV;

      uint32_t const maxTuples = maxNumberOfTuples_.evaluate(nHitsV, emptyV);
      uint32_t const maxDoublets = maxNumberOfDoublets_.evaluate(nHitsV, emptyV);

      iEvent.emplace(tokenTrack_,
                     deviceAlgo_.makeTuplesAsync(hits, geometry, bf, maxDoublets, maxTuples, iEvent.queue()));

    } else {
      edm::LogWarning("CAHitNtupletAlpaka") << "No hit on BPix1 (" << hits.offsetBPIX2()
                                            << ") and all the starting pairs has BPix1 as inner layer.\nIt's useless "
                                            << "to run the CA. Returning with 0 tracks!";
      auto& queue = iEvent.queue();
      reco::TracksSoACollection tracks({{0, 0}}, queue);
      auto ntracks_d = cms::alpakatools::make_device_view(queue, tracks.view().nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      iEvent.emplace(tokenTrack_, std::move(tracks));
    }
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
