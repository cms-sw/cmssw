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
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

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
          maxR_(iConfig.getParameter<std::vector<double>>("maxR")) {}

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
    assert(int(*std::max_element(iCache->startingPairs_.begin(), iCache->startingPairs_.end())) <= n_pairs);
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
    auto layerCount = 0;

    std::vector<int> layerStarts(n_layers + 1);
    std::vector<int> moduleToindexInDets;

    auto isPinPSinOTBarrel = [&](DetId detId) {
      //    std::cout << (int)trackerGeometry->getDetectorType(detId) << " " << (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP) << "\n";
      //    std::cout << (int)detId.subdetId() << " " << (detId.subdetId() == StripSubdetector::TOB) << std::endl;
      // Select only P-hits from the OT barrel
      return (trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
      detId.subdetId() == StripSubdetector::TOB);
    };
    auto isPh2Pixel = [&](DetId detId) {
      return (trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PXB
      || trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PXF
      || trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PXF
      || trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PXF3D);
    };
    if constexpr (std::is_base_of_v<pixelTopology::Phase2, TrackerTraits>) {
      int counter = 0;
      for (auto& det : dets) {
        DetId detid = det->geographicalId();
        auto layer = trackerTopology.layer(detid);
        // Logic:
        // - if we are not inside pixels, we need to ignore anything **but** the OT.
        // - for the time being, this is assuming that the CA extension will
        //   only cover the OT barrel part, and will ignore the OT forward.
        if (isPh2Pixel(detid)) {
          if (layer != oldLayer) {
            layerStarts[layerCount++] = n_modules;
            if (layerCount > n_layers + 1)
              break;
            oldLayer = layer;
          }
          moduleToindexInDets.push_back(counter);
          n_modules++;
        } else {
          auto const & detUnits = det->components();
          for (auto& detUnit : detUnits)
          {
            DetId unitDetId(detUnit->geographicalId());
            if (isPinPSinOTBarrel(unitDetId)) {
              if (layer != oldLayer) {
                layerStarts[layerCount++] = n_modules;
                if (layerCount > n_layers + 1)
                  break;
                oldLayer = layer;
              }
              moduleToindexInDets.push_back(counter);
              n_modules++;
            }
          }
        }
        counter++;
      }
    } else {
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

          if (layerCount > n_layers + 1)
            break;

          oldLayer = layer;
#ifdef GPU_DEBUG
          std::cout << " > New layer at module : " << n_modules << " (detId: " << detid << ")" << std::endl;
#endif
        }

        n_modules++;
      }
    }

    reco::CAGeometryHost product{{{n_layers + 1, n_pairs, n_modules}}, cms::alpakatools::host()};

    auto layerSoA = product.view();
    auto cellSoA = product.view<::reco::CAGraphSoA>();
    auto modulesSoA = product.view<::reco::CAModulesSoA>();

    if constexpr (std::is_base_of_v<pixelTopology::Phase2, TrackerTraits>) {
      for (int i = 0; i < n_modules; ++i) {
        auto idx = moduleToindexInDets[i];
        auto det = dets[idx];
        auto vv = det->surface().position();
        auto rr = Rotation(det->surface().rotation());
        modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
      }

      for (int i = 0; i < n_layers; ++i) {
        layerSoA.layerStarts()[i] = layerStarts[i];
        layerSoA.caThetaCut()[i] = iCache->caThetaCuts_[i];
        layerSoA.caDCACut()[i] = iCache->caDCACuts_[i];
      }
    } else {
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

    for (const int& i : iCache->startingPairs_)
    cellSoA.startingPair()[i] = true;

    return std::make_shared<CAGeometryCache>(std::move(product));
  }

  static std::unique_ptr<::reco::CAGeoemtryParams> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::make_unique<::reco::CAGeoemtryParams>(iConfig.getParameterSet("geometry"));
  }

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
  const device::EDGetToken<HitsOnDevice> tokenHit_;
  const device::EDPutToken<TkSoADevice> tokenTrack_;

  const TFormula maxNumberOfDoublets_;
  const TFormula maxNumberOfTuples_;
  Algo deviceAlgo_;
};

template <typename TrackerTraits>
CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig,
                                                      const ::reco::CAGeoemtryParams* iCache)
  : EDProducer(iConfig),
  tokenField_(esConsumes()),
  tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
  tokenTrack_(produces()),
  maxNumberOfDoublets_(
    TFormula("doubletsHitsDependecy", iConfig.getParameter<std::string>("maxNumberOfDoublets").data())),
  maxNumberOfTuples_(
    TFormula("tracksHitsDependency", iConfig.getParameter<std::string>("maxNumberOfTuples").data())),
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

  uint32_t const maxTuples = maxNumberOfTuples_.Eval(hits.nHits());
  uint32_t const maxDoublets = maxNumberOfDoublets_.Eval(hits.nHits());

  iEvent.emplace(tokenTrack_,
                 deviceAlgo_.makeTuplesAsync(hits, geometry, bf, maxDoublets, maxTuples, iEvent.queue()));
}

using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
