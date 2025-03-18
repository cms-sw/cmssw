#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/CAGeometrySoA.h"
#include "RecoTracker/Record/interface/CAGeometryHost.h"
#include "RecoTracker/Record/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class CAGeometryESProducer : public ESProducer {
  public:
    CAGeometryESProducer(edm::ParameterSet const& iConfig);
    std::optional<reco::CAGeometryHost> produce(const TrackerRecoGeometryRecord& iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

  private:
    // Layers params
    const std::vector<float> caThetaCuts_;
    const std::vector<double> caDCACuts_;

    // Cells params
    // TODO: move to unsigned int here
    const std::vector<int> pairGraph_;
    const std::vector<int> startingPairs_;
    const std::vector<int> phiCuts_;
    const std::vector<double> minZ_;
    const std::vector<double> maxZ_;
    const std::vector<double> maxR_;

    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopologyToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeometryToken_;
  };

  CAGeometryESProducer::CAGeometryESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig),
        caThetaCuts_(iConfig.getParameter<std::vector<float>>("caThetaCuts")),
        caDCACuts_(iConfig.getParameter<std::vector<double>>("caDCACuts")),
        pairGraph_(iConfig.getParameter<std::vector<int>>("pairGraph")),
        startingPairs_(iConfig.getParameter<std::vector<int>>("startingPairs")),
        phiCuts_(iConfig.getParameter<std::vector<int>>("phiCuts")),
        minZ_(iConfig.getParameter<std::vector<double>>("minZ")),
        maxZ_(iConfig.getParameter<std::vector<double>>("maxZ")),
        maxR_(iConfig.getParameter<std::vector<double>>("maxR")) {
    auto cc = setWhatProduced(this);
    tTopologyToken_ = cc.consumes();
    tGeometryToken_ = cc.consumes();
  }

  std::optional<reco::CAGeometryHost> CAGeometryESProducer::produce(const TrackerRecoGeometryRecord& iRecord) {
    assert(minZ_.size() == maxZ_.size());
    assert(minZ_.size() == maxR_.size());
    assert(minZ_.size() == phiCuts_.size());

    assert(caThetaCuts_.size() == caDCACuts_.size());

    int n_layers = caThetaCuts_.size();
    int n_pairs = pairGraph_.size() / 2;
    int n_modules = 0;

#ifdef GPU_DEBUG
    std::cout << "No. Layers to be used = " << n_layers << std::endl;
    std::cout << "No. Pairs to be used = " << n_pairs << std::endl;
#endif

    assert(int(n_pairs) == int(minZ_.size()));
    assert(*std::max_element(startingPairs_.begin(), startingPairs_.end()) <= n_pairs);
    assert(*std::max_element(pairGraph_.begin(), pairGraph_.end()) < n_layers);

    const auto& trackerTopology = &iRecord.get(tTopologyToken_);
    const auto& trackerGeometry = &iRecord.get(tGeometryToken_);
    auto const& dets = trackerGeometry->dets();

#ifdef GPU_DEBUG
    auto subSystem = 1;
    auto subSystemName = GeomDetEnumerators::tkDetEnum[subSystem];
    auto subSystemOffset = trackerGeometry->offsetDU(subSystemName);
    std::cout
        << "========================================================================================================="
        << std::endl;
    std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
    subSystemName = GeomDetEnumerators::tkDetEnum[++subSystem];
    subSystemOffset = trackerGeometry->offsetDU(subSystemName);
#endif

    auto oldLayer = 0u;
    auto layerCount = 0;

    std::vector<int> layerStarts(n_layers + 1);

    for (auto& det : dets) {
      DetId detid = det->geographicalId();
#ifdef GPU_DEBUG
      if (n_modules >= int(subSystemOffset)) {
        subSystemName = GeomDetEnumerators::tkDetEnum[++subSystem];
        subSystemOffset = trackerGeometry->offsetDU(subSystemName);
        std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
      }
#endif

      auto layer = trackerTopology->layer(detid);

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
      layerSoA.caThetaCut()[i] = caThetaCuts_[i];
      layerSoA.caDCACut()[i] = caDCACuts_[i];
    }

    layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

    for (int i = 0; i < n_pairs; ++i) {
      cellSoA.graph()[i] = {{uint32_t(pairGraph_[2 * i]), uint32_t(pairGraph_[2 * i + 1])}};
      cellSoA.phiCuts()[i] = phiCuts_[i];
      cellSoA.minz()[i] = minZ_[i];
      cellSoA.maxz()[i] = maxZ_[i];
      cellSoA.maxr()[i] = maxR_[i];
      cellSoA.startingPair()[i] = false;
    }

    for (const int& i : startingPairs_)
      cellSoA.startingPair()[i] = true;

    return product;
  }

  void CAGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    using namespace phase1PixelTopology;
    constexpr auto nPairs = pixelTopology::Phase1::nPairsForQuadruplets;
    edm::ParameterSetDescription desc;

    // layers params
    desc.add<std::vector<double>>("caDCACuts", {0.15f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f})
        ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
    desc.add<std::vector<float>>("caThetaCuts",
                                  {0.002f, 0.002f, 0.002f, 0.002f, 0.003f, 0.003f, 0.003f, 0.003f, 0.003f, 0.003f})
        ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
    desc.add<std::vector<int>>("startingPairs", {0, 1, 2})
        ->setComment(
            "The list of the ids of pairs from which the CA ntuplets building may start.");  //TODO could be parsed via an expression
    // cells params
    desc.add<std::vector<int>>("pairGraph",
                               std::vector<int>(std::begin(layerPairs), std::begin(layerPairs) + (nPairs * 2)))
        ->setComment("CA graph");
    desc.add<std::vector<int>>("phiCuts", std::vector<int>(std::begin(phicuts), std::begin(phicuts) + nPairs))
        ->setComment("Cuts in phi for cells");
    desc.add<std::vector<double>>("minZ", std::vector<double>(std::begin(minz), std::begin(minz) + nPairs))
        ->setComment("Cuts in min z (on inner hit) for cells");
    desc.add<std::vector<double>>("maxZ", std::vector<double>(std::begin(maxz), std::begin(maxz) + nPairs))
        ->setComment("Cuts in max z (on inner hit) for cells");
    desc.add<std::vector<double>>("maxR", std::vector<double>(std::begin(maxr), std::begin(maxr) + nPairs))
        ->setComment("Cuts in max r for cells");

    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(CAGeometryESProducer);
