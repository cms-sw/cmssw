// OTStubProducerVectorHitStyle - VectorHit-style stub formation for the Phase-2 Outer Tracker.
// Define OTSTUB_DEBUG_VERBOSE (or CXXFLAGS="-DOTSTUB_DEBUG_VERBOSE") for verbose GPU->CPU debug.

#include <vector>

#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationVectorHitStyleKernelsWrapper.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class OTStubProducerVectorHitStyle : public stream::SynchronizingEDProducer<> {
  public:
    explicit OTStubProducerVectorHitStyle(edm::ParameterSet const& iConfig);
    ~OTStubProducerVectorHitStyle() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    device::EDGetToken<reco::OTRecHitsSoACollection> hitToken_;

    device::ESGetToken<reco::StackedModuleGeometrySoACollection, StackedModuleGeometryRecord> geomToken_;

    device::EDPutToken<reco::StubsSoACollection> stubToken_;

    std::vector<double> maxWidthBarrelFlat_;
    std::vector<double> maxWidthBarrelTilted_;
    std::vector<double> maxWidthEndcap_;
    std::vector<int32_t> barrelFlatMaxCSDiff_;
    std::vector<int32_t> barrelTiltedMaxCSDiff_;
    std::vector<int32_t> endcapMaxCSDiff_;
    std::vector<int32_t> barrelFlatMaxCS_;
    std::vector<int32_t> barrelTiltedMaxCS_;
    std::vector<int32_t> endcapMaxCS_;
    std::vector<int32_t> barrelFlatMaxCSSum_;
    std::vector<int32_t> barrelTiltedMaxCSSum_;
    std::vector<int32_t> endcapMaxCSSum_;

    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> stubOffsets_d_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> maxWidthBarrelFlat_d_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> maxWidthBarrelTilted_d_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> maxWidthEndcap_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelFlatMaxCSDiff_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelTiltedMaxCSDiff_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> endcapMaxCSDiff_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelFlatMaxCS_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelTiltedMaxCS_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> endcapMaxCS_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelFlatMaxCSSum_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelTiltedMaxCSSum_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> endcapMaxCSSum_d_;
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> nStubs_h_;
    // The cut tables are job constants: uploaded once per stream, then only read.
    bool cutsUploaded_ = false;
    uint32_t nModules_ = 0;
  };

  OTStubProducerVectorHitStyle::OTStubProducerVectorHitStyle(edm::ParameterSet const& iConfig)
      : SynchronizingEDProducer<>(iConfig),
        hitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))),
        geomToken_(esConsumes()),
        stubToken_(produces()),
        maxWidthBarrelFlat_(iConfig.getParameter<std::vector<double>>("maxWidthBarrelFlat")),
        maxWidthBarrelTilted_(iConfig.getParameter<std::vector<double>>("maxWidthBarrelTilted")),
        maxWidthEndcap_(iConfig.getParameter<std::vector<double>>("maxWidthEndcap")),
        barrelFlatMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeDiffBarrelFlat")),
        barrelTiltedMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeDiffBarrelTilted")),
        endcapMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeDiffEndcap")),
        barrelFlatMaxCS_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeBarrelFlat")),
        barrelTiltedMaxCS_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeBarrelTilted")),
        endcapMaxCS_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeEndcap")),
        barrelFlatMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeBarrelFlatSum")),
        barrelTiltedMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeSumBarrelTilted")),
        endcapMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeSumEndcap")) {}

  void OTStubProducerVectorHitStyle::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("otRecHitsSoAConverter"))
        ->setComment("Input OT RecHits SoA collection");
    desc.add<std::vector<double>>("maxWidthBarrelFlat", {0.0, 0.05, 0.06, 0.08, 0.09, 0.12, 0.2})
        ->setComment("Layer-dependent width cuts for flat barrel modules (cm), indexed by layer 0-6");
    desc.add<std::vector<double>>("maxWidthBarrelTilted", {0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15})
        ->setComment(
            "Layer-dependent width cuts for tilted barrel modules (cm), indexed by layer 0-6. Only layers 1-3 have "
            "tilted modules.");
    desc.add<std::vector<double>>("maxWidthEndcap", {0.0, 0.1, 0.1, 0.1, 0.1, 0.1})
        ->setComment("Layer-dependent width cuts for endcap (cm), indexed by layer 0-5");
    desc.add<std::vector<int32_t>>("maxClusterSizeDiffBarrelFlat", {999, 999, 999, 999, 999, 999, 999})
        ->setComment(
            "Per-layer max |clusterSize_lower - clusterSize_upper| for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeDiffBarrelTilted", {999, 999, 999, 999, 999, 999, 999})
        ->setComment(
            "Per-layer max |clusterSize_lower - clusterSize_upper| for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeDiffEndcap", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for endcap (layers 0-5). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeBarrelFlat", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeBarrelTilted", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeEndcap", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for endcap (layers 0-5). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeBarrelFlatSum", {999, 999, 999, 999, 999, 999, 999})
        ->setComment(
            "Per-layer max (clusterSize_lower + clusterSize_upper) for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeSumBarrelTilted", {999, 999, 999, 999, 999, 999, 999})
        ->setComment(
            "Per-layer max (clusterSize_lower + clusterSize_upper) for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeSumEndcap", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for endcap (layers 0-5). 999 = disabled.");
    descriptions.addWithDefaultLabel(desc);
  }

  void OTStubProducerVectorHitStyle::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();

    auto const& otRecHits = iEvent.get(hitToken_);

    auto const& geomDevice = iSetup.getData(geomToken_);
    auto const& geomView = geomDevice.const_view();
    nModules_ = geomView.metadata().size();

    auto const& hitsView = otRecHits.const_view().otRecHits();
    auto const& moduleView = otRecHits.const_view().otHitModules();

    if (!cutsUploaded_) {
      std::vector<float> maxWidthBarrelFlatFloat(maxWidthBarrelFlat_.begin(), maxWidthBarrelFlat_.end());
      std::vector<float> maxWidthBarrelTiltedFloat(maxWidthBarrelTilted_.begin(), maxWidthBarrelTilted_.end());
      std::vector<float> maxWidthEndcapFloat(maxWidthEndcap_.begin(), maxWidthEndcap_.end());

#ifdef OTSTUB_DEBUG_VERBOSE
      edm::LogPrint("OTStubProducerVectorHitStyle") << "Barrel flat cuts:";
      for (size_t i = 0; i < maxWidthBarrelFlatFloat.size(); ++i) {
        edm::LogPrint("OTStubProducerVectorHitStyle")
            << "  maxWidthBarrelFlat[" << i << "] = " << maxWidthBarrelFlatFloat[i];
      }
      edm::LogPrint("OTStubProducerVectorHitStyle") << "Barrel tilted cuts:";
      for (size_t i = 0; i < maxWidthBarrelTiltedFloat.size(); ++i) {
        edm::LogPrint("OTStubProducerVectorHitStyle")
            << "  maxWidthBarrelTilted[" << i << "] = " << maxWidthBarrelTiltedFloat[i];
      }
#endif

      maxWidthBarrelFlat_d_ = cms::alpakatools::make_device_buffer<float[]>(queue, maxWidthBarrelFlatFloat.size());
      maxWidthBarrelTilted_d_ = cms::alpakatools::make_device_buffer<float[]>(queue, maxWidthBarrelTiltedFloat.size());
      maxWidthEndcap_d_ = cms::alpakatools::make_device_buffer<float[]>(queue, maxWidthEndcapFloat.size());

      alpaka::memcpy(queue,
                     *maxWidthBarrelFlat_d_,
                     cms::alpakatools::make_host_view(maxWidthBarrelFlatFloat.data(), maxWidthBarrelFlatFloat.size()));
      alpaka::memcpy(
          queue,
          *maxWidthBarrelTilted_d_,
          cms::alpakatools::make_host_view(maxWidthBarrelTiltedFloat.data(), maxWidthBarrelTiltedFloat.size()));
      alpaka::memcpy(queue,
                     *maxWidthEndcap_d_,
                     cms::alpakatools::make_host_view(maxWidthEndcapFloat.data(), maxWidthEndcapFloat.size()));

      barrelFlatMaxCSDiff_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelFlatMaxCSDiff_.size());
      barrelTiltedMaxCSDiff_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelTiltedMaxCSDiff_.size());
      endcapMaxCSDiff_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, endcapMaxCSDiff_.size());
      barrelFlatMaxCS_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelFlatMaxCS_.size());
      barrelTiltedMaxCS_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelTiltedMaxCS_.size());
      endcapMaxCS_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, endcapMaxCS_.size());

      alpaka::memcpy(queue,
                     *barrelFlatMaxCSDiff_d_,
                     cms::alpakatools::make_host_view(barrelFlatMaxCSDiff_.data(), barrelFlatMaxCSDiff_.size()));
      alpaka::memcpy(queue,
                     *barrelTiltedMaxCSDiff_d_,
                     cms::alpakatools::make_host_view(barrelTiltedMaxCSDiff_.data(), barrelTiltedMaxCSDiff_.size()));
      alpaka::memcpy(queue,
                     *endcapMaxCSDiff_d_,
                     cms::alpakatools::make_host_view(endcapMaxCSDiff_.data(), endcapMaxCSDiff_.size()));
      alpaka::memcpy(queue,
                     *barrelFlatMaxCS_d_,
                     cms::alpakatools::make_host_view(barrelFlatMaxCS_.data(), barrelFlatMaxCS_.size()));
      alpaka::memcpy(queue,
                     *barrelTiltedMaxCS_d_,
                     cms::alpakatools::make_host_view(barrelTiltedMaxCS_.data(), barrelTiltedMaxCS_.size()));
      alpaka::memcpy(
          queue, *endcapMaxCS_d_, cms::alpakatools::make_host_view(endcapMaxCS_.data(), endcapMaxCS_.size()));

      barrelFlatMaxCSSum_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelFlatMaxCSSum_.size());
      barrelTiltedMaxCSSum_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelTiltedMaxCSSum_.size());
      endcapMaxCSSum_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, endcapMaxCSSum_.size());

      alpaka::memcpy(queue,
                     *barrelFlatMaxCSSum_d_,
                     cms::alpakatools::make_host_view(barrelFlatMaxCSSum_.data(), barrelFlatMaxCSSum_.size()));
      alpaka::memcpy(queue,
                     *barrelTiltedMaxCSSum_d_,
                     cms::alpakatools::make_host_view(barrelTiltedMaxCSSum_.data(), barrelTiltedMaxCSSum_.size()));
      alpaka::memcpy(
          queue, *endcapMaxCSSum_d_, cms::alpakatools::make_host_view(endcapMaxCSSum_.data(), endcapMaxCSSum_.size()));
      cutsUploaded_ = true;
    }

    OTStubFormationVectorHitStyleKernelsWrapper kernels(queue);

    stubOffsets_d_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nModules_ + 1);
    alpaka::memset(queue, *stubOffsets_d_, 0);

    kernels.countStubs(queue,
                       hitsView,
                       moduleView,
                       geomView,
                       maxWidthBarrelFlat_d_->data(),
                       maxWidthBarrelTilted_d_->data(),
                       maxWidthEndcap_d_->data(),
                       barrelFlatMaxCSDiff_d_->data(),
                       barrelTiltedMaxCSDiff_d_->data(),
                       endcapMaxCSDiff_d_->data(),
                       barrelFlatMaxCS_d_->data(),
                       barrelTiltedMaxCS_d_->data(),
                       endcapMaxCS_d_->data(),
                       barrelFlatMaxCSSum_d_->data(),
                       barrelTiltedMaxCSSum_d_->data(),
                       endcapMaxCSSum_d_->data(),
                       stubOffsets_d_->data(),
                       nModules_);

    // Device-side prefix scan turning the per-module counts into exclusive offsets.
    kernels.finalizeOffsets(queue, stubOffsets_d_->data(), nModules_);

    // Total stub count for the allocation in produce(); async, the framework synchronises.
    nStubs_h_ = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 1u);
    auto nStubs_view_d =
        cms::alpakatools::make_device_view(alpaka::getDev(queue), stubOffsets_d_->data() + nModules_, 1u);
    alpaka::memcpy(queue, *nStubs_h_, nStubs_view_d);

#ifdef OTSTUB_DEBUG_VERBOSE
    // Per-module stub count diagnostic; the geometry is on the device, so no per-layer breakdown.
    {
      auto offsets_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules_ + 1);
      alpaka::memcpy(queue, offsets_h, *stubOffsets_d_);
      alpaka::wait(queue);
      auto const* offsets = alpaka::getPtrNative(offsets_h);

      uint64_t moduleHash = 0;
      uint32_t nModulesWithStubs = 0;
      for (uint32_t i = 0; i < nModules_; ++i) {
        uint32_t count = offsets[i + 1] - offsets[i];
        if (count > 0) {
          nModulesWithStubs++;
          moduleHash ^= (uint64_t(i) * 2654435761u) ^ (uint64_t(count) * 40503u);
        }
      }
      edm::LogPrint("OTStubProducerVectorHitStyle")
          << "VectorHitStyle: " << offsets[nModules_] << " stubs in " << nModulesWithStubs
          << " modules (per-module count hash " << moduleHash << ")";
    }
#endif
  }

  void OTStubProducerVectorHitStyle::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();

    uint32_t nStubs = nStubs_h_->data()[0];

    reco::StubsSoACollection outputStubs(queue, nStubs, nModules_);

    if (nStubs > 0) {
      auto const& otRecHits = iEvent.get(hitToken_);
      auto const& geomDevice = iSetup.getData(geomToken_);
      auto const& geomView = geomDevice.const_view();

      auto const& hitsView = otRecHits.const_view().otRecHits();
      auto const& moduleView = otRecHits.const_view().otHitModules();

      auto stubsView = outputStubs.view().stubs();

      OTStubFormationVectorHitStyleKernelsWrapper kernels(queue);
      kernels.formStubs(queue,
                        hitsView,
                        moduleView,
                        geomView,
                        maxWidthBarrelFlat_d_->data(),
                        maxWidthBarrelTilted_d_->data(),
                        maxWidthEndcap_d_->data(),
                        barrelFlatMaxCSDiff_d_->data(),
                        barrelTiltedMaxCSDiff_d_->data(),
                        endcapMaxCSDiff_d_->data(),
                        barrelFlatMaxCS_d_->data(),
                        barrelTiltedMaxCS_d_->data(),
                        endcapMaxCS_d_->data(),
                        barrelFlatMaxCSSum_d_->data(),
                        barrelTiltedMaxCSSum_d_->data(),
                        endcapMaxCSSum_d_->data(),
                        stubOffsets_d_->data(),
                        stubsView,
                        nModules_);
    }

    iEvent.emplace(stubToken_, std::move(outputStubs));

    // Per-event members only: the cut-table buffers are job constants.
    stubOffsets_d_.reset();
    nStubs_h_.reset();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(OTStubProducerVectorHitStyle);
