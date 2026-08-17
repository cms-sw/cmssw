// Phase2OTVectorHitsSoAProducer - vector hit formation for the Phase-2 Outer Tracker.
// Define OTVECTORHIT_DEBUG_VERBOSE (or CXXFLAGS="-DOTVECTORHIT_DEBUG_VERBOSE") for verbose GPU->CPU debug.

#include <algorithm>
#include <memory>
#include <vector>

#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/VectorHitsSoACollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/Phase2OTVectorHitsFormationKernelsWrapper.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The kernels index the cut tables by the TrackerTopology layer of the stack: 1-6 in the barrel, 1-5 in the
  // endcap; entry 0 is not used.
  struct WidthCuts {
    float barrelFlat[7];
    float barrelTilted[7];
    float endcap[6];
  };
  using WidthCutsCache = cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<WidthCuts>>;

  class Phase2OTVectorHitsSoAProducer : public stream::SynchronizingEDProducer<edm::GlobalCache<WidthCutsCache>> {
  public:
    Phase2OTVectorHitsSoAProducer(edm::ParameterSet const& iConfig, WidthCutsCache const*);
    ~Phase2OTVectorHitsSoAProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static std::unique_ptr<WidthCutsCache> initializeGlobalCache(edm::ParameterSet const& iConfig);
    static void globalEndJob(WidthCutsCache*) {}

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    device::EDGetToken<reco::OTRecHitsSoACollection> hitToken_;

    device::ESGetToken<reco::StackedModuleGeometrySoACollection, StackedModuleGeometryRecord> geomToken_;

    device::EDPutToken<reco::VectorHitsSoACollection> vectorHitToken_;

    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> vectorHitOffsets_d_;
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> nVectorHits_h_;
    uint32_t nModules_ = 0;
  };

  Phase2OTVectorHitsSoAProducer::Phase2OTVectorHitsSoAProducer(edm::ParameterSet const& iConfig, WidthCutsCache const*)
      : SynchronizingEDProducer(iConfig),
        hitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))),
        geomToken_(esConsumes()),
        vectorHitToken_(produces()) {}

  std::unique_ptr<WidthCutsCache> Phase2OTVectorHitsSoAProducer::initializeGlobalCache(
      edm::ParameterSet const& iConfig) {
    auto fill = [&iConfig](char const* name, auto& cuts) {
      auto const table = iConfig.getParameter<std::vector<double>>(name);
      if (table.size() < std::size(cuts)) {
        throw cms::Exception("Configuration")
            << name << " has " << table.size() << " entries, at least " << std::size(cuts)
            << " are needed (entry i is the cut of layer i, entry 0 unused)";
      }
      std::copy_n(table.begin(), std::size(cuts), cuts);
    };
    PortableHostObject<WidthCuts> widthCuts(cms::alpakatools::host());
    fill("maxWidthBarrelFlat", widthCuts->barrelFlat);
    fill("maxWidthBarrelTilted", widthCuts->barrelTilted);
    fill("maxWidthEndcap", widthCuts->endcap);
    return std::make_unique<WidthCutsCache>(std::move(widthCuts));
  }

  void Phase2OTVectorHitsSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("caOTRecHitsSoA"))
        ->setComment("Input OT RecHits SoA collection");
    desc.add<std::vector<double>>("maxWidthBarrelFlat", {0.0, 0.0575, 0.06, 0.08, 0.09, 0.12, 0.2})
        ->setComment(
            "Layer-dependent width cuts for flat barrel modules (cm), indexed by the barrel layer 1-6 (entry 0 "
            "unused); at least 7 entries");
    desc.add<std::vector<double>>("maxWidthBarrelTilted", {0.0, 0.0575, 0.06, 0.08, 0.09, 0.12, 0.2})
        ->setComment(
            "Layer-dependent width cuts for tilted barrel modules (cm), indexed by the barrel layer 1-6 (entry 0 "
            "unused); at least 7 entries. Only layers 1-3 have tilted modules.");
    desc.add<std::vector<double>>("maxWidthEndcap", {0.0, 0.1, 0.1, 0.1, 0.1, 0.1})
        ->setComment(
            "Layer-dependent width cuts for endcap modules (cm), indexed by the disk 1-5 (entry 0 unused); at least 6 "
            "entries");
    descriptions.addWithDefaultLabel(desc);
  }

  void Phase2OTVectorHitsSoAProducer::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();

    auto const& otRecHits = iEvent.get(hitToken_);

    auto const& geomDevice = iSetup.getData(geomToken_);
    auto const& geomView = geomDevice.const_view();
    nModules_ = geomView.metadata().size();

    auto const& hitsView = otRecHits.const_view().otRecHits();
    auto const& moduleView = otRecHits.const_view().otHitModules();
    auto const* widthCuts = globalCache()->get(queue).data();

    Phase2OTVectorHitsFormationKernelsWrapper kernels;

    vectorHitOffsets_d_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nModules_ + 1);
    alpaka::memset(queue, *vectorHitOffsets_d_, 0);

    kernels.countVectorHits(queue,
                            hitsView,
                            moduleView,
                            geomView,
                            widthCuts->barrelFlat,
                            widthCuts->barrelTilted,
                            widthCuts->endcap,
                            vectorHitOffsets_d_->data(),
                            nModules_);

    // Device-side prefix scan turning the per-module counts into exclusive offsets.
    kernels.finalizeOffsets(queue, vectorHitOffsets_d_->data(), nModules_);

    // Total vector hit count for the allocation in produce(); async, the framework synchronises.
    nVectorHits_h_ = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 1u);
    auto nVectorHits_view_d = cms::alpakatools::make_device_view(queue, vectorHitOffsets_d_->data() + nModules_, 1u);
    alpaka::memcpy(queue, *nVectorHits_h_, nVectorHits_view_d);

#ifdef OTVECTORHIT_DEBUG_VERBOSE
    // Per-module vector hit count diagnostic; the geometry is on the device, so no per-layer breakdown.
    {
      auto offsets_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules_ + 1);
      alpaka::memcpy(queue, offsets_h, *vectorHitOffsets_d_);
      alpaka::wait(queue);
      auto const* offsets = alpaka::getPtrNative(offsets_h);

      uint64_t moduleHash = 0;
      uint32_t nModulesWithVectorHits = 0;
      for (uint32_t i = 0; i < nModules_; ++i) {
        uint32_t count = offsets[i + 1] - offsets[i];
        if (count > 0) {
          nModulesWithVectorHits++;
          moduleHash ^= (uint64_t(i) * 2654435761u) ^ (uint64_t(count) * 40503u);
        }
      }
      edm::LogPrint("Phase2OTVectorHitsSoAProducer")
          << ": " << offsets[nModules_] << " vector hits in " << nModulesWithVectorHits
          << " modules (per-module count hash " << moduleHash << ")";
    }
#endif
  }

  void Phase2OTVectorHitsSoAProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();

    uint32_t nVectorHits = nVectorHits_h_->data()[0];

    // nModules_ is the number of OT stacks, so the module block has nModules_ + 1 rows.
    reco::VectorHitsSoACollection outputVectorHits(queue, nVectorHits, nModules_);

    // The per-module vector hit offsets the acquire() prefix scan produced are the vector-hit-local module
    // starts: an exclusive scan over the OT module range with vectorHitOffsets[nModules_] == nVectorHits, and
    // the formation kernel writes each module's vector hits into [vectorHitOffsets[i], vectorHitOffsets[i + 1]).
    // Copy them into the SoA device-to-device on the same queue (no host round trip). With no vector hits
    // the scan output is all zeros, which is still the right answer, so this is unconditional.
    {
      auto moduleStartView = cms::alpakatools::make_device_view(
          alpaka::getDev(queue), outputVectorHits.view().vectorHitModules().moduleStart().data(), nModules_ + 1);
      alpaka::memcpy(queue, moduleStartView, *vectorHitOffsets_d_);
    }

    if (nVectorHits > 0) {
      auto vectorHitsView = outputVectorHits.view().vectorHits();

      auto const& otRecHits = iEvent.get(hitToken_);
      auto const& geomDevice = iSetup.getData(geomToken_);
      auto const& geomView = geomDevice.const_view();

      auto const& hitsView = otRecHits.const_view().otRecHits();
      auto const& moduleView = otRecHits.const_view().otHitModules();
      auto const* widthCuts = globalCache()->get(queue).data();

      Phase2OTVectorHitsFormationKernelsWrapper kernels;
      kernels.formVectorHits(queue,
                             hitsView,
                             moduleView,
                             geomView,
                             widthCuts->barrelFlat,
                             widthCuts->barrelTilted,
                             widthCuts->endcap,
                             vectorHitOffsets_d_->data(),
                             vectorHitsView,
                             nModules_);
    }

    iEvent.emplace(vectorHitToken_, std::move(outputVectorHits));

    vectorHitOffsets_d_.reset();
    nVectorHits_h_.reset();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(Phase2OTVectorHitsSoAProducer);
