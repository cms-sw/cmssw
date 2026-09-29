#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelSeedingOTRecHitsSoAConverter : public stream::EDProducer<edm::stream::WatchRuns> {
  public:
    explicit PixelSeedingOTRecHitsSoAConverter(const edm::ParameterSet& iConfig);
    ~PixelSeedingOTRecHitsSoAConverter() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;

  private:
    void produce(device::Event& iEvent, const device::EventSetup& es) override;

    edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitToken_;
    edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;

    edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;

    edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomTokenEvent_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

    edm::EDPutTokenT<::reco::OTRecHitsHost> otRecHitsToken_;
    // Also produce module start indices for legacy track converter (HMSstorage = std::vector<uint32_t>)
    edm::EDPutTokenT<std::vector<uint32_t>> hmsToken_;

    std::unordered_map<uint32_t, uint32_t> stackDetIdToGeomIdx_;  // stack DetId -> StackedModuleGeometry index (0-13199)
    std::vector<uint32_t> orderedStacks_;                         // ordered list of stack DetIds
    uint32_t nModules_;                                           // total number of stacked modules in geometry

    // Per-sensor (DetSet) descriptor: pointers valid for the input collection's lifetime;
    // heavy per-hit work deferred to the fill pass. Held as members to reuse capacity.
    struct DetSetInfo {
      const Phase2TrackerRecHit1D* begin;
      const Phase2TrackerRecHit1D* end;
      const GeomDet* det;
      uint32_t iGeom;        // StackedModuleGeometry index
      uint32_t sensorDetId;  // raw DetId of this sensor (constant within the DetSet)
      uint32_t flatStart;    // flat running RecHit index of this DetSet's first hit
      bool isLower;
    };
    std::vector<DetSetInfo> keptDetSets_;  // selected OT sensors for the current event
    std::vector<uint32_t> lowerCount_;     // per-module lower-sensor hit count
    std::vector<uint32_t> upperCount_;     // per-module upper-sensor hit count
    std::vector<uint32_t> lowerCursor_;    // per-module lower-sensor write cursor
    std::vector<uint32_t> upperCursor_;    // per-module upper-sensor write cursor

    static constexpr uint32_t nPixelModules_ = 4000;  // Phase-2 has fixed 4000 pixel modules
    uint32_t nBarrelModules_;                         // OT barrel modules (for logging)
    uint32_t nBackwardModules_;                       // OT backward disk modules (for logging)
    uint32_t nForwardModules_;                        // OT forward disk modules (for logging)

    bool psOnly_;  // If true, exclude hits from SS modules (only PS modules loaded)
  };

  PixelSeedingOTRecHitsSoAConverter::PixelSeedingOTRecHitsSoAConverter(const edm::ParameterSet& iConfig)
      : stream::EDProducer<edm::stream::WatchRuns>(iConfig),
        recHitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitSource"))),
        beamSpotToken_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
        geomToken_(esConsumes<edm::Transition::BeginRun>()),
        geomTokenEvent_(esConsumes()),
        trackerGeomToken_(esConsumes()),
        topoToken_(esConsumes()),
        otRecHitsToken_(produces()),
        hmsToken_(produces()),
        psOnly_(iConfig.getParameter<bool>("psOnly")) {}

  void PixelSeedingOTRecHitsSoAConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("otRecHitSource", edm::InputTag("siPhase2RecHits"))
        ->setComment("Input Phase2TrackerRecHit1D collection");
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"))
        ->setComment("Beam spot for position correction");
    desc.add<bool>("psOnly", false)
        ->setComment("If true, only include hits from PS (Pixel-Strip) modules, excluding SS (Strip-Strip) modules");
    descriptions.addWithDefaultLabel(desc);
  }

  void PixelSeedingOTRecHitsSoAConverter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
    stackDetIdToGeomIdx_.clear();
    orderedStacks_.clear();

    const auto& geomHost = iSetup.getData(geomToken_);
    auto const& geomView = geomHost.const_view();
    nModules_ = geomView.metadata().size();

    // Build map: stack DetId -> StackedModuleGeometry index.
    stackDetIdToGeomIdx_.reserve(2 * nModules_);
    orderedStacks_.reserve(nModules_);
    for (uint32_t iGeom = 0; iGeom < nModules_; ++iGeom) {
      uint32_t stackDetId = geomView[iGeom].stackedDetId();
      stackDetIdToGeomIdx_[stackDetId] = iGeom;
      orderedStacks_.push_back(stackDetId);
    }

    // Size the per-event scratch buffers once (reused across events in produce()).
    keptDetSets_.reserve(2 * nModules_);  // at most 2 sensors (lower + upper) per stacked module
    lowerCount_.resize(nModules_);
    upperCount_.resize(nModules_);
    lowerCursor_.resize(nModules_);
    upperCursor_.resize(nModules_);

    nForwardModules_ = 0;
    nBackwardModules_ = 0;
    nBarrelModules_ = 0;

    for (uint32_t iGeom = 0; iGeom < nModules_; ++iGeom) {
      bool isBarrel = geomView[iGeom].isBarrel();
      bool isFwdEndcap = geomView[iGeom].isFwdEndcap();

      if (isBarrel) {
        nBarrelModules_++;
      } else if (isFwdEndcap) {
        nForwardModules_++;
      } else {
        nBackwardModules_++;
      }
    }

    // Compute CA offsets (for logging)
    uint32_t caBarrelOffset = nPixelModules_;
    uint32_t caForwardOffset = caBarrelOffset + nBarrelModules_;  // z > 0 disks follow the barrel
    uint32_t caBackwardOffset = caForwardOffset + nForwardModules_;

    edm::LogInfo("PixelSeedingOTRecHitsSoAConverter")
        << "OT Module Configuration (geometry sorted in CA order):\n"
        << "  Pixel modules: " << nPixelModules_ << " (CA index 0-" << (nPixelModules_ - 1) << ")\n"
        << "  OT Barrel: " << nBarrelModules_ << " modules (CA index " << caBarrelOffset << "-" << (caForwardOffset - 1)
        << ")\n"
        << "  OT endcap z > 0: " << nForwardModules_ << " modules (CA index " << caForwardOffset << "-"
        << (caBackwardOffset - 1) << ")\n"
        << "  OT endcap z < 0: " << nBackwardModules_ << " modules (CA index " << caBackwardOffset << "-"
        << (caBackwardOffset + nBackwardModules_ - 1) << ")";
  }

  void PixelSeedingOTRecHitsSoAConverter::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();
    const auto& bs = iEvent.get(beamSpotToken_);
    const auto& trackerGeom = iSetup.getData(trackerGeomToken_);
    const auto& topo = iSetup.getData(topoToken_);
    const auto& recHits = iEvent.get(recHitToken_);

    const auto& geomHost = iSetup.getData(geomTokenEvent_);
    auto const& geomView = geomHost.const_view();

    // Beam-spot offsets kept as double to get better precision in the corrections
    const double bsX = bs.x0();
    const double bsY = bs.y0();
    const double bsZ = bs.z0();

    auto& keptDetSets = keptDetSets_;
    auto& lowerCount = lowerCount_;
    auto& upperCount = upperCount_;
    keptDetSets.clear();
    std::fill(lowerCount.begin(), lowerCount.end(), 0u);
    std::fill(upperCount.begin(), upperCount.end(), 0u);

    // First pass: select the stacked-module sensors and accumulate per-module counts.
    // The flat RecHit index runs over every DetSet (including skipped ones) so that
    // origRecHitIdx matches the position in the input collection.
    uint32_t totalHits = 0;
    uint32_t flatRecHitIdx = 0;
    for (const auto& detSet : recHits) {
      uint32_t n = detSet.size();
      DetId detId(detSet.detId());

      uint32_t stackDetId = topo.stack(detId);
      if (stackDetId == 0) {
        flatRecHitIdx += n;
        continue;
      }

      auto geomIt = stackDetIdToGeomIdx_.find(stackDetId);
      if (geomIt == stackDetIdToGeomIdx_.end()) {
        flatRecHitIdx += n;
        continue;
      }

      uint32_t iGeom = geomIt->second;

      if (psOnly_ && !geomView[iGeom].isPS()) {
        flatRecHitIdx += n;
        continue;
      }

      bool isLower = topo.isLower(detId);
      keptDetSets.push_back(
          {detSet.begin(), detSet.end(), trackerGeom.idToDet(detId), iGeom, detId.rawId(), flatRecHitIdx, isLower});
      if (isLower) {
        lowerCount[iGeom] += n;
      } else {
        upperCount[iGeom] += n;
      }
      totalHits += n;
      flatRecHitIdx += n;
    }

    // Host collection, filled entirely on the host; pinned memory from the queue's caching
    // allocator gives asynchronous DMA and block reuse across events. No sync is needed: the queue
    // only selects the allocator and the product outlives the device transform's copy.
    const uint32_t nStacks = orderedStacks_.size();
    ::reco::OTRecHitsHost hostHits(queue, totalHits, nModules_);

    auto hitsView = hostHits.view().otRecHits();
    auto moduleView = hostHits.view().otHitModules();

    // Per-module hit ranges as a prefix sum: [moduleStart, upperSensorStart) holds the
    // lower-sensor hits, [upperSensorStart, nextModuleStart) the upper-sensor hits. The same
    // running offsets seed the per-module write cursors (fully overwritten, no reset needed).
    auto& lowerCursor = lowerCursor_;
    auto& upperCursor = upperCursor_;
    uint32_t running = 0;
    for (uint32_t iGeom = 0; iGeom < nModules_; ++iGeom) {
      uint32_t moduleStart = running;
      uint32_t upperStart = running + lowerCount[iGeom];
      moduleView[iGeom].moduleStart() = moduleStart;
      moduleView[iGeom].upperSensorStart() = upperStart;
      moduleView[iGeom].stackDetId() = orderedStacks_[iGeom];
      lowerCursor[iGeom] = moduleStart;
      upperCursor[iGeom] = upperStart;
      running += lowerCount[iGeom] + upperCount[iGeom];
    }

    // Final sentinel element at index nModules_.
    moduleView[nModules_].moduleStart() = totalHits;
    moduleView[nModules_].upperSensorStart() = totalHits;
    moduleView[nModules_].stackDetId() = 0;

    // Compute each hit's position once and place it directly into the SoA.
    for (const auto& info : keptDetSets) {
      uint32_t writeIdx = info.isLower ? lowerCursor[info.iGeom] : upperCursor[info.iGeom];
      uint32_t flatIdx = info.flatStart;
      // CA module index: nPixelModules + geometry index (geometry sorted in CA order).
      const uint16_t detectorIndex = static_cast<uint16_t>(nPixelModules_ + info.iGeom);
      // The sensor surface is constant over this DetSet; hoist the reference out of the hit loop.
      const auto& sensorSurface = info.det->surface();

      for (const Phase2TrackerRecHit1D* rh = info.begin; rh != info.end; ++rh) {
        auto localPos = rh->localPosition();
        auto localErr = rh->localPositionError();
        // Global position (beam-spot corrected, consistent with pixel hits)
        auto globalPos = sensorSurface.toGlobal(localPos);

        // One element proxy per hit instead of one per column assignment.
        auto hit = hitsView[writeIdx];
        hit.xLocal() = localPos.x();
        hit.yLocal() = localPos.y();
        hit.xerrLocal() = localErr.xx();  // Store variance, not sigma (consistent with pixel hits)
        hit.yerrLocal() = localErr.yy();  // Store variance, not sigma (consistent with pixel hits)
        hit.xGlobal() = globalPos.x() - bsX;
        hit.yGlobal() = globalPos.y() - bsY;
        hit.zGlobal() = globalPos.z() - bsZ;
        hit.detectorIndex() = detectorIndex;
        hit.sensorDetId() = info.sensorDetId;
        hit.origRecHitIdx() = flatIdx;
        hit.clusterSize() = rh->cluster()->size();
        ++writeIdx;
        ++flatIdx;
      }

      if (info.isLower) {
        lowerCursor[info.iGeom] = writeIdx;
      } else {
        upperCursor[info.iGeom] = writeIdx;
      }
    }

    LogDebug("PixelSeedingOTRecHitsSoAConverter")
        << "Converted " << totalHits << " OT hits from " << nStacks << " stacked modules (geometry has " << nModules_
        << " total modules)";

    // HMS storage: module start indices indexed by geometry index, a bulk copy of moduleView.
    std::vector<uint32_t> hmsStorage(nModules_ + 1);
    std::memcpy(hmsStorage.data(), moduleView.moduleStart().data(), sizeof(uint32_t) * (nModules_ + 1));

    iEvent.emplace(hmsToken_, std::move(hmsStorage));
    iEvent.emplace(otRecHitsToken_, std::move(hostHits));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelSeedingOTRecHitsSoAConverter);
