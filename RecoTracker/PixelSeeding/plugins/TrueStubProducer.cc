// TrueStubProducer - Creates "true stubs" from simulation truth (TrackingParticles).
// CPU-only EDProducer: for each TrackingParticle, finds its RecHits on OT stacked modules,
// pairs the lower/upper sensor hits, and computes stub quantities. The output StubsHost can be
// read by the CA (and the rest of the stub chain) in place of the normal OTStubProducer's, giving it a
// perfect set of stubs (zero fakes, no missed genuine pairs) to measure upper-bound performance.
// No bend or pT cut is applied: all genuine pairs are kept regardless of kinematics.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class TrueStubProducer : public edm::stream::EDProducer<> {
public:
  explicit TrueStubProducer(const edm::ParameterSet& iConfig);
  ~TrueStubProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitCollectionToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsToken_;

  edm::ESGetToken<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  // TrackerHitAssociator config
  TrackerHitAssociator::Config hitAssocConfig_;

  // TP selection cuts
  double tpMinPt_;
  double tpMaxEta_;
  double tpMaxVtxZ_;
  double tpMaxD0_;
  double tpMaxLxy_;
  int tpMinNLayers_;

  // PHitOnly mode: true adds only P-hits from selected TPs that did not form genuine stubs, false
  // adds every unused lower-sensor P-hit on a PS module, as the normal pipeline does.
  bool tpAwarePHitOnly_;

  edm::EDPutTokenT<reco::StubsHost> stubsToken_;
};

TrueStubProducer::TrueStubProducer(const edm::ParameterSet& iConfig)
    : recHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("recHitSrc"))),
      tpToken_(consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      otRecHitsToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"))),
      geomToken_(esConsumes<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord>()),
      topoToken_(esConsumes()),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()),
      tpMinPt_(iConfig.getParameter<double>("TP_minPt")),
      tpMaxEta_(iConfig.getParameter<double>("TP_maxEta")),
      tpMaxVtxZ_(iConfig.getParameter<double>("TP_maxVtxZ")),
      tpMaxD0_(iConfig.getParameter<double>("TP_maxD0")),
      tpMaxLxy_(iConfig.getParameter<double>("TP_maxLxy")),
      tpMinNLayers_(iConfig.getParameter<int>("TP_minNLayers")),
      tpAwarePHitOnly_(iConfig.getParameter<bool>("tpAwarePHitOnly")),
      stubsToken_(produces<reco::StubsHost>()) {}

void TrueStubProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& recHits = iEvent.get(recHitCollectionToken_);
  edm::Handle<std::vector<TrackingParticle>> tpHandle;
  iEvent.getByToken(tpToken_, tpHandle);
  const auto& otHits = iEvent.get(otRecHitsToken_);

  const auto& geomHost = iSetup.getData(geomToken_);
  const auto& topo = iSetup.getData(topoToken_);

  auto hitsView = otHits.view().otRecHits();
  auto const& geomView = geomHost.view();
  uint32_t nOTHits = otHits.nHits();
  uint32_t nModules = geomView.metadata().size();

  TrackerHitAssociator hitAssociator(iEvent, hitAssocConfig_);

  // Build (simTrackId, eventId) -> TrackingParticle map
  std::map<SimHitIdpr, const TrackingParticle*> simTrackToTP;
  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    const auto& tp = (*tpHandle)[iTP];
    for (const auto& g4Track : tp.g4Tracks()) {
      simTrackToTP[{g4Track.trackId(), g4Track.eventId()}] = &tp;
    }
  }

  // Build reverse map: origRecHitIdx -> OTRecHitsSoA index
  std::unordered_map<uint32_t, uint32_t> origIdxToSoAIdx;
  origIdxToSoAIdx.reserve(nOTHits);
  for (uint32_t iHit = 0; iHit < nOTHits; ++iHit) {
    origIdxToSoAIdx[hitsView[iHit].origRecHitIdx()] = iHit;
  }

  // Build stackedDetId -> geometry module index map
  std::unordered_map<uint32_t, uint32_t> stackDetIdToGeomIdx;
  stackDetIdToGeomIdx.reserve(nModules);
  for (uint32_t iGeom = 0; iGeom < nModules; ++iGeom) {
    stackDetIdToGeomIdx[geomView[iGeom].stackedDetId()] = iGeom;
  }

  // Pass 2: for each TP on a stacked module collect its lower and upper sensor hits, then match by
  // (stackDetId, TP) to form genuine pairs.

  // Key: (stackDetId, TP pointer)
  using TPStackKey = std::pair<uint32_t, const TrackingParticle*>;
  struct SensorHitInfo {
    std::vector<uint32_t> lowerOrigIdx;  // origRecHitIdx for lower sensor hits
    std::vector<uint32_t> upperOrigIdx;  // origRecHitIdx for upper sensor hits
  };
  std::map<TPStackKey, SensorHitInfo> tpStackHits;

#ifdef TRUESTUB_DEBUG
  uint32_t nHitsProcessed = 0, nHitsWithIds = 0, nIdsMatched = 0;
#endif

  uint32_t flatIdx = 0;
  for (const auto& detSet : recHits) {
    DetId detId(detSet.detId());
    uint32_t stackId = topo.stack(detId);
    bool isOnStack = (stackId != 0);

    if (!isOnStack) {
      flatIdx += detSet.size();
      continue;
    }

    bool isLower = topo.isLower(detId);

    for (const auto& recHit : detSet) {
      std::vector<SimHitIdpr> ids;
      hitAssociator.associatePhase2TrackerRecHit(&recHit, ids);

#ifdef TRUESTUB_DEBUG
      nHitsProcessed++;
      if (!ids.empty())
        nHitsWithIds++;
      for (const auto& id : ids) {
        if (simTrackToTP.find(id) != simTrackToTP.end())
          nIdsMatched++;
      }
#endif

      for (const auto& id : ids) {
        auto it = simTrackToTP.find(id);
        if (it != simTrackToTP.end()) {
          const TrackingParticle* tp = it->second;
          TPStackKey key{stackId, tp};
          if (isLower) {
            tpStackHits[key].lowerOrigIdx.push_back(flatIdx);
          } else {
            tpStackHits[key].upperOrigIdx.push_back(flatIdx);
          }
        }
      }
      flatIdx++;
    }
  }

#ifdef TRUESTUB_DEBUG
  edm::LogPrint("TrueStubProducer") << "=== Truth Association Diagnostics ===\n"
                                    << "  OT hits on stacked modules processed: " << nHitsProcessed << "\n"
                                    << "  Hits with SimLink associations: " << nHitsWithIds << " ("
                                    << (nHitsProcessed > 0 ? 100.0f * nHitsWithIds / nHitsProcessed : 0.0f) << "%)\n"
                                    << "  SimTrackId-to-TP matches: " << nIdsMatched;
#endif

  // Pre-select TPs passing kinematic cuts
  std::set<const TrackingParticle*> selectedTPs;
#ifdef TRUESTUB_DEBUG
  uint32_t nTPsProcessed = 0;
  uint32_t nTPsCharged = 0, nTPsInTime = 0, nTPsPt = 0, nTPsEta = 0, nTPsVtxZ = 0, nTPsD0 = 0, nTPsLxy = 0;
#endif

  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    const auto& tp = (*tpHandle)[iTP];
#ifdef TRUESTUB_DEBUG
    nTPsProcessed++;
#endif
    if (tp.charge() == 0)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsCharged++;
#endif
    if (tp.eventId().bunchCrossing() != 0)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsInTime++;
#endif
    if (tp.pt() < tpMinPt_)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsPt++;
#endif
    if (std::abs(tp.eta()) > tpMaxEta_)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsEta++;
#endif
    if (std::abs(tp.vz()) > tpMaxVtxZ_)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsVtxZ++;
#endif
    if (std::abs(tp.d0()) > tpMaxD0_)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsD0++;
#endif
    float lxy = std::sqrt(tp.vx() * tp.vx() + tp.vy() * tp.vy());
    if (lxy > tpMaxLxy_)
      continue;
#ifdef TRUESTUB_DEBUG
    nTPsLxy++;
#endif
    selectedTPs.insert(&tp);
  }

  // Additionally, count lower-sensor hits per TP to apply minNLayers cut
  // Build TP -> number of stacked modules with lower hits
  std::map<const TrackingParticle*, std::set<uint32_t>> tpLowerStacks;
  for (const auto& [key, sensorHits] : tpStackHits) {
    if (!sensorHits.lowerOrigIdx.empty()) {
      tpLowerStacks[key.second].insert(key.first);
    }
  }

#ifdef TRUESTUB_DEBUG
  // Distribution of stacked module counts before the minNLayers cut
  {
    std::map<int, int> nStacksDistribution;
    for (const auto* tp : selectedTPs) {
      auto stackIt = tpLowerStacks.find(tp);
      int nStacks = (stackIt != tpLowerStacks.end()) ? static_cast<int>(stackIt->second.size()) : 0;
      nStacksDistribution[nStacks]++;
    }
    edm::LogPrint("TrueStubProducer") << "=== Stacked module count distribution (before minNLayers cut) ===";
    for (const auto& [n, count] : nStacksDistribution) {
      edm::LogPrint("TrueStubProducer") << "  nStacks=" << n << ": " << count << " TPs";
    }
  }

  // Signal vs PU breakdown for selected TPs
  {
    int nSignalTPs = 0, nSignalTPsWithStacks = 0;
    int nPUTPs = 0, nPUTPsWithStacks = 0;
    for (const auto* tp : selectedTPs) {
      bool isSignal = (tp->eventId().bunchCrossing() == 0 && tp->eventId().event() == 0);
      auto stackIt = tpLowerStacks.find(tp);
      int nStacks = (stackIt != tpLowerStacks.end()) ? static_cast<int>(stackIt->second.size()) : 0;
      if (isSignal) {
        nSignalTPs++;
        if (nStacks > 0)
          nSignalTPsWithStacks++;
      } else {
        nPUTPs++;
        if (nStacks > 0)
          nPUTPsWithStacks++;
      }
    }
    edm::LogPrint("TrueStubProducer") << "=== Signal vs PU Breakdown ===\n"
                                      << "  Signal TPs: " << nSignalTPs << " (with OT stacks: " << nSignalTPsWithStacks
                                      << ")\n"
                                      << "  PU TPs: " << nPUTPs << " (with OT stacks: " << nPUTPsWithStacks << ")";
  }

  // Detailed diagnostic for PU TPs with nStacks=0: check SimHit-level info
  {
    int nDiagPrinted = 0;
    int nPUZeroWithSimHits = 0, nPUZeroWithoutSimHits = 0;
    for (const auto* tp : selectedTPs) {
      bool isSignal = (tp->eventId().bunchCrossing() == 0 && tp->eventId().event() == 0);
      if (isSignal)
        continue;
      auto stackIt = tpLowerStacks.find(tp);
      int nStacks = (stackIt != tpLowerStacks.end()) ? static_cast<int>(stackIt->second.size()) : 0;
      if (nStacks > 0)
        continue;
      int nTrackerHits = tp->numberOfTrackerHits();
      int nTrackerLayers = tp->numberOfTrackerLayers();
      int nG4Tracks = tp->g4Tracks().size();
      if (nTrackerHits > 0)
        nPUZeroWithSimHits++;
      else
        nPUZeroWithoutSimHits++;
      if (nDiagPrinted < 10) {
        edm::LogPrint("TrueStubProducer")
            << "  PU TP nStacks=0: pT=" << tp->pt() << " eta=" << tp->eta() << " pdgId=" << tp->pdgId() << " eventId=("
            << tp->eventId().bunchCrossing() << "," << tp->eventId().event() << ")"
            << " nTrackerHits=" << nTrackerHits << " nTrackerLayers=" << nTrackerLayers << " nG4Tracks=" << nG4Tracks;
        for (const auto& g4t : tp->g4Tracks()) {
          edm::LogPrint("TrueStubProducer")
              << "    g4Track: trackId=" << g4t.trackId() << " eventId=(" << g4t.eventId().bunchCrossing() << ","
              << g4t.eventId().event() << ")" << " type=" << g4t.type();
        }
        nDiagPrinted++;
      }
    }
    edm::LogPrint("TrueStubProducer") << "=== PU TPs with nStacks=0 SimHit summary ===\n"
                                      << "  With tracker SimHits: " << nPUZeroWithSimHits << "\n"
                                      << "  Without tracker SimHits: " << nPUZeroWithoutSimHits;
  }
#endif

  // Remove TPs that don't have enough lower-sensor stacked modules
  for (auto it = selectedTPs.begin(); it != selectedTPs.end();) {
    auto stackIt = tpLowerStacks.find(*it);
    if (stackIt == tpLowerStacks.end() || static_cast<int>(stackIt->second.size()) < tpMinNLayers_) {
      it = selectedTPs.erase(it);
    } else {
      ++it;
    }
  }

#ifdef TRUESTUB_DEBUG
  uint32_t nTPsSelected = selectedTPs.size();
#endif

#ifdef TRUESTUB_DEBUG
  edm::LogPrint("TrueStubProducer") << "=== TP Selection Diagnostics ===\n"
                                    << "  Total TPs: " << tpHandle->size() << "\n"
                                    << "  Charged: " << nTPsCharged << "\n"
                                    << "  + in-time (BX=0): " << nTPsInTime << "\n"
                                    << "  + pT > " << tpMinPt_ << ": " << nTPsPt << "\n"
                                    << "  + |eta| < " << tpMaxEta_ << ": " << nTPsEta << "\n"
                                    << "  + |vz| < " << tpMaxVtxZ_ << ": " << nTPsVtxZ << "\n"
                                    << "  + |d0| < " << tpMaxD0_ << ": " << nTPsD0 << "\n"
                                    << "  + Lxy < " << tpMaxLxy_ << ": " << nTPsLxy << "\n"
                                    << "  + minNLayers >= " << tpMinNLayers_ << ": " << nTPsSelected;
#endif

  // Collect genuine pairs from selected TPs
  struct GenuineStub {
    uint32_t lowerSoAIdx;  // index in OTRecHitsSoA for lower sensor hit
    uint32_t upperSoAIdx;  // index in OTRecHitsSoA for upper sensor hit
    uint32_t geomModIdx;   // index in StackedModuleGeometry
  };
  std::vector<GenuineStub> genuineStubs;

  for (const auto& [key, sensorHits] : tpStackHits) {
    const TrackingParticle* tp = key.second;
    uint32_t stackId = key.first;

    if (selectedTPs.find(tp) == selectedTPs.end())
      continue;

    if (sensorHits.lowerOrigIdx.empty() || sensorHits.upperOrigIdx.empty())
      continue;

    auto geomIt = stackDetIdToGeomIdx.find(stackId);
    if (geomIt == stackDetIdToGeomIdx.end())
      continue;
    uint32_t geomIdx = geomIt->second;

    // Form stubs from all lower-upper pairs on this stack for this TP
    for (uint32_t lowerOrig : sensorHits.lowerOrigIdx) {
      auto lowerSoAIt = origIdxToSoAIdx.find(lowerOrig);
      if (lowerSoAIt == origIdxToSoAIdx.end())
        continue;

      for (uint32_t upperOrig : sensorHits.upperOrigIdx) {
        auto upperSoAIt = origIdxToSoAIdx.find(upperOrig);
        if (upperSoAIt == origIdxToSoAIdx.end())
          continue;

        genuineStubs.push_back({lowerSoAIt->second, upperSoAIt->second, geomIdx});
      }
    }
  }

#ifdef TRUESTUB_DEBUG
  {
    uint32_t nPairsBoth = 0, nPairsLowerOnly = 0, nPairsUpperOnly = 0;
    for (const auto& [key, sensorHits] : tpStackHits) {
      if (selectedTPs.find(key.second) == selectedTPs.end())
        continue;
      bool hasLower = !sensorHits.lowerOrigIdx.empty();
      bool hasUpper = !sensorHits.upperOrigIdx.empty();
      if (hasLower && hasUpper)
        nPairsBoth++;
      else if (hasLower)
        nPairsLowerOnly++;
      else if (hasUpper)
        nPairsUpperOnly++;
    }
    edm::LogPrint("TrueStubProducer") << "=== Stub Formation Diagnostics (selected TPs only) ===\n"
                                      << "  (stackId, TP) pairs with both sensors: " << nPairsBoth << "\n"
                                      << "  (stackId, TP) pairs lower-only: " << nPairsLowerOnly << "\n"
                                      << "  (stackId, TP) pairs upper-only: " << nPairsUpperOnly;
  }
#endif

  // PHitOnly recovery: P-hits on PS modules that do not form genuine stubs. tpAwarePHitOnly_ true
  // adds only P-hits from selected TPs (lower-only pairs); false adds every unused lower-sensor
  // P-hit on a PS module, as in the normal pipeline where every P-hit is visible.
  struct PHitOnlyEntry {
    uint32_t lowerSoAIdx;
    uint32_t geomModIdx;
  };

  std::set<uint32_t> usedLowerHits;
  for (const auto& gs : genuineStubs) {
    usedLowerHits.insert(gs.lowerSoAIdx);
  }

  auto otModuleView = otHits.view().otHitModules();
  std::vector<PHitOnlyEntry> phitOnlyEntries;

  if (tpAwarePHitOnly_) {
    // TP-aware mode: only lower-sensor P-hits from selected TPs that did not form genuine stubs
    // (lower-only pairs on PS modules).
    for (const auto& [key, sensorHits] : tpStackHits) {
      const TrackingParticle* tp = key.second;
      uint32_t stackId = key.first;

      if (selectedTPs.find(tp) == selectedTPs.end())
        continue;

      // Only interested in lower-sensor hits that didn't pair with upper
      if (sensorHits.lowerOrigIdx.empty())
        continue;

      auto geomIt = stackDetIdToGeomIdx.find(stackId);
      if (geomIt == stackDetIdToGeomIdx.end())
        continue;
      uint32_t geomIdx = geomIt->second;

      // Only PS modules produce PHitOnly entries
      uint8_t moduleType = geomView[geomIdx].moduleType();
      bool isPS = (moduleType == 0 || moduleType == 1);
      if (!isPS)
        continue;

      for (uint32_t lowerOrig : sensorHits.lowerOrigIdx) {
        auto lowerSoAIt = origIdxToSoAIdx.find(lowerOrig);
        if (lowerSoAIt == origIdxToSoAIdx.end())
          continue;
        uint32_t soaIdx = lowerSoAIt->second;
        if (usedLowerHits.find(soaIdx) == usedLowerHits.end()) {
          phitOnlyEntries.push_back({soaIdx, geomIdx});
          usedLowerHits.insert(soaIdx);  // avoid duplicates from multiple TPs sharing a hit
        }
      }
    }
  } else {
    // Ungated mode: every unused lower-sensor P-hit on a PS module.
    for (uint32_t iMod = 0; iMod < nModules; ++iMod) {
      uint8_t moduleType = geomView[iMod].moduleType();
      bool isPS = (moduleType == 0 || moduleType == 1);
      if (!isPS)
        continue;
      uint32_t hitStart = otModuleView[iMod].moduleStart();
      uint32_t upperStart = otModuleView[iMod].upperSensorStart();

      // Lower-sensor P-hits: hitStart to upperStart
      for (uint32_t iHit = hitStart; iHit < upperStart; ++iHit) {
        if (usedLowerHits.find(iHit) == usedLowerHits.end()) {
          phitOnlyEntries.push_back({iHit, iMod});
        }
      }
    }
  }

  // Stubs are written grouped by module (detectorIndex), the order the module starts describe.
  struct OutputEntry {
    uint32_t geomModIdx;
    bool isGenuine;
    uint32_t sourceIdx;  // index into genuineStubs or phitOnlyEntries
  };

  uint32_t nGenuine = genuineStubs.size();
  uint32_t nPHitOnly = phitOnlyEntries.size();
  uint32_t nStubs = nGenuine + nPHitOnly;

  std::vector<OutputEntry> outputOrder;
  outputOrder.reserve(nStubs);
  for (uint32_t i = 0; i < nGenuine; ++i)
    outputOrder.push_back({genuineStubs[i].geomModIdx, true, i});
  for (uint32_t i = 0; i < nPHitOnly; ++i)
    outputOrder.push_back({phitOnlyEntries[i].geomModIdx, false, i});

  std::stable_sort(outputOrder.begin(), outputOrder.end(), [](const OutputEntry& a, const OutputEntry& b) {
    return a.geomModIdx < b.geomModIdx;
  });

#ifdef TRUESTUB_DEBUG
  edm::LogPrint("TrueStubProducer") << "TPs processed: " << nTPsProcessed << ", selected: " << nTPsSelected
                                    << ", genuine stubs: " << nGenuine << ", PHitOnly: " << nPHitOnly
                                    << ", total stubs: " << nStubs;
#endif

  // Passes 3 and 4: compute the stub quantities and fill the SoA.

  auto output = std::make_unique<reco::StubsHost>(cms::alpakatools::host(), nStubs, nModules);

  // Stub-local module starts: outputOrder is stable-sorted by geomModIdx, so each module's stubs are
  // contiguous. moduleStart[i] is the first stub of OT CA module nModulesPix + i and
  // moduleStart[nModules] is nStubs.
  {
    auto stubModulesView = output->view().stubModules();
    uint32_t iStub = 0;
    for (uint32_t iMod = 0; iMod < nModules; ++iMod) {
      stubModulesView[iMod].moduleStart() = iStub;
      while (iStub < nStubs && outputOrder[iStub].geomModIdx == iMod)
        ++iStub;
    }
    stubModulesView[nModules].moduleStart() = nStubs;
  }

  if (nStubs > 0) {
    auto stubsView = output->view().stubs();

    for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
      const auto& entry = outputOrder[iStub];
      uint32_t geomIdx = entry.geomModIdx;

      bool isBarrel = geomView[geomIdx].isBarrel();
      bool isFlat = geomView[geomIdx].isFlat();
      uint8_t layer = geomView[geomIdx].layer();
      uint8_t moduleType = geomView[geomIdx].moduleType();
      bool isPS = (moduleType == 0 || moduleType == 1);

      if (entry.isGenuine) {
        // Genuine stub.
        const auto& gs = genuineStubs[entry.sourceIdx];
        bool isFlipped = geomView[geomIdx].isFlipped();

        uint32_t lowerIdx = gs.lowerSoAIdx;
        uint32_t upperIdx = gs.upperSoAIdx;
        uint32_t innerIdx = isFlipped ? upperIdx : lowerIdx;
        uint32_t outerIdx = isFlipped ? lowerIdx : upperIdx;

        float xi = hitsView[innerIdx].xGlobal();
        float yi = hitsView[innerIdx].yGlobal();
        float zi = hitsView[innerIdx].zGlobal();
        float xo = hitsView[outerIdx].xGlobal();
        float yo = hitsView[outerIdx].yGlobal();

        float ri2 = xi * xi + yi * yi;
        float ro2 = xo * xo + yo * yo;
        float ri = std::sqrt(ri2);
        float ro = std::sqrt(ro2);

        float xg = hitsView[lowerIdx].xGlobal();
        float yg = hitsView[lowerIdx].yGlobal();
        stubsView[iStub].iphi() = unsafe_atan2s<7>(yg, xg);

        // Position, errors and CA module of the published hit (lowerIdx here).
        stubsView[iStub].xGlobal() = xg;
        stubsView[iStub].yGlobal() = yg;
        stubsView[iStub].zGlobal() = hitsView[lowerIdx].zGlobal();
        stubsView[iStub].rGlobal() = std::sqrt(xg * xg + yg * yg);
        stubsView[iStub].xerrLocal() = hitsView[lowerIdx].xerrLocal();
        stubsView[iStub].yerrLocal() = hitsView[lowerIdx].yerrLocal();
        stubsView[iStub].detectorIndex() = hitsView[lowerIdx].detectorIndex();

        float phi_lower = std::atan2(hitsView[lowerIdx].yGlobal(), hitsView[lowerIdx].xGlobal());
        float phi_upper = std::atan2(hitsView[upperIdx].yGlobal(), hitsView[upperIdx].xGlobal());
        float bend = isFlipped ? (phi_lower - phi_upper) : (phi_upper - phi_lower);
        if (bend > M_PI)
          bend -= 2.0f * M_PI;
        if (bend < -M_PI)
          bend += 2.0f * M_PI;

        // dPhiDr with parallax correction
        {
          float phi_inner = std::atan2(yi, xi);
          float phi_outer = std::atan2(yo, xo);
          float dphi_raw = phi_outer - phi_inner;
          if (dphi_raw > M_PI)
            dphi_raw -= 2.0f * M_PI;
          if (dphi_raw < -M_PI)
            dphi_raw += 2.0f * M_PI;

          float tiltAngle = geomView[geomIdx].tiltAngle();
          float sinTilt = geomView[geomIdx].sinTilt();
          float cosTilt = geomView[geomIdx].cosTilt();
          float separation_cm = geomView[geomIdx].sensorSeparation() * 0.1f;

          float dphi;
          bool applyDphiParallax = !isFlat || !isBarrel;
          if (applyDphiParallax) {
            float gR_inner = ri;
            float parallCorr = 0.0f;
            if (gR_inner > 1e-6f) {
              float cosTheta = std::cos(tiltAngle);
              float sinTheta = std::sin(tiltAngle);
              float lV_x = -gR_inner * sinTheta + zi * cosTheta;
              float lV_z = gR_inner * cosTheta + zi * sinTheta;
              if (std::abs(lV_z) > 1e-6f) {
                float lV_norm_x = lV_x / lV_z;
                parallCorr = lV_norm_x * separation_cm;
                if (edm::isNotFinite(parallCorr))
                  parallCorr = 0.0f;
              }
            }
            float dphi_parallax = (ri > 1e-6f) ? parallCorr / ri : 0.0f;
            dphi = dphi_raw - dphi_parallax;
          } else {
            dphi = dphi_raw;
          }

          float x_stub = hitsView[lowerIdx].xGlobal();
          float y_stub = hitsView[lowerIdx].yGlobal();
          float r_stub = std::sqrt(x_stub * x_stub + y_stub * y_stub);
          float z_stub = hitsView[lowerIdx].zGlobal();
          // tiltAngle = atan2(dz, dr) measured from radial axis, so sinTilt > 0 for +z tilt
          float denominator = cosTilt + sinTilt * z_stub / r_stub;
          float dr_effective;
          if (std::abs(denominator) > 1e-6f)
            dr_effective = separation_cm / denominator;
          else
            dr_effective = ro - ri;

          stubsView[iStub].dPhiDr() = (std::abs(dr_effective) > 1e-6f) ? dphi / dr_effective : 0.0f;

          // Global phi variance: sigma^2(phi) = (sin^2(phi)*Cxx - 2*sin*cos*Cxy + cos^2*Cyy) / r^2,
          // recomputed from the stacked geometry's per-sensor frames (OTRecHitsSoA has no pre-computed
          // global-error columns).
          auto innerFrame = isFlipped ? geomView[geomIdx].upperSensorFrame() : geomView[geomIdx].lowerSensorFrame();
          auto outerFrame = isFlipped ? geomView[geomIdx].lowerSensorFrame() : geomView[geomIdx].upperSensorFrame();
          float ge_i[6], ge_o[6];
          innerFrame.toGlobal(hitsView[innerIdx].xerrLocal(), 0.0f, hitsView[innerIdx].yerrLocal(), ge_i);
          outerFrame.toGlobal(hitsView[outerIdx].xerrLocal(), 0.0f, hitsView[outerIdx].yerrLocal(), ge_o);

          float sinphi_i = std::sin(phi_inner);
          float cosphi_i = std::cos(phi_inner);
          float ri2 = ri * ri;
          float sig2phi_i = (ri2 > 0.0f) ? (sinphi_i * sinphi_i * ge_i[0] - 2.0f * sinphi_i * cosphi_i * ge_i[1] +
                                            cosphi_i * cosphi_i * ge_i[2]) /
                                               ri2
                                         : 0.0f;
          float sinphi_o = std::sin(phi_outer);
          float cosphi_o = std::cos(phi_outer);
          float ro2 = ro * ro;
          float sig2phi_o = (ro2 > 0.0f) ? (sinphi_o * sinphi_o * ge_o[0] - 2.0f * sinphi_o * cosphi_o * ge_o[1] +
                                            cosphi_o * cosphi_o * ge_o[2]) /
                                               ro2
                                         : 0.0f;
          float dphi_err = std::sqrt(sig2phi_i + sig2phi_o);
          stubsView[iStub].dPhiDrError() = (std::abs(dr_effective) > 1e-6f) ? dphi_err / std::abs(dr_effective) : 0.0f;
          // The precision-only bend error is not computed by this producer; copy dPhiDrError so
          // the column is initialised.
          stubsView[iStub].dPhiDrErrorPrec() = stubsView[iStub].dPhiDrError();
        }

        stubsView[iStub].posHitIdx() = lowerIdx;  // this producer publishes the lower hit's position
        stubsView[iStub].lowerHitIdx() = lowerIdx;
        stubsView[iStub].upperHitIdx() = upperIdx;
        stubsView[iStub].flags() = reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer, isPS);
      } else {
        // PHitOnly entry.
        const auto& ph = phitOnlyEntries[entry.sourceIdx];
        uint32_t hitIdx = ph.lowerSoAIdx;

        {
          float xg = hitsView[hitIdx].xGlobal();
          float yg = hitsView[hitIdx].yGlobal();
          // iphi from (xGlobal, yGlobal)
          stubsView[iStub].iphi() = unsafe_atan2s<7>(yg, xg);

          // Same published-hit quantities as above, here the P-hit itself.
          stubsView[iStub].xGlobal() = xg;
          stubsView[iStub].yGlobal() = yg;
          stubsView[iStub].zGlobal() = hitsView[hitIdx].zGlobal();
          stubsView[iStub].rGlobal() = std::sqrt(xg * xg + yg * yg);
          stubsView[iStub].xerrLocal() = hitsView[hitIdx].xerrLocal();
          stubsView[iStub].yerrLocal() = hitsView[hitIdx].yerrLocal();
          stubsView[iStub].detectorIndex() = hitsView[hitIdx].detectorIndex();
        }

        stubsView[iStub].dPhiDr() = 0.0f;
        stubsView[iStub].dPhiDrError() = -1.0f;
        stubsView[iStub].dPhiDrErrorPrec() = -1.0f;
        stubsView[iStub].posHitIdx() = hitIdx;
        stubsView[iStub].lowerHitIdx() = hitIdx;
        stubsView[iStub].upperHitIdx() = UINT32_MAX;
        stubsView[iStub].flags() = reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer, isPS);
      }
    }
  }

  iEvent.put(stubsToken_, std::move(output));
}

void TrueStubProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitSrc", edm::InputTag("siPhase2RecHits"))
      ->setComment("Phase2TrackerRecHit1DCollectionNew for truth matching");
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("TrackingParticle collection");
  desc.add<edm::InputTag>("otRecHitsSrc", edm::InputTag("pixelSeedingOTRecHitsSoA"))
      ->setComment("OTRecHitsSoA collection (host) for hit positions");

  // TP selection
  desc.add<double>("TP_minPt", 0.75)->setComment("Minimum TP pT [GeV]");
  desc.add<double>("TP_maxEta", 2.6)->setComment("Maximum |eta|");
  desc.add<double>("TP_maxVtxZ", 30.0)->setComment("Maximum |vz| [cm]");
  desc.add<double>("TP_maxD0", 1.0)->setComment("Maximum |d0| [cm]");
  desc.add<double>("TP_maxLxy", 1.0)->setComment("Maximum Lxy [cm]");
  desc.add<int>("TP_minNLayers", 4)->setComment("Minimum number of TP lower-sensor hits");
  desc.add<bool>("tpAwarePHitOnly", true)
      ->setComment(
          "If true, PHitOnly entries are restricted to P-hits from selected TPs only. "
          "If false, all unused lower-sensor P-hits on PS modules are added (matching normal pipeline).");

  // TrackerHitAssociator config
  edm::ParameterSetDescription hitAssocDesc;
  hitAssocDesc.add<bool>("associatePixel", true);
  hitAssocDesc.add<bool>("associateStrip", true);
  hitAssocDesc.add<bool>("usePhase2Tracker", true);
  hitAssocDesc.add<bool>("associateRecoTracks", false);
  hitAssocDesc.add<bool>("associateHitbySimTrack", false);
  hitAssocDesc.add<edm::InputTag>("phase2TrackerSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  hitAssocDesc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis", "Pixel"));
  hitAssocDesc.add<edm::InputTag>("stripSimLinkSrc", edm::InputTag("simSiStripDigis"));
  hitAssocDesc.add<std::vector<std::string>>("ROUList",
                                             {"TrackerHitsPixelBarrelLowTof",
                                              "TrackerHitsPixelBarrelHighTof",
                                              "TrackerHitsPixelEndcapLowTof",
                                              "TrackerHitsPixelEndcapHighTof",
                                              "TrackerHitsTIBLowTof",
                                              "TrackerHitsTIBHighTof",
                                              "TrackerHitsTIDLowTof",
                                              "TrackerHitsTIDHighTof",
                                              "TrackerHitsTOBLowTof",
                                              "TrackerHitsTOBHighTof",
                                              "TrackerHitsTECLowTof",
                                              "TrackerHitsTECHighTof"});
  desc.add<edm::ParameterSetDescription>("hitAssociatorConfig", hitAssocDesc);

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TrueStubProducer);
