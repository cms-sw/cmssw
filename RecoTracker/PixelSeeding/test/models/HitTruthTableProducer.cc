// Per-hit to TrackingParticle truth FlatTable for the built-triplet dataset: one row per hit of the
// global (pixel + stub) index space, the row index being the hit index carried by the Triplet table, holding the
// matched TrackingParticle key or -1. A triplet is real when its three hits share one tpKey >= 0.
//
// Association chain:
//   i  < offsetStubs  pixel hit; invert the SoA->legacy pixel mapping of SiPixelRecHitFromSoAAlpaka
//                     (mergedIdx = pixelModuleStart[gind] + cluster.originalId()), then associate
//                     the legacy pixel rechit to a TP.
//   i >= offsetStubs  stubIdx = i - offsetStubs -> StubsSoA lower/upper hit index ->
//                     OTRecHitsSoA origRecHitIdx() -> flat Phase2TrackerRecHit1DCollectionNew ->
//                     TrackerHitAssociator -> SimHitIdpr -> TP.
// A paired stub requires both sensor hits to share the TP (intersection); a PHitOnly stub uses the
// lower-sensor hit alone. The matched TP is the smallest selected TP index sharing the hit.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/PixelSeeding/test/models/TrackerLayerId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "RecoTracker/PixelSeeding/interface/CAHitsView.h"

class HitTruthTableProducer : public edm::stream::EDProducer<> {
public:
  explicit HitTruthTableProducer(const edm::ParameterSet& iConfig);
  ~HitTruthTableProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const std::string tableName_;

  edm::EDGetTokenT<reco::TrackingRecHitHost> pixelHitsSoAToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsSoAToken_;
  edm::EDGetTokenT<reco::StubsHost> stubsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitCollectionToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitCollectionToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;

  // Maps a pixel DetId to its geom det index, which is the pixel moduleStart index, and provides
  // local-to-global for the merged-hit-index self-check.
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  bool emitHitFeatures_;
  // Second table, one row per raw OT rechit in OTRecHitsSoA index space: the join target for
  // caOTHitTag-tagged hit ids, which do not live in the merged-hit space of the primary table.
  std::string otTableName_;

  TrackerHitAssociator::Config hitAssocConfig_;

  bool tpSignalOnly_;
  bool tpChargedOnly_;
  double tpMinPt_;
  double tpMaxEta_;
  double tpMaxTip_;
  double tpMaxVtxZ_;
};

HitTruthTableProducer::HitTruthTableProducer(const edm::ParameterSet& iConfig)
    : tableName_(iConfig.getParameter<std::string>("tableName")),
      pixelHitsSoAToken_(consumes<reco::TrackingRecHitHost>(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASrc"))),
      otRecHitsSoAToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSoASrc"))),
      stubsToken_(consumes<reco::StubsHost>(iConfig.getParameter<edm::InputTag>("stubsSrc"))),
      otRecHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("otRecHitSrc"))),
      pixelRecHitCollectionToken_(
          consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      tpToken_(consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      geomToken_(esConsumes()),
      emitHitFeatures_(iConfig.getParameter<bool>("emitHitFeatures")),
      otTableName_(iConfig.getParameter<std::string>("otTableName")),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()),
      tpSignalOnly_(iConfig.getParameter<bool>("TP_signalOnly")),
      tpChargedOnly_(iConfig.getParameter<bool>("TP_chargedOnly")),
      tpMinPt_(iConfig.getParameter<double>("TP_minPt")),
      tpMaxEta_(iConfig.getParameter<double>("TP_maxEta")),
      tpMaxTip_(iConfig.getParameter<double>("TP_maxTip")),
      tpMaxVtxZ_(iConfig.getParameter<double>("TP_maxVtxZ")) {
  produces<nanoaod::FlatTable>(tableName_);
  if (emitHitFeatures_) {
    topoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
    produces<nanoaod::FlatTable>(otTableName_);
  }
}

void HitTruthTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& pixelHitsSoA = iEvent.get(pixelHitsSoAToken_);
  const auto& otHitsSoA = iEvent.get(otRecHitsSoAToken_);
  const auto& stubsHost = iEvent.get(stubsToken_);
  const auto& otRecHitCollection = iEvent.get(otRecHitCollectionToken_);
  const auto& pixelRecHitCollection = iEvent.get(pixelRecHitCollectionToken_);
  edm::Handle<std::vector<TrackingParticle>> tpHandle;
  iEvent.getByToken(tpToken_, tpHandle);

  auto otHitsView = otHitsSoA.view().otRecHits();
  auto stubsView = stubsHost.const_view().stubs();
  // The global hit index space the CA indexes: pixel rechits first, then stubs (CAHitsView.h).
  const caStructures::CAHitsView mergedView(pixelHitsSoA.const_view().trackingHits(),
                                            pixelHitsSoA.const_view().hitModules(),
                                            stubsHost.const_view().stubs(),
                                            stubsHost.const_view().stubModules(),
                                            pixelHitsSoA.nHits(),
                                            stubsHost.nStubs(),
                                            pixelHitsSoA.nModules());

  uint32_t nMergedHits = uint32_t(mergedView.size());
  uint32_t offsetStubs = mergedView.offsetStubs();
  uint32_t nOTHits = otHitsSoA.nHits();
  uint32_t nStubs = stubsView.metadata().size();

  TrackerHitAssociator hitAssociator(iEvent, hitAssocConfig_);

  // (simTrackId, eventId) -> index into tpHandle.
  std::map<SimHitIdpr, uint32_t> simTrackToTP;
  for (uint32_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    const auto& tp = (*tpHandle)[iTP];
    for (const auto& g4Track : tp.g4Tracks()) {
      simTrackToTP[{g4Track.trackId(), g4Track.eventId()}] = iTP;
    }
  }

  std::vector<bool> tpSelected(tpHandle->size(), false);
  for (uint32_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    const auto& tp = (*tpHandle)[iTP];
    if (tpChargedOnly_ && tp.charge() == 0)
      continue;
    if (tpSignalOnly_ && (tp.eventId().bunchCrossing() != 0 || tp.eventId().event() != 0))
      continue;
    if (tp.pt() < tpMinPt_)
      continue;
    if (std::abs(tp.eta()) > tpMaxEta_)
      continue;
    if (std::abs(tp.d0()) > tpMaxTip_)
      continue;
    if (std::abs(tp.vz()) > tpMaxVtxZ_)
      continue;
    tpSelected[iTP] = true;
  }

  // Flat Phase2 rechit index -> OTRecHitsSoA index.
  std::unordered_map<uint32_t, uint32_t> origIdxToSoAIdx;
  origIdxToSoAIdx.reserve(nOTHits);
  for (uint32_t iHit = 0; iHit < nOTHits; ++iHit) {
    origIdxToSoAIdx[otHitsView[iHit].origRecHitIdx()] = iHit;
  }

  // Needed for the per-hit CA layerId only; origIdxToLayer maps a flat Phase2 rechit index to it.
  const TrackerTopology* tTopo = nullptr;
  const TrackerGeometry* tGeom = nullptr;
  if (emitHitFeatures_) {
    tTopo = &iSetup.getData(topoToken_);
    tGeom = &iSetup.getData(geomToken_);
  }
  std::unordered_map<uint32_t, int> origIdxToLayer;

  // Flat Phase2 index -> selected TP indices, in the same sequential iteration over the DetSetVector
  // that PixelSeedingOTRecHitsSoAConverter uses to assign origRecHitIdx.
  std::unordered_map<uint32_t, std::set<uint32_t>> origIdxToTPs;
  {
    uint32_t flatIdx = 0;
    for (const auto& detSet : otRecHitCollection) {
      // The CA layer is constant within a detSet.
      const int detLayer =
          emitHitFeatures_
              ? int(tracknano::getLayerId<pixelTopology::Phase2, uint16_t>(DetId(detSet.detId()), tTopo, tGeom))
              : -1;
      for (const auto& recHit : detSet) {
        std::vector<SimHitIdpr> ids;
        hitAssociator.associatePhase2TrackerRecHit(&recHit, ids);
        for (const auto& id : ids) {
          auto it = simTrackToTP.find(id);
          if (it != simTrackToTP.end() && tpSelected[it->second])
            origIdxToTPs[flatIdx].insert(it->second);
        }
        if (emitHitFeatures_)
          origIdxToLayer[flatIdx] = detLayer;
        flatIdx++;
      }
    }
  }

  // OTRecHitsSoA index -> selected TP indices.
  auto soaIdxToTPs = [&](uint32_t soaIdx) -> const std::set<uint32_t>* {
    if (soaIdx >= nOTHits)
      return nullptr;
    auto it = origIdxToTPs.find(otHitsView[soaIdx].origRecHitIdx());
    return (it != origIdxToTPs.end()) ? &it->second : nullptr;
  };

  // Row index is the merged-hit index. tpKey is the smallest selected TP sharing the hit, or -1.
  constexpr int kNoTP = -1;
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  std::vector<int> tpKey(nMergedHits, kNoTP);
  std::vector<float> tpPt(nMergedHits, kNaN);
  std::vector<float> tpEta(nMergedHits, kNaN);

  // Optional geometry/feature columns, empty unless emitHitFeatures_.
  std::vector<float> xg, yg, zg, rg, dPhiDrC, dPhiDrErrC;
  std::vector<int> layerC, isStubC, stubFlagsC, clSizeX, clSizeY;
  if (emitHitFeatures_) {
    xg.assign(nMergedHits, kNaN);
    yg.assign(nMergedHits, kNaN);
    zg.assign(nMergedHits, kNaN);
    rg.assign(nMergedHits, kNaN);
    dPhiDrC.assign(nMergedHits, kNaN);
    dPhiDrErrC.assign(nMergedHits, kNaN);
    layerC.assign(nMergedHits, -1);
    isStubC.assign(nMergedHits, 0);
    stubFlagsC.assign(nMergedHits, 0);
    clSizeX.assign(nMergedHits, -1);
    clSizeY.assign(nMergedHits, -1);
    // OT layer via the flat rechit index; the pixel-region layer is filled in the pixel loop below.
    auto soaLayer = [&](uint32_t soaIdx) -> int {
      if (soaIdx >= nOTHits)
        return -1;
      auto it = origIdxToLayer.find(otHitsView[soaIdx].origRecHitIdx());
      return (it != origIdxToLayer.end()) ? it->second : -1;
    };
    for (uint32_t i = 0; i < nMergedHits; ++i) {
      xg[i] = mergedView[i].xGlobal();
      yg[i] = mergedView[i].yGlobal();
      zg[i] = mergedView[i].zGlobal();
      rg[i] = mergedView[i].rGlobal();
      // Source-specific columns come from the matching element of the hit view; a pixel hit publishes
      // bend 0, error -1 and no flags, an outer-tracker entry publishes no cluster.
      if (mergedView.isOTEntry(int32_t(i))) {
        auto const stub = mergedView.stub(int32_t(i));
        dPhiDrC[i] = stub.dPhiDr();
        dPhiDrErrC[i] = stub.dPhiDrError();
        stubFlagsC[i] = int(stub.flags());
        clSizeX[i] = 0;
        clSizeY[i] = 0;
      } else {
        auto const pix = mergedView.pixel(int32_t(i));
        dPhiDrC[i] = 0.f;
        dPhiDrErrC[i] = -1.f;
        stubFlagsC[i] = 0;
        clSizeX[i] = int(pix.clusterSizeX());
        clSizeY[i] = int(pix.clusterSizeY());
      }
      isStubC[i] = isStub(mergedView, int32_t(i)) ? 1 : 0;
      if (i >= offsetStubs) {
        const uint32_t stubIdx = i - offsetStubs;
        if (stubIdx < nStubs)
          layerC[i] = soaLayer(stubsView[stubIdx].lowerHitIdx());
      }
    }
  }

  for (uint32_t i = 0; i < nMergedHits; ++i) {
    // The stub-region gate is the index, not isStub(mergedView, i): the latter is the
    // paired-stub flag, false for a PHitOnly hit, and would drop single-sensor stub-region hits.
    bool inStubRegion = (i >= offsetStubs);

    std::set<uint32_t> tps;
    if (inStubRegion) {
      uint32_t stubIdx = i - offsetStubs;
      if (stubIdx < nStubs) {
        uint32_t lowerIdx = stubsView[stubIdx].lowerHitIdx();
        uint32_t upperIdx = stubsView[stubIdx].upperHitIdx();
        const auto* lowerTPs = soaIdxToTPs(lowerIdx);
        const auto* upperTPs = soaIdxToTPs(upperIdx);
        bool hasUpper = reco::isStub(stubsView, stubIdx) && upperIdx < nOTHits;
        if (lowerTPs) {
          if (hasUpper) {
            // Genuine pair: intersect the lower and upper TP sets.
            if (upperTPs)
              std::set_intersection(lowerTPs->begin(),
                                    lowerTPs->end(),
                                    upperTPs->begin(),
                                    upperTPs->end(),
                                    std::inserter(tps, tps.begin()));
          } else {
            // PHitOnly: lower sensor only.
            tps = *lowerTPs;
          }
        }
      }
    }
    // Pixel merged hits (i < offsetStubs) are labeled in the pixel loop below.

    if (!tps.empty()) {
      const uint32_t matched = *tps.begin();  // smallest selected TP index sharing the hit
      tpKey[i] = static_cast<int>(matched);
      const auto& tp = (*tpHandle)[matched];
      tpPt[i] = static_cast<float>(tp.pt());
      tpEta[i] = static_cast<float>(tp.eta());
    }
  }

  // Pixel hit labeling, which the OT loop above leaves at tpKey = -1. A legacy pixel rechit
  // built from cluster clust in det gind = geom.idToDetUnit(detid)->index() occupies pixel-SoA index
  // moduleStart[gind] + clust.originalId(), which is also its index in the global hit index space.
  // Only indices < offsetStubs are written.
  // The global-position self-check below guards against a wrong inversion for a given geometry.
  {
    const TrackerGeometry& geom = iSetup.getData(geomToken_);
    auto hitModulesView = pixelHitsSoA.const_view().hitModules();
    uint32_t nPixChecked = 0, nPixMismatch = 0, nPixLabeled = 0;
    for (auto const& detSet : pixelRecHitCollection) {
      const DetId detid(detSet.detId());
      const GeomDet* det = geom.idToDetUnit(detid);
      if (det == nullptr)
        continue;
      const uint32_t gind = static_cast<uint32_t>(det->index());
      const uint32_t base = hitModulesView.moduleStart()[gind];
      // Pixel CA layer, 0-27, constant within a detSet.
      const int pixLayer =
          emitHitFeatures_ ? int(tracknano::getLayerId<pixelTopology::Phase2, uint16_t>(detid, tTopo, &geom)) : -1;
      for (auto const& rechit : detSet) {
        const uint32_t mergedIdx = base + rechit.cluster()->originalId();
        if (mergedIdx >= offsetStubs || mergedIdx >= nMergedHits)
          continue;  // pixel region only, which also guards a bad inversion
        if (emitHitFeatures_)
          layerC[mergedIdx] = pixLayer;

        // Only (x,y) are compared: the SoA zGlobal carries a pixel-CPE sensor-depth reference that
        // the host toGlobal(local 2D) does not, a 150-200 um systematic in z, while (x,y) agree to
        // about 1e-5 cm. A wrong index lands on a different hit and disagrees transversely.
        const GlobalPoint gp = det->toGlobal(rechit.localPosition());
        const float dx = gp.x() - mergedView[mergedIdx].xGlobal();
        const float dy = gp.y() - mergedView[mergedIdx].yGlobal();
        ++nPixChecked;
        if (dx * dx + dy * dy > 1.e-3f)  // above ~0.03 cm transverse the index is wrong
          ++nPixMismatch;

        std::vector<SimHitIdpr> ids = hitAssociator.associateHitId(rechit);
        std::set<uint32_t> tps;
        for (const auto& id : ids) {
          auto it = simTrackToTP.find(id);
          if (it != simTrackToTP.end() && tpSelected[it->second])
            tps.insert(it->second);
        }
        if (!tps.empty()) {
          const uint32_t matched = *tps.begin();
          tpKey[mergedIdx] = static_cast<int>(matched);
          const auto& tp = (*tpHandle)[matched];
          tpPt[mergedIdx] = static_cast<float>(tp.pt());
          tpEta[mergedIdx] = static_cast<float>(tp.eta());
          ++nPixLabeled;
        }
      }
    }
    if (nPixMismatch > 0)
      edm::LogError("HitTruthTableProducer")
          << "PIXEL MERGED-HIT INDEX MAPPING MISMATCH: " << nPixMismatch << "/" << nPixChecked
          << " pixel rechits disagree with the merged SoA global position by > 0.01 cm. The "
             "moduleStart+originalId inversion is WRONG for this geometry/config -- DO NOT trust the "
             "pixel tpKey labels; fix HitTruthTableProducer before retraining.";
    LogDebug("HitTruthTableProducer") << "pixel labeling: checked=" << nPixChecked << " labeled=" << nPixLabeled
                                      << " mismatch=" << nPixMismatch;
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nMergedHits, tableName_, /*singleton*/ false, /*extension*/ false);
  table->addColumn<int>("tpKey", tpKey, "matched TrackingParticle key (-1 if none)", -1);
  table->addColumn<float>("tpPt", tpPt, "matched TP pT [GeV] (NaN if none)", -1);
  table->addColumn<float>("tpEta", tpEta, "matched TP eta (NaN if none)", -1);
  if (emitHitFeatures_) {
    // Row index is the merged-hit index, which is the join key a merged-hit id resolves with.
    table->addColumn<float>("xg", xg, "global x [cm]", -1);
    table->addColumn<float>("yg", yg, "global y [cm]", -1);
    table->addColumn<float>("zg", zg, "global z [cm]", -1);
    table->addColumn<float>("rg", rg, "global r [cm]", -1);
    table->addColumn<int>("layerId", layerC, "CA layerId (pixel 0-27, TOB1-6 28-33, TID 34-53; -1 unresolved)", -1);
    table->addColumn<int>("isStub", isStubC, "1 if paired 2S stub (dPhiDrError>=0)", -1);
    table->addColumn<int>("stubFlags", stubFlagsC, "packed StubFlags (isBarrel/isFlat/isValid/layer); 0 if pixel", -1);
    table->addColumn<float>("dPhiDr", dPhiDrC, "stub bend / local track angle (P2 discriminant); 0 if not a stub", -1);
    table->addColumn<float>("dPhiDrError", dPhiDrErrC, "error on stub bend; <0 if not a stub", -1);
    table->addColumn<int>("clusterSizeX", clSizeX, "cluster size in X", -1);
    table->addColumn<int>("clusterSizeY", clSizeY, "cluster size in Y", -1);
  }
  iEvent.put(std::move(table), tableName_);

  // Second table, one row per raw OT rechit, so caOTHitTag-tagged hit ids join to truth and
  // geometry. Row index k is otIdx == (id & ~kOTHitTag). tpKey and layer are keyed on origRecHitIdx,
  // the same map the merged path uses, so a raw-OT hit and a stub built from it resolve to one TP.
  // dPhiDr is stub-level and has no meaning on a single OT rechit, so it is not emitted here.
  if (emitHitFeatures_) {
    std::vector<int> otTpKey(nOTHits, kNoTP), otLayer(nOTHits, -1), otClSize(nOTHits, -1);
    std::vector<int> otOrigIdx(nOTHits, -1);
    std::vector<float> otTpPt(nOTHits, kNaN), otTpEta(nOTHits, kNaN);
    std::vector<float> otX(nOTHits, kNaN), otY(nOTHits, kNaN), otZ(nOTHits, kNaN), otR(nOTHits, kNaN);
    for (uint32_t k = 0; k < nOTHits; ++k) {
      const uint32_t origIdx = otHitsView[k].origRecHitIdx();
      otOrigIdx[k] = int(origIdx);
      const float x = otHitsView[k].xGlobal(), y = otHitsView[k].yGlobal(), z = otHitsView[k].zGlobal();
      otX[k] = x;
      otY[k] = y;
      otZ[k] = z;
      otR[k] = std::sqrt(x * x + y * y);
      otClSize[k] = int(otHitsView[k].clusterSize());
      auto itL = origIdxToLayer.find(origIdx);
      if (itL != origIdxToLayer.end())
        otLayer[k] = itL->second;
      auto itT = origIdxToTPs.find(origIdx);
      if (itT != origIdxToTPs.end() && !itT->second.empty()) {
        const uint32_t matched = *itT->second.begin();  // smallest selected TP, as in the merged path
        otTpKey[k] = int(matched);
        const auto& tp = (*tpHandle)[matched];
        otTpPt[k] = float(tp.pt());
        otTpEta[k] = float(tp.eta());
      }
    }
    auto otTable =
        std::make_unique<nanoaod::FlatTable>(nOTHits, otTableName_, /*singleton*/ false, /*extension*/ false);
    otTable->addColumn<int>("tpKey", otTpKey, "matched TrackingParticle key (-1 if none)", -1);
    otTable->addColumn<float>("tpPt", otTpPt, "matched TP pT [GeV] (NaN if none)", -1);
    otTable->addColumn<float>("tpEta", otTpEta, "matched TP eta (NaN if none)", -1);
    otTable->addColumn<float>("xg", otX, "global x [cm]", -1);
    otTable->addColumn<float>("yg", otY, "global y [cm]", -1);
    otTable->addColumn<float>("zg", otZ, "global z [cm]", -1);
    otTable->addColumn<float>("rg", otR, "global r [cm]", -1);
    otTable->addColumn<int>("layerId", otLayer, "CA layerId (OT 28-53; -1 unresolved)", -1);
    otTable->addColumn<int>("clusterSize", otClSize, "OT rechit cluster size", -1);
    otTable->addColumn<int>("origRecHitIdx", otOrigIdx, "flat Phase2 rechit index (cross-join key)", -1);
    iEvent.put(std::move(otTable), otTableName_);
  }
}

void HitTruthTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("tableName", "HitTruth");
  desc.add<edm::InputTag>("pixelRecHitSoASrc", edm::InputTag("hltPhase2SiPixelRecHitsSoA"))
      ->setComment("Pixel TrackingRecHit SoA host copy; with stubsSrc it defines the CA global hit index space");
  desc.add<edm::InputTag>("otRecHitsSoASrc", edm::InputTag("hltPixelSeedingOTRecHitsSoA"))
      ->setComment("OTRecHitsSoA host copy (origRecHitIdx -> flat Phase2 index)");
  desc.add<edm::InputTag>("stubsSrc", edm::InputTag("hltOTStubProducer"))
      ->setComment("StubsHost (lowerHitIdx/upperHitIdx -> OTRecHitsSoA indices)");
  desc.add<edm::InputTag>("otRecHitSrc", edm::InputTag("hltSiPhase2RecHits"))
      ->setComment("Phase2TrackerRecHit1DCollectionNew for OT truth matching");
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"))
      ->setComment(
          "SiPixelRecHitCollection (SoA->legacy via SiPixelRecHitFromSoAAlpaka); pixel merged "
          "hits are truth-matched by inverting moduleStart+cluster.originalId().");
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("TrackingParticle collection");

  // When false the table carries tpKey/tpPt/tpEta only; when true it also carries per-hit position,
  // layerId, stub flags, bend and cluster sizes.
  desc.add<bool>("emitHitFeatures", false)
      ->setComment("Also emit per-hit position/layer/stub-bend/cluster-size feature columns (OFF => byte-identical)");
  desc.add<std::string>("otTableName", "HitTruthOT")
      ->setComment("Name of the per-raw-OT-rechit truth+geometry table (only produced when emitHitFeatures=True)");

  // TP_signalOnly defaults to false so that pileup reals stay in the triplet training population;
  // the charged and pt/eta/tip/vz windows define reconstructability.
  desc.add<bool>("TP_signalOnly", false)->setComment("Require BX=0, event=0 (hard scatter)");
  desc.add<bool>("TP_chargedOnly", true)->setComment("Require non-zero charge");
  desc.add<double>("TP_minPt", 0.0)->setComment("Minimum TP pT [GeV]");
  desc.add<double>("TP_maxEta", 4.5)->setComment("Maximum |eta|");
  desc.add<double>("TP_maxTip", 60.0)->setComment("Maximum |d0| transverse impact parameter [cm]");
  desc.add<double>("TP_maxVtxZ", 60.0)->setComment("Maximum |vz| [cm]");

  // TrackerHitAssociator config, mirroring TrueStubProducer.
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

DEFINE_FWK_MODULE(HitTruthTableProducer);
