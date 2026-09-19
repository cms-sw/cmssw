// CATrackFeaturesTableProducer - nano flat table with the displaced track classifier's
// 35-feature vector (CATrackDNN ABI), computed HOST-SIDE from the final track SoA hit
// lists via the shared RecoTracker/PixelSeeding/interface/CATrackFeatures.h -- the same
// single-source-of-truth function Kernel_classifyTracks evaluates on device.
//
// This is the robust ntuple route for displaced track-classifier training data
// (replaces the printf dump): pair this table with the TrkDispTruth table from
// trackNano_config.py and join offline by (fitPt, phi).
//
// Rows: ALL tracks of the SoA in SoA order (same row space as TrkDispFull);
// tracks with <3 hits / corrupt hit lists get NaN features (drop offline).
//
// OPTIONAL ATTACH-PURITY TABLE (emitHitTruth=True, OFF by default). A SECOND FlatTable, one row
// PER (track, hit) in SoA hit-list order, carrying per-hit truth so the purity of the attached OT
// rechits can be measured against the track's own TrackingParticle. This
// producer walks the PRE-conversion track SoA, where the tag bit caOTHitTag::isOTId(h) marks an
// attached raw-OT extra and origRecHitIdx resolves it to the flat Phase2 rechit collection, so it
// can join hits to TPs the same way HitTruthTableProducer does (TrackerHitAssociator over
// Phase2TrackerRecHit1DCollectionNew; Phase2OTDigiSimLink -> SimHitIdpr -> TP). The track's own TP
// is the plurality TP over the track's resolvable (stub + OT-extra) hits -- an in-producer,
// hit-based match (same family as the associator the truth tables use), used here because the SoA
// row space differs from the legacy-converted-track row space the tpKey ValueMap keys on. Emit
// ownTpNShared so a shared-hit threshold can be applied offline.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/PixelSeeding/interface/CAHitsView.h"
#include "RecoTracker/PixelSeeding/interface/CATrackFeatures.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "RecoTracker/PixelSeeding/test/models/TrackerLayerId.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class CATrackFeaturesTableProducer : public edm::global::EDProducer<> {
public:
  explicit CATrackFeaturesTableProducer(const edm::ParameterSet& params)
      : tableName_(params.getParameter<std::string>("tableName")),
        tracksToken_(consumes<reco::TracksHost>(params.getParameter<edm::InputTag>("trackSrc"))),
        pixelHitsToken_(consumes<reco::TrackingRecHitHost>(params.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        stubsToken_(consumes<reco::StubsHost>(params.getParameter<edm::InputTag>("stubsSrc"))),
        otHitsToken_(consumes<reco::OTRecHitsHost>(params.getParameter<edm::InputTag>("otRecHitsSoASrc"))),
        emitHitTruth_(params.getParameter<bool>("emitHitTruth")),
        hitTableName_(params.getParameter<std::string>("hitTableName")),
        minSharedForOwnTP_(params.getParameter<int>("minSharedForOwnTP")) {
    produces<nanoaod::FlatTable>(tableName_);
    if (emitHitTruth_) {
      otRecHitCollToken_ =
          consumes<Phase2TrackerRecHit1DCollectionNew>(params.getParameter<edm::InputTag>("otRecHitSrc"));
      tpToken_ = consumes<std::vector<TrackingParticle>>(params.getParameter<edm::InputTag>("trackingParticleSrc"));
      topoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
      geomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
      hitAssocConfig_.emplace(params.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector());
      tpChargedOnly_ = params.getParameter<bool>("TP_chargedOnly");
      tpSignalOnly_ = params.getParameter<bool>("TP_signalOnly");
      tpMinPt_ = params.getParameter<double>("TP_minPt");
      tpMaxEta_ = params.getParameter<double>("TP_maxEta");
      tpMaxTip_ = params.getParameter<double>("TP_maxTip");
      tpMaxVtxZ_ = params.getParameter<double>("TP_maxVtxZ");
      produces<nanoaod::FlatTable>(hitTableName_);
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("tableName", "TrkDispCA");
    desc.add<edm::InputTag>("trackSrc", edm::InputTag("hltPhase2PixelTracksSoALowPt"));
    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltPhase2SiPixelRecHitsSoA"));
    // Raw OT-rechit SoA: resolves tagged OT extras attached by the in-CA fit extension so the
    // deployed 12-feature vector matches the device Kernel_classifyTracks on OT-extended tracks.
    desc.add<edm::InputTag>("otRecHitsSoASrc", edm::InputTag("hltPixelSeedingOTRecHitsSoA"));

    // --- Optional attach-purity per-hit truth table (OFF by default; test-config only). ---
    desc.add<bool>("emitHitTruth", false)
        ->setComment("Also emit a per-(track,hit) truth table (isOTExtra/isStub/isTrueForOwnTP) for the purity study");
    desc.add<std::string>("hitTableName", "TrkDispCAHit");
    desc.add<int>("minSharedForOwnTP", 3)
        ->setComment("(diagnostic only) # shared resolvable hits for the own-TP plurality to count as matched");
    desc.add<edm::InputTag>("stubsSrc", edm::InputTag("hltOTStubProducer"))
        ->setComment("StubsHost (lowerHitIdx/upperHitIdx -> OTRecHitsSoA indices)");
    desc.add<edm::InputTag>("otRecHitSrc", edm::InputTag("hltSiPhase2RecHits"))
        ->setComment("Phase2TrackerRecHit1DCollectionNew for OT truth matching");
    desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
    desc.add<bool>("TP_signalOnly", false);
    desc.add<bool>("TP_chargedOnly", true);
    desc.add<double>("TP_minPt", 0.0);
    desc.add<double>("TP_maxEta", 4.5);
    desc.add<double>("TP_maxTip", 60.0);
    desc.add<double>("TP_maxVtxZ", 60.0);
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

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    const auto& tracksHost = iEvent.get(tracksToken_);
    const auto& pixelHits = iEvent.get(pixelHitsToken_);
    const auto& stubsColl = iEvent.get(stubsToken_);
    const auto& otHits = iEvent.get(otHitsToken_);
    const auto tracks = tracksHost.const_view().tracks();
    const auto trackHits = tracksHost.const_view().trackHits();
    // The global hit index space the track hit ids point into: pixel rechits, then stubs. Same
    // facade the CA and the device selectors read, so the columns agree bit for bit.
    const caStructures::CAHitsView hh(pixelHits.const_view().trackingHits(),
                                      pixelHits.const_view().hitModules(),
                                      stubsColl.const_view().stubs(),
                                      stubsColl.const_view().stubModules(),
                                      pixelHits.nHits(),
                                      stubsColl.nStubs(),
                                      pixelHits.nModules());
    const auto otView = otHits.const_view().otRecHits();
    const int nHitsTot = hh.size();
    const uint32_t nOTHits = otView.metadata().size();
    const ::reco::OTRecHitsConstView* otViewPtr = (nOTHits > 0u) ? &otView : nullptr;
    const int nTracks = tracks.nTracks();

    // Column names == caTrackFeatures::fill order == train_disp_nano.py FEATS (12 feats).
    static const char* kNames[caTrackFeatures::kNFeat] = {
        "fitChi2", "psFrac", "r0", "nPS", "nh", "spanZ", "nStubs", "nl", "logChi2Stub", "kErr", "dcaEst", "nBarrel"};

    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::vector<float>> cols(caTrackFeatures::kNFeat, std::vector<float>(nTracks, kNaN));
    std::vector<float> phiCol(nTracks, kNaN);  // join key vs the truth table
    std::vector<int> qualityCol(nTracks, -1);  // SoA quality enum value

    // --- Candidate features (host-only study; targeting the hard 1-10cm band). The existing
    // CA features capture stub-INTERNAL consistency (logChi2Stub, kErr); these add extrapolation/shape,
    // which the current set lacks. If any prove out, port to caTrackFeatures::fill for deployment.
    // Order == kExtraNames.
    // nTilted/tiltedFrac target the tilted-module fakes (|eta|~1-2, <5cm): barrel stubs on TILTED
    // modules (StubFlags !isFlat) -- biased bend on tilted sensors -> near-beamline combinatorial fakes.
    static const char* kExtraNames[7] = {
        "meanStubKappa", "leverArm", "rMax", "rzChi2", "meanClusterY", "nTilted", "tiltedFrac"};
    std::vector<std::vector<float>> ex(7, std::vector<float>(nTracks, kNaN));

    float feat[caTrackFeatures::kNFeat];
    for (int it = 0; it < nTracks; ++it) {
      const uint32_t start = (it == 0) ? 0 : tracks[it - 1].hitOffsets();
      const uint32_t end = tracks[it].hitOffsets();
      if (end <= start || end > uint32_t(trackHits.metadata().size()))
        continue;
      const auto* hitsBegin = trackHits.id().data() + start;
      const auto* hitsEnd = trackHits.id().data() + end;
      const bool ok = caTrackFeatures::fill(
          hitsBegin, hitsEnd, hh, nHitsTot, float(tracks[it].nLayers()), tracks[it].chi2(), feat, nullptr, otViewPtr);
      if (!ok)
        continue;
      for (int f = 0; f < caTrackFeatures::kNFeat; ++f)
        cols[f][it] = feat[f];
      phiCol[it] = tracks[it].state()(0);
      qualityCol[it] = int(tracks[it].quality());

      // Candidate-feature host walk over the same hit list.
      float r0 = -1.f, rMax = 0.f, sumKw = 0.f, sumKwk = 0.f;
      int nClY = 0, nrz = 0, nTilted = 0;
      double sumClY = 0.0, Sr = 0, Sz = 0, Srr = 0, Srz = 0, Szz = 0;
      for (const uint32_t* ph = hitsBegin; ph != hitsEnd; ++ph) {
        const uint32_t h = *ph;
        // Tagged OT extras index the OT SoA, not hh, and are never stubs -> skip them in this
        // host-only study walk (do NOT break, which would truncate the sums at the first
        // OT hit). The 12-feature vector above is OT-aware via fill(); these study-only
        // columns (leverArm/rMax/rzChi2/...) omit OT extras.
        if (caOTHitTag::isOTId(h))
          continue;
        if (h >= uint32_t(nHitsTot))
          break;
        const float rg = hh[h].rGlobal(), zg = hh[h].zGlobal();
        if (r0 < 0.f)
          r0 = rg;
        rMax = std::max(rMax, rg);
        Sr += rg;
        Sz += zg;
        Srr += double(rg) * rg;
        Srz += double(rg) * zg;
        Szz += double(zg) * zg;
        ++nrz;
        // Cluster sizes are a pixel-only column; an outer-tracker entry has none.
        if (!hh.isOTEntry(int32_t(h))) {
          const short cly = hh.pixel(int32_t(h)).clusterSizeY();
          if (cly > 0) {
            sumClY += cly;
            ++nClY;
          }
        }
        if (isStub(hh, h)) {
          auto const stub = hh.stub(int32_t(h));
          if (stub.isBarrel() && !stub.isFlat())
            ++nTilted;  // tilted barrel module (|eta|~1-2 transition)
          // isStub(hh, h) guarantees an outer-tracker stub entry, so the bend columns are readable.
          const float s = stub.dPhiDrError();
          if (s > 0.f) {
            const float d = stub.dPhiDr();
            float den, w;  // same shared kappa formula as CATrackFeatures::fill (single source)
            caTrackFeatures::stubDenWeight(rg * rg, d, s, den, w);
            const float k = d / std::sqrt(den);
            sumKw += w;
            sumKwk += w * k;
          }
        }
      }
      ex[0][it] = (sumKw > 0.f) ? sumKwk / sumKw : 0.f;  // meanStubKappa
      ex[1][it] = rMax - r0;                             // leverArm
      ex[2][it] = rMax;                                  // rMax
      if (nrz >= 3) {                                    // reduced chi2 of a straight line z = a + b r (RSS from sums)
        const double D = nrz * Srr - Sr * Sr;
        if (std::abs(D) > 0.0) {
          const double b = (nrz * Srz - Sr * Sz) / D, a = (Sz - b * Sr) / nrz;
          ex[3][it] = float(std::max(0.0, (Szz - a * Sz - b * Srz)) / std::max(1, nrz - 2));
        }
      }
      ex[4][it] = (nClY > 0) ? float(sumClY / nClY) : 0.f;  // meanClusterY
      ex[5][it] = float(nTilted);                           // nTilted
      ex[6][it] = nTilted / std::max(feat[6], 1.f);         // tiltedFrac (feat[6]=nStubs)
    }

    auto table = std::make_unique<nanoaod::FlatTable>(nTracks, tableName_, /*singleton*/ false, /*extension*/ false);
    for (int f = 0; f < caTrackFeatures::kNFeat; ++f)
      table->addColumn<float>(kNames[f], cols[f], "CA track classifier feature", -1);
    for (int f = 0; f < 7; ++f)
      table->addColumn<float>(
          kExtraNames[f], ex[f], "candidate track feature under study (extrapolation/shape, host-only)", -1);
    table->addColumn<float>("phi", phiCol, "track phi (join key vs the truth table)", -1);
    table->addColumn<int>("quality", qualityCol, "SoA quality enum value", -1);
    iEvent.put(std::move(table), tableName_);

    if (emitHitTruth_)
      produceHitTruthTable(iEvent, iSetup, tracks, trackHits, hh, otView, nOTHits, nTracks);
  }

  // ---- Attach-purity per-(track,hit) truth table -----------------------------------------------
  void produceHitTruthTable(edm::Event& iEvent,
                            const edm::EventSetup& iSetup,
                            const ::reco::TrackSoAConstView& tracks,
                            const ::reco::TrackHitSoAConstView& trackHits,
                            const caStructures::CAHitsView& hh,
                            const ::reco::OTRecHitsConstView& otView,
                            uint32_t nOTHits,
                            int nTracks) const {
    const auto& tTopo = iSetup.getData(topoToken_);
    const auto& tGeom = iSetup.getData(geomToken_);
    const auto& stubsHost = iEvent.get(stubsToken_);  // NOLINT: re-fetched, cheap reference
    const auto stubsView = stubsHost.const_view().stubs();
    const uint32_t nStubs = stubsView.metadata().size();
    const uint32_t offsetStubs = hh.offsetStubs();
    const auto& otRecHitColl = iEvent.get(otRecHitCollToken_);
    edm::Handle<std::vector<TrackingParticle>> tpH;
    iEvent.getByToken(tpToken_, tpH);

    TrackerHitAssociator hitAssociator(iEvent, *hitAssocConfig_);

    // (simTrackId, eventId) -> TP index
    std::map<SimHitIdpr, uint32_t> simTrackToTP;
    for (uint32_t iTP = 0; iTP < tpH->size(); ++iTP)
      for (const auto& g4 : (*tpH)[iTP].g4Tracks())
        simTrackToTP[{g4.trackId(), g4.eventId()}] = iTP;

    // TP selection (broad; the purity join wants any real association not to be dropped).
    std::vector<char> tpSel(tpH->size(), 0);
    for (uint32_t iTP = 0; iTP < tpH->size(); ++iTP) {
      const auto& tp = (*tpH)[iTP];
      if (tpChargedOnly_ && tp.charge() == 0)
        continue;
      if (tpSignalOnly_ && (tp.eventId().bunchCrossing() != 0 || tp.eventId().event() != 0))
        continue;
      if (tp.pt() < tpMinPt_ || std::abs(tp.eta()) > tpMaxEta_ || std::abs(tp.d0()) > tpMaxTip_ ||
          std::abs(tp.vz()) > tpMaxVtxZ_)
        continue;
      tpSel[iTP] = 1;
    }

    // flat Phase2 rechit index -> {set<selected TP index>, layerId} (same flat iteration the
    // OTRecHitsSoA converter used to assign origRecHitIdx).
    std::unordered_map<uint32_t, std::set<uint32_t>> origIdxToTPs;
    std::unordered_map<uint32_t, int> origIdxToLayer;
    {
      uint32_t flatIdx = 0;
      for (const auto& detSet : otRecHitColl) {
        const int layer =
            int(tracknano::getLayerId<pixelTopology::Phase2, uint16_t>(DetId(detSet.detId()), &tTopo, &tGeom));
        for (const auto& recHit : detSet) {
          origIdxToLayer[flatIdx] = layer;
          std::vector<SimHitIdpr> ids;
          hitAssociator.associatePhase2TrackerRecHit(&recHit, ids);
          for (const auto& id : ids) {
            auto it = simTrackToTP.find(id);
            if (it != simTrackToTP.end() && tpSel[it->second])
              origIdxToTPs[flatIdx].insert(it->second);
          }
          ++flatIdx;
        }
      }
    }
    auto soaTPs = [&](uint32_t soaIdx) -> const std::set<uint32_t>* {
      if (soaIdx >= nOTHits)
        return nullptr;
      auto it = origIdxToTPs.find(otView[soaIdx].origRecHitIdx());
      return (it != origIdxToTPs.end()) ? &it->second : nullptr;
    };
    auto soaLayer = [&](uint32_t soaIdx) -> int {
      if (soaIdx >= nOTHits)
        return -1;
      auto it = origIdxToLayer.find(otView[soaIdx].origRecHitIdx());
      return (it != origIdxToLayer.end()) ? it->second : -1;
    };

    // Per-(track,hit) rows.
    const uint32_t* hitIds = trackHits.id().data();
    std::vector<int> cTrackIdx, cLayer, cIsStub, cIsOTExtra, cIsTrue, cOwnTp, cOwnN, cHasTP, cHitId, cHitNTP, cHitTpKey;
    for (int it = 0; it < nTracks; ++it) {
      const uint32_t start = (it == 0) ? 0 : tracks[it - 1].hitOffsets();
      const uint32_t end = tracks[it].hitOffsets();
      if (end <= start || end > uint32_t(trackHits.metadata().size()))
        continue;

      // Resolve every hit's TP set + layer + kind first (one short list per track).
      const uint32_t nh = end - start;
      std::vector<std::set<uint32_t>> hitTPs(nh);
      std::vector<int> hitLayer(nh, -1), hitIsStub(nh, 0), hitIsOT(nh, 0), hitHasTP(nh, 0), hitNTP(nh, 0),
          hitTpKey(nh, -1);
      std::map<uint32_t, int> votes;  // TP index -> # of the track's resolvable hits sharing it
      for (uint32_t k = 0; k < nh; ++k) {
        const uint32_t h = hitIds[start + k];
        std::set<uint32_t> tps;
        if (caOTHitTag::isOTId(h)) {
          hitIsOT[k] = 1;
          const uint32_t o = caOTHitTag::otIdx(h);
          hitLayer[k] = soaLayer(o);
          if (const auto* p = soaTPs(o))
            tps = *p;
        } else if (h >= offsetStubs) {
          hitIsStub[k] = 1;
          const uint32_t stubIdx = h - offsetStubs;
          if (stubIdx < nStubs) {
            const uint32_t lowerIdx = stubsView[stubIdx].lowerHitIdx();
            const uint32_t upperIdx = stubsView[stubIdx].upperHitIdx();
            hitLayer[k] = soaLayer(lowerIdx);
            const auto* lo = soaTPs(lowerIdx);
            const auto* up = soaTPs(upperIdx);
            const bool paired = ::reco::isStub(stubsView, stubIdx) && upperIdx < nOTHits;
            if (lo) {
              if (paired) {
                if (up)
                  std::set_intersection(
                      lo->begin(), lo->end(), up->begin(), up->end(), std::inserter(tps, tps.begin()));
              } else {
                tps = *lo;  // PHitOnly: lower sensor only
              }
            }
          }
        }
        // pixel hits (h < offsetStubs, not tagged): left unresolved (tps empty, layer -1).
        if (!tps.empty()) {
          hitHasTP[k] = 1;
          for (uint32_t tp : tps)
            ++votes[tp];
        }
        hitNTP[k] = int(tps.size());  // raw associator TP-set size, BEFORE the plurality vote (shared-hit multiplicity)
        // Per-HIT TP key: the primary (smallest-index) selected TP this individual hit resolves to,
        // independent of the track's own-TP plurality. Lets an offline tool join hit -> TP -> layerId
        // across ALL tracks, e.g. to reconstruct each TP's true OT layer set. hitNTP>1 hits are ambiguous.
        hitTpKey[k] = tps.empty() ? -1 : int(*tps.begin());
        hitTPs[k] = std::move(tps);
      }

      // own TP = plurality over resolvable hits (ties -> smallest index, since std::map is ordered).
      int ownTp = -1, ownN = 0;
      for (const auto& kv : votes)
        if (kv.second > ownN) {
          ownN = kv.second;
          ownTp = int(kv.first);
        }

      for (uint32_t k = 0; k < nh; ++k) {
        cTrackIdx.push_back(it);
        cHitId.push_back(int(hitIds[start + k]));
        cLayer.push_back(hitLayer[k]);
        cIsStub.push_back(hitIsStub[k]);
        cIsOTExtra.push_back(hitIsOT[k]);
        cOwnTp.push_back(ownTp);
        cOwnN.push_back(ownN);
        cHasTP.push_back(hitHasTP[k]);
        cHitNTP.push_back(hitNTP[k]);
        cHitTpKey.push_back(hitTpKey[k]);
        // isTrueForOwnTP: only for resolvable (OT-extra / stub) hits on a RELIABLY matched track
        // (own-TP plurality shared by >= minSharedForOwnTP_ resolvable hits). ownTpNShared is also
        // emitted so a different threshold can be applied offline.
        int isTrue = -1;
        if ((hitIsOT[k] || hitIsStub[k]) && ownTp >= 0 && ownN >= minSharedForOwnTP_)
          isTrue = hitTPs[k].count(uint32_t(ownTp)) ? 1 : 0;
        cIsTrue.push_back(isTrue);
      }
    }

    const unsigned nRows = cTrackIdx.size();
    auto t = std::make_unique<nanoaod::FlatTable>(nRows, hitTableName_, /*singleton*/ false, /*extension*/ false);
    t->addColumn<int>("trackIdx", cTrackIdx, "SoA track index this hit belongs to", -1);
    t->addColumn<int>("hitId", cHitId, "raw track-hit id (global pixel/stub index; OT extras tagged bit30)", -1);
    t->addColumn<int>("layerId", cLayer, "getLayerId (Phase2 V1) OT layer 28-53, -1 if pixel/unresolved", -1);
    t->addColumn<int>("isStub", cIsStub, "1 if merged stub-region hit", -1);
    t->addColumn<int>("isOTExtra", cIsOTExtra, "1 if attached raw-OT extra (tag bit set)", -1);
    t->addColumn<int>("isTrueForOwnTP", cIsTrue, "1 hit shares track's own TP, 0 not, -1 unresolved/no ownTP", -1);
    t->addColumn<int>("ownTpKey", cOwnTp, "plurality TP index over the track's resolvable hits (-1 none)", -1);
    t->addColumn<int>("ownTpNShared", cOwnN, "# resolvable hits sharing the plurality TP", -1);
    t->addColumn<int>("hitHasTP", cHasTP, "1 if this hit resolved to >=1 selected TP", -1);
    t->addColumn<int>(
        "hitNTP",
        cHitNTP,
        "# distinct selected TrackingParticles the associator matched to this hit, BEFORE the "
        "track's plurality vote (0 = unresolved/pixel or no TP; used for the shared-hit multiplicity study)",
        -1);
    t->addColumn<int>("hitTpKey",
                      cHitTpKey,
                      "primary (smallest-index) selected TP this hit resolves to, independent of the track's ownTpKey "
                      "(-1 = unresolved/pixel or no TP). Join hit->TP->layerId across tracks for the Phase-0 per-layer "
                      "budget/captured/lost decomposition; hitNTP>1 flags an ambiguous (multi-TP) hit",
                      -1);
    iEvent.put(std::move(t), hitTableName_);
  }

  const std::string tableName_;
  const edm::EDGetTokenT<reco::TracksHost> tracksToken_;
  const edm::EDGetTokenT<reco::TrackingRecHitHost> pixelHitsToken_;
  const edm::EDGetTokenT<reco::StubsHost> stubsToken_;
  const edm::EDGetTokenT<reco::OTRecHitsHost> otHitsToken_;

  // Optional attach-purity table members (used only when emitHitTruth_).
  const bool emitHitTruth_;
  const std::string hitTableName_;
  const int minSharedForOwnTP_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitCollToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  std::optional<TrackerHitAssociator::Config> hitAssocConfig_;
  bool tpChargedOnly_ = true;
  bool tpSignalOnly_ = false;
  double tpMinPt_ = 0.0;
  double tpMaxEta_ = 4.5;
  double tpMaxTip_ = 60.0;
  double tpMaxVtxZ_ = 60.0;
};

DEFINE_FWK_MODULE(CATrackFeaturesTableProducer);
