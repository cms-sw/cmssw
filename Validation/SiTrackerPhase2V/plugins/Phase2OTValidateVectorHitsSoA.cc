// DQM validation of  vector hits: fake rate (vector hits that are combinatoric or unknown),
// efficiency (TrackingParticle lower-sensor RecHits with a genuine vector hit) and per-layer diagnostics,
// broken down into barrel-flat, barrel-tilted, forward+ and forward- modules.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

class Phase2OTValidateVectorHitsSoA : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateVectorHitsSoA(const edm::ParameterSet& iConfig);
  ~Phase2OTValidateVectorHitsSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  enum ModuleCategory { BarrelFlat = 0, BarrelTilted = 1, ForwardPos = 2, ForwardNeg = 3, NCategories = 4 };
  static constexpr const char* categoryNames_[NCategories] = {"barrel_flat", "barrel_tilted", "fwd_pos", "fwd_neg"};
  static constexpr const char* categoryTitles_[NCategories] = {"Barrel Flat", "Barrel Tilted", "Forward+", "Forward-"};

  ModuleCategory moduleCategory(uint8_t flags, float z) const {
    if (reco::VectorHitFlags::isBarrel(flags)) {
      return reco::VectorHitFlags::isFlat(flags) ? BarrelFlat : BarrelTilted;
    }
    return (z > 0) ? ForwardPos : ForwardNeg;
  }

  ModuleCategory moduleCategoryFromDetId(DetId detId, const TrackerTopology& topo, const TrackerGeometry& geom) const {
    bool isBarrel = (detId.subdetId() == StripSubdetector::TOB);
    if (isBarrel) {
      auto tiltType = topo.barrelTiltTypeP2(detId);
      return (tiltType == Phase2Tracker::flat) ? BarrelFlat : BarrelTilted;
    }
    auto const* detUnit = geom.idToDetUnit(detId);
    if (detUnit) {
      return (detUnit->position().z() > 0) ? ForwardPos : ForwardNeg;
    }
    return ForwardPos;
  }

  static int nLayersForCategory(ModuleCategory cat) {
    switch (cat) {
      case BarrelFlat:
        return 6;
      case BarrelTilted:
        return 6;
      case ForwardPos:
        return 5;
      case ForwardNeg:
        return 5;
      default:
        return 6;
    }
  }

  static constexpr int NPerLayerCategories = 22;

  static int perLayerIndex(bool isBarrel, bool isFlat, int layer, bool zPositive) {
    if (isBarrel) {
      if (layer < 1 || layer > 6) {
        return -1;
      }
      return (layer - 1) * 2 + (isFlat ? 0 : 1);
    }
    if (layer < 1 || layer > 5) {
      return -1;
    }
    if (zPositive) {
      return 12 + (layer - 1);
    }
    return 17 + (layer - 1);
  }

  static int perLayerIndexFromVectorHit(uint8_t flags, float z) {
    bool barrel = reco::VectorHitFlags::isBarrel(flags);
    bool flat = reco::VectorHitFlags::isFlat(flags);
    int layer = reco::VectorHitFlags::layer(flags);
    return perLayerIndex(barrel, flat, layer, z > 0);
  }

  int perLayerIndexFromDetId(DetId detId, const TrackerTopology& topo, const TrackerGeometry& geom) const {
    bool barrel = (detId.subdetId() == StripSubdetector::TOB);
    int layer = topo.layer(detId);
    if (barrel) {
      bool flat = (topo.barrelTiltTypeP2(detId) == Phase2Tracker::flat);
      return perLayerIndex(true, flat, layer, true /*unused for barrel*/);
    }
    auto const* detUnit = geom.idToDetUnit(detId);
    bool zPositive = detUnit ? (detUnit->position().z() > 0) : true;
    return perLayerIndex(false, false, layer, zPositive);
  }

  static const char* perLayerCategoryName(int idx) {
    static const char* names[NPerLayerCategories] = {
        "B1_flat", "B1_tilted", "B2_flat", "B2_tilted", "B3_flat", "B3_tilted", "B4_flat", "B4_tilted",
        "B5_flat", "B5_tilted", "B6_flat", "B6_tilted", "FwdP_D1", "FwdP_D2",   "FwdP_D3", "FwdP_D4",
        "FwdP_D5", "FwdN_D1",   "FwdN_D2", "FwdN_D3",   "FwdN_D4", "FwdN_D5"};
    return names[idx];
  }

  static const char* perLayerCategoryTitle(int idx) {
    static const char* titles[NPerLayerCategories] = {
        "Barrel L1 Flat",   "Barrel L1 Tilted", "Barrel L2 Flat",   "Barrel L2 Tilted", "Barrel L3 Flat",
        "Barrel L3 Tilted", "Barrel L4 Flat",   "Barrel L4 Tilted", "Barrel L5 Flat",   "Barrel L5 Tilted",
        "Barrel L6 Flat",   "Barrel L6 Tilted", "Forward+ Disk 1",  "Forward+ Disk 2",  "Forward+ Disk 3",
        "Forward+ Disk 4",  "Forward+ Disk 5",  "Forward- Disk 1",  "Forward- Disk 2",  "Forward- Disk 3",
        "Forward- Disk 4",  "Forward- Disk 5"};
    return titles[idx];
  }

  static bool replicaDPhiDr(reco::StackedModuleGeometryConstView geom,
                            uint32_t iGeom,
                            bool posIsInner,
                            float xp,
                            float yp,
                            float zp,
                            float xq,
                            float yq,
                            float zq,
                            float& dPhiDr);

  std::string topFolder_;

  edm::EDGetTokenT<reco::VectorHitsHost> vectorHitsToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitCollectionToken_;
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<ClusterTPAssociation> clusterTPToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESGetToken<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> stackGeomToken_;

  double tpMinPt_;
  double tpMaxEta_;
  double tpMaxVtxZ_;
  double tpMaxD0_;
  double tpMaxLxy_;
  int tpMinNLayers_;

  MonitorElement* h_nVectorHits_;
  MonitorElement* h_vectorHitCategory_;
  MonitorElement* h_fakeRatePerEvent_;
  MonitorElement* h_vector_hit_rz_;

  MonitorElement* h_genuine_per_layer_;
  MonitorElement* h_total_per_layer_;
  MonitorElement* h_fake_per_layer_;

  MonitorElement* h_genuine_per_layer_cat_[NCategories];
  MonitorElement* h_total_per_layer_cat_[NCategories];
  MonitorElement* h_fake_per_layer_cat_[NCategories];

  MonitorElement* h_genuine_tp_pt_;

  MonitorElement* h_eff_denom_pt_barrel_;
  MonitorElement* h_eff_numer_pt_barrel_;
  MonitorElement* h_eff_denom_pt_endcap_;
  MonitorElement* h_eff_numer_pt_endcap_;
  MonitorElement* h_eff_denom_pt_all_;
  MonitorElement* h_eff_numer_pt_all_;

  MonitorElement* h_eff_denom_pt_cat_[NCategories];
  MonitorElement* h_eff_numer_pt_cat_[NCategories];

  MonitorElement* h_eff_denom_eta_barrel_;
  MonitorElement* h_eff_numer_eta_barrel_;
  MonitorElement* h_eff_denom_eta_endcap_;
  MonitorElement* h_eff_numer_eta_endcap_;
  MonitorElement* h_eff_denom_eta_all_;
  MonitorElement* h_eff_numer_eta_all_;

  MonitorElement* h_eff_denom_eta_cat_[NCategories];
  MonitorElement* h_eff_numer_eta_cat_[NCategories];

  MonitorElement* h_fake_denom_dPhiDr_barrel_;
  MonitorElement* h_fake_numer_dPhiDr_barrel_;
  MonitorElement* h_fake_denom_dPhiDr_endcap_;
  MonitorElement* h_fake_numer_dPhiDr_endcap_;
  MonitorElement* h_fake_denom_dPhiDr_all_;
  MonitorElement* h_fake_numer_dPhiDr_all_;

  MonitorElement* h_fake_denom_dPhiDr_cat_[NCategories];
  MonitorElement* h_fake_numer_dPhiDr_cat_[NCategories];

  MonitorElement* h_fake_denom_eta_barrel_;
  MonitorElement* h_fake_numer_eta_barrel_;
  MonitorElement* h_fake_denom_eta_endcap_;
  MonitorElement* h_fake_numer_eta_endcap_;
  MonitorElement* h_fake_denom_eta_all_;
  MonitorElement* h_fake_numer_eta_all_;

  MonitorElement* h_fake_denom_eta_cat_[NCategories];
  MonitorElement* h_fake_numer_eta_cat_[NCategories];

  MonitorElement* h_eff_numer_pt_perlayer_[NPerLayerCategories];
  MonitorElement* h_eff_denom_pt_perlayer_[NPerLayerCategories];
  MonitorElement* h_eff_numer_eta_perlayer_[NPerLayerCategories];
  MonitorElement* h_eff_denom_eta_perlayer_[NPerLayerCategories];

  MonitorElement* h_fake_numer_dPhiDr_perlayer_[NPerLayerCategories];
  MonitorElement* h_fake_denom_dPhiDr_perlayer_[NPerLayerCategories];
  MonitorElement* h_fake_numer_eta_perlayer_[NPerLayerCategories];
  MonitorElement* h_fake_denom_eta_perlayer_[NPerLayerCategories];

  MonitorElement* h_dPhiDr_res_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDr_pull_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDr_truthStatus_;
  MonitorElement* h_dPhiDr_replicaClosure_;

  MonitorElement* h_vector_hits_per_lowerhit_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_vector_hits_per_lowerhit_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_vector_hits_per_lowerhit_all_perlayer_[NPerLayerCategories];
  MonitorElement* h_mean_vector_hits_per_lowerhit_genuine_;
  MonitorElement* h_mean_vector_hits_per_lowerhit_fake_;
  MonitorElement* h_mean_vector_hits_per_lowerhit_all_;

  MonitorElement* h_width_sim_perlayer_[NPerLayerCategories];
  MonitorElement* h_width_sim_corr_perlayer_[NPerLayerCategories];
  MonitorElement* h_sim_pairs_per_layer_;
  MonitorElement* h_sim_pt_;
  MonitorElement* h_sim_eta_;

  MonitorElement* h_genuine_per_event_;
  MonitorElement* h_combinatoric_per_event_;
  MonitorElement* h_unknown_per_event_;
  MonitorElement* h_phitonly_per_event_;
  MonitorElement* h_genuine_matched_tp_pt_;
};

constexpr const char* Phase2OTValidateVectorHitsSoA::categoryNames_[];
constexpr const char* Phase2OTValidateVectorHitsSoA::categoryTitles_[];

Phase2OTValidateVectorHitsSoA::Phase2OTValidateVectorHitsSoA(const edm::ParameterSet& iConfig)
    : topFolder_(iConfig.getParameter<std::string>("TopFolderName")),
      vectorHitsToken_(consumes<reco::VectorHitsHost>(iConfig.getParameter<edm::InputTag>("vectorHitsSrc"))),
      otRecHitsToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"))),
      recHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("recHitSrc"))),
      tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      clusterTPToken_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("clusterTPAssociationSrc"))),
      beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      stackGeomToken_(esConsumes()),
      tpMinPt_(iConfig.getParameter<double>("TP_minPt")),
      tpMaxEta_(iConfig.getParameter<double>("TP_maxEta")),
      tpMaxVtxZ_(iConfig.getParameter<double>("TP_maxVtxZ")),
      tpMaxD0_(iConfig.getParameter<double>("TP_maxD0")),
      tpMaxLxy_(iConfig.getParameter<double>("TP_maxLxy")),
      tpMinNLayers_(iConfig.getParameter<int>("TP_minNLayers")) {
  for (const auto& tag : iConfig.getParameter<std::vector<edm::InputTag>>("simHitSrc")) {
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(tag));
  }
}

// Replica of the dPhiDr of the vector hit formation kernel (FormVectorHitsKernel in
// RecoTracker/PixelSeeding/plugins/alpaka/Phase2OTVectorHitsFormationKernels.h), evaluated on two given
// beam-spot corrected global points: P on the position sensor and Q on the other sensor of the stack iGeom.
// It is kept separate from the kernel on purpose, so it serves as an independent reference for the true
// dPhiDr of the pull; any change of the kernel formula has to be mirrored here. Returns false when the
// geometry leaves no bend measurement, where the kernel tags the entry as a non-vector-hit.
bool Phase2OTValidateVectorHitsSoA::replicaDPhiDr(reco::StackedModuleGeometryConstView geom,
                                                  uint32_t iGeom,
                                                  bool posIsInner,
                                                  float xp,
                                                  float yp,
                                                  float zp,
                                                  float xq,
                                                  float yq,
                                                  float zq,
                                                  float& dPhiDr) {
  const auto stack = geom[iGeom];
  const bool isFlipped = stack.isFlipped();
  const float nX = stack.globalLowUpNormX(), nY = stack.globalLowUpNormY(), nZ = stack.globalLowUpNormZ();
  const float separation_cm = stack.sensorSeparation() * 0.1f;
  auto const innerFrame = isFlipped ? stack.upperSensorFrame() : stack.lowerSensorFrame();
  auto const outerFrame = isFlipped ? stack.lowerSensorFrame() : stack.upperSensorFrame();
  auto const& rotOther = posIsInner ? outerFrame.rotation() : innerFrame.rotation();

  const float rp = std::sqrt(xp * xp + yp * yp);

  // strip-count bend: e_x(other) . (Q - t P), with t P the crossing of the ray through P with Q's plane
  const float nDotP = nX * xp + nY * yp + nZ * zp;
  const float nDotQ = nX * xq + nY * yq + nZ * zq;
  const float t = (std::abs(nDotP) > 1e-6f) ? nDotQ / nDotP : 1.0f;
  const float bendLocal = rotOther.xx() * (xq - t * xp) + rotOther.xy() * (yq - t * yp) + rotOther.xz() * (zq - t * zp);

  // lever arm
  const float rq = std::sqrt(xq * xq + yq * yq);
  const float drGeometric = posIsInner ? rq - rp : rp - rq;
  const float denominator = stack.cosTilt() + stack.sinTilt() * zp / rp;
  const float drEffective = (std::abs(denominator) > 1e-6f) ? separation_cm / denominator : drGeometric;
  if (std::abs(drEffective) <= 1e-6f) {
    return false;
  }

  // azimuth per unit local x at the projection, u_x = cos(alpha) / r_proj, and the inner -> outer sign
  const float rpInv = (rp > 1e-6f) ? 1.0f / rp : 0.0f;
  const float rHatX = xp * rpInv, rHatY = yp * rpInv;
  const float cosAlpha = -rHatY * rotOther.xx() + rHatX * rotOther.xy();
  const float rProj = t * rp;
  const float uxProj = (rProj > 1e-6f) ? cosAlpha / rProj : 0.0f;
  const float qSign = posIsInner ? 1.0f : -1.0f;
  dPhiDr = qSign * uxProj * bendLocal / drEffective;
  return true;
}

void Phase2OTValidateVectorHitsSoA::bookHistograms(DQMStore::IBooker& iBooker,
                                                   edm::Run const&,
                                                   edm::EventSetup const&) {
  iBooker.setCurrentFolder(topFolder_ + "/Summary");

  h_nVectorHits_ =
      iBooker.book1D("nVectorHits", "Number of vector hits per event;N vector hits;Events", 200, 0, 200000);
  h_vectorHitCategory_ =
      iBooker.book1D("vectorHitCategory", "Vector hit classification;Category;Vector hits", 4, -0.5, 3.5);
  h_vectorHitCategory_->setBinLabel(1, "Genuine");
  h_vectorHitCategory_->setBinLabel(2, "Combinatoric");
  h_vectorHitCategory_->setBinLabel(3, "Unknown");
  h_vectorHitCategory_->setBinLabel(4, "PHitOnly");
  h_fakeRatePerEvent_ = iBooker.book1D("fakeRatePerEvent", "Fake rate per event;Fake rate;Events", 100, 0, 1);
  h_vector_hit_rz_ = iBooker.book2D("vector_hit_rz", "Vector hit R-Z;z [cm];R [cm]", 600, -300, 300, 240, 0, 120);

  // per-layer bins: 6 barrel + 5 endcap
  h_genuine_per_layer_ =
      iBooker.book1D("genuine_per_layer", "Genuine vector hits per layer;Layer;Vector hits", 11, 0.5, 11.5);
  h_total_per_layer_ =
      iBooker.book1D("total_per_layer", "Total vector hits per layer;Layer;Vector hits", 11, 0.5, 11.5);
  h_fake_per_layer_ =
      iBooker.book1D("fake_per_layer", "Non-genuine vector hits per layer;Layer;Vector hits", 11, 0.5, 11.5);
  for (int i = 1; i <= 6; ++i) {
    std::string label = "B" + std::to_string(i);
    h_genuine_per_layer_->setBinLabel(i, label);
    h_total_per_layer_->setBinLabel(i, label);
    h_fake_per_layer_->setBinLabel(i, label);
  }
  for (int i = 1; i <= 5; ++i) {
    std::string label = "D" + std::to_string(i);
    h_genuine_per_layer_->setBinLabel(6 + i, label);
    h_total_per_layer_->setBinLabel(6 + i, label);
    h_fake_per_layer_->setBinLabel(6 + i, label);
  }

  // Per-layer per-category histograms
  for (int c = 0; c < NCategories; ++c) {
    int nLayers = nLayersForCategory(static_cast<ModuleCategory>(c));
    std::string prefix = (c <= BarrelTilted) ? "B" : "D";

    h_genuine_per_layer_cat_[c] =
        iBooker.book1D("genuine_per_layer_" + std::string(categoryNames_[c]),
                       "Genuine vector hits per layer (" + std::string(categoryTitles_[c]) + ");Layer;VectorHits",
                       nLayers,
                       0.5,
                       nLayers + 0.5);
    h_total_per_layer_cat_[c] =
        iBooker.book1D("total_per_layer_" + std::string(categoryNames_[c]),
                       "Total vector hits per layer (" + std::string(categoryTitles_[c]) + ");Layer;VectorHits",
                       nLayers,
                       0.5,
                       nLayers + 0.5);
    h_fake_per_layer_cat_[c] =
        iBooker.book1D("fake_per_layer_" + std::string(categoryNames_[c]),
                       "Non-genuine vector hits per layer (" + std::string(categoryTitles_[c]) + ");Layer;VectorHits",
                       nLayers,
                       0.5,
                       nLayers + 0.5);

    for (int i = 1; i <= nLayers; ++i) {
      std::string label = prefix + std::to_string(i);
      h_genuine_per_layer_cat_[c]->setBinLabel(i, label);
      h_total_per_layer_cat_[c]->setBinLabel(i, label);
      h_fake_per_layer_cat_[c]->setBinLabel(i, label);
    }
  }

  h_genuine_tp_pt_ =
      iBooker.book1D("genuine_tp_pt", "TP p_{T} for genuine vector hits;p_{T} [GeV];Vector hits", 100, 0, 100);

  h_genuine_per_event_ =
      iBooker.book1D("genuine_per_event", "Genuine vector hits per event;N genuine;Events", 200, 0, 100000);
  h_combinatoric_per_event_ = iBooker.book1D(
      "combinatoric_per_event", "Combinatoric vector hits per event;N combinatoric;Events", 200, 0, 100000);
  h_unknown_per_event_ =
      iBooker.book1D("unknown_per_event", "Unknown vector hits per event;N unknown;Events", 200, 0, 100000);
  h_phitonly_per_event_ =
      iBooker.book1D("phitonly_per_event", "PHitOnly vector hits per event;N PHitOnly;Events", 200, 0, 50000);
  h_genuine_matched_tp_pt_ = iBooker.book1D(
      "genuine_matched_tp_pt", "Matched TP p_{T} for genuine vector hits;p_{T} [GeV];Genuine vector hits", 200, 0, 100);

  iBooker.setCurrentFolder(topFolder_ + "/Efficiency/Ingredients");

  h_eff_denom_pt_barrel_ =
      iBooker.book1D("eff_denom_pt_barrel", "Efficiency denominator (barrel);p_{T} [GeV];RecHits", 100, 0, 50);
  h_eff_numer_pt_barrel_ =
      iBooker.book1D("eff_numer_pt_barrel", "Efficiency numerator (barrel);p_{T} [GeV];RecHits", 100, 0, 50);
  h_eff_denom_pt_endcap_ =
      iBooker.book1D("eff_denom_pt_endcap", "Efficiency denominator (endcap);p_{T} [GeV];RecHits", 100, 0, 50);
  h_eff_numer_pt_endcap_ =
      iBooker.book1D("eff_numer_pt_endcap", "Efficiency numerator (endcap);p_{T} [GeV];RecHits", 100, 0, 50);
  h_eff_denom_pt_all_ =
      iBooker.book1D("eff_denom_pt_all", "Efficiency denominator (all);p_{T} [GeV];RecHits", 100, 0, 50);
  h_eff_numer_pt_all_ =
      iBooker.book1D("eff_numer_pt_all", "Efficiency numerator (all);p_{T} [GeV];RecHits", 100, 0, 50);

  for (int c = 0; c < NCategories; ++c) {
    h_eff_denom_pt_cat_[c] =
        iBooker.book1D("eff_denom_pt_" + std::string(categoryNames_[c]),
                       "Efficiency denominator (" + std::string(categoryTitles_[c]) + ");p_{T} [GeV];RecHits",
                       100,
                       0,
                       50);
    h_eff_numer_pt_cat_[c] =
        iBooker.book1D("eff_numer_pt_" + std::string(categoryNames_[c]),
                       "Efficiency numerator (" + std::string(categoryTitles_[c]) + ");p_{T} [GeV];RecHits",
                       100,
                       0,
                       50);
  }

  h_eff_denom_eta_barrel_ =
      iBooker.book1D("eff_denom_eta_barrel", "Efficiency denominator (barrel);#eta;RecHits", 100, -4.2, 4.2);
  h_eff_numer_eta_barrel_ =
      iBooker.book1D("eff_numer_eta_barrel", "Efficiency numerator (barrel);#eta;RecHits", 100, -4.2, 4.2);
  h_eff_denom_eta_endcap_ =
      iBooker.book1D("eff_denom_eta_endcap", "Efficiency denominator (endcap);#eta;RecHits", 100, -4.2, 4.2);
  h_eff_numer_eta_endcap_ =
      iBooker.book1D("eff_numer_eta_endcap", "Efficiency numerator (endcap);#eta;RecHits", 100, -4.2, 4.2);
  h_eff_denom_eta_all_ =
      iBooker.book1D("eff_denom_eta_all", "Efficiency denominator (all);#eta;RecHits", 100, -4.2, 4.2);
  h_eff_numer_eta_all_ = iBooker.book1D("eff_numer_eta_all", "Efficiency numerator (all);#eta;RecHits", 100, -4.2, 4.2);

  for (int c = 0; c < NCategories; ++c) {
    h_eff_denom_eta_cat_[c] =
        iBooker.book1D("eff_denom_eta_" + std::string(categoryNames_[c]),
                       "Efficiency denominator (" + std::string(categoryTitles_[c]) + ");#eta;RecHits",
                       100,
                       -4.2,
                       4.2);
    h_eff_numer_eta_cat_[c] =
        iBooker.book1D("eff_numer_eta_" + std::string(categoryNames_[c]),
                       "Efficiency numerator (" + std::string(categoryTitles_[c]) + ");#eta;RecHits",
                       100,
                       -4.2,
                       4.2);
  }

  iBooker.setCurrentFolder(topFolder_ + "/Efficiency/PerLayer");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_eff_numer_pt_perlayer_[i] =
        iBooker.book1D("eff_numer_pt_" + name, "Efficiency numerator (" + title + ");p_{T} [GeV];RecHits", 100, 0, 50);
    h_eff_denom_pt_perlayer_[i] = iBooker.book1D(
        "eff_denom_pt_" + name, "Efficiency denominator (" + title + ");p_{T} [GeV];RecHits", 100, 0, 50);
    h_eff_numer_eta_perlayer_[i] =
        iBooker.book1D("eff_numer_eta_" + name, "Efficiency numerator (" + title + ");#eta;RecHits", 100, -4.2, 4.2);
    h_eff_denom_eta_perlayer_[i] =
        iBooker.book1D("eff_denom_eta_" + name, "Efficiency denominator (" + title + ");#eta;RecHits", 100, -4.2, 4.2);
  }

  iBooker.setCurrentFolder(topFolder_ + "/FakeRate/Ingredients");

  h_fake_denom_dPhiDr_barrel_ = iBooker.book1D(
      "fake_denom_dPhiDr_barrel", "Fake rate denominator (barrel);dPhiDr [rad/cm];Vector hits", 100, -0.01, 0.01);
  h_fake_numer_dPhiDr_barrel_ = iBooker.book1D(
      "fake_numer_dPhiDr_barrel", "Fake rate numerator (barrel);dPhiDr [rad/cm];Vector hits", 100, -0.01, 0.01);
  h_fake_denom_dPhiDr_endcap_ = iBooker.book1D(
      "fake_denom_dPhiDr_endcap", "Fake rate denominator (endcap);dPhiDr [rad/cm];Vector hits", 100, -0.01, 0.01);
  h_fake_numer_dPhiDr_endcap_ = iBooker.book1D(
      "fake_numer_dPhiDr_endcap", "Fake rate numerator (endcap);dPhiDr [rad/cm];Vector hits", 100, -0.01, 0.01);
  h_fake_denom_dPhiDr_all_ = iBooker.book1D(
      "fake_denom_dPhiDr_all", "Fake rate denominator (all);dPhiDr [rad/cm];Vector hits", 100, -0.01, 0.01);
  h_fake_numer_dPhiDr_all_ = iBooker.book1D(
      "fake_numer_dPhiDr_all", "Fake rate numerator (all);dPhiDr [rad/cm];Vector hits", 100, -0.01, 0.01);

  for (int c = 0; c < NCategories; ++c) {
    h_fake_denom_dPhiDr_cat_[c] =
        iBooker.book1D("fake_denom_dPhiDr_" + std::string(categoryNames_[c]),
                       "Fake rate denominator (" + std::string(categoryTitles_[c]) + ");dPhiDr [rad/cm];Vector hits",
                       100,
                       -0.01,
                       0.01);
    h_fake_numer_dPhiDr_cat_[c] =
        iBooker.book1D("fake_numer_dPhiDr_" + std::string(categoryNames_[c]),
                       "Fake rate numerator (" + std::string(categoryTitles_[c]) + ");dPhiDr [rad/cm];Vector hits",
                       100,
                       -0.01,
                       0.01);
  }

  h_fake_denom_eta_barrel_ =
      iBooker.book1D("fake_denom_eta_barrel", "Fake rate denominator (barrel);#eta;Vector hits", 100, -4.2, 4.2);
  h_fake_numer_eta_barrel_ =
      iBooker.book1D("fake_numer_eta_barrel", "Fake rate numerator (barrel);#eta;Vector hits", 100, -4.2, 4.2);
  h_fake_denom_eta_endcap_ =
      iBooker.book1D("fake_denom_eta_endcap", "Fake rate denominator (endcap);#eta;Vector hits", 100, -4.2, 4.2);
  h_fake_numer_eta_endcap_ =
      iBooker.book1D("fake_numer_eta_endcap", "Fake rate numerator (endcap);#eta;Vector hits", 100, -4.2, 4.2);
  h_fake_denom_eta_all_ =
      iBooker.book1D("fake_denom_eta_all", "Fake rate denominator (all);#eta;Vector hits", 100, -4.2, 4.2);
  h_fake_numer_eta_all_ =
      iBooker.book1D("fake_numer_eta_all", "Fake rate numerator (all);#eta;Vector hits", 100, -4.2, 4.2);

  for (int c = 0; c < NCategories; ++c) {
    h_fake_denom_eta_cat_[c] =
        iBooker.book1D("fake_denom_eta_" + std::string(categoryNames_[c]),
                       "Fake rate denominator (" + std::string(categoryTitles_[c]) + ");#eta;VectorHits",
                       100,
                       -4.2,
                       4.2);
    h_fake_numer_eta_cat_[c] =
        iBooker.book1D("fake_numer_eta_" + std::string(categoryNames_[c]),
                       "Fake rate numerator (" + std::string(categoryTitles_[c]) + ");#eta;VectorHits",
                       100,
                       -4.2,
                       4.2);
  }

  iBooker.setCurrentFolder(topFolder_ + "/FakeRate/PerLayer");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_fake_numer_dPhiDr_perlayer_[i] = iBooker.book1D("fake_numer_dPhiDr_" + name,
                                                      "Fake rate numerator (" + title + ");dPhiDr [rad/cm];Vector hits",
                                                      100,
                                                      -0.01,
                                                      0.01);
    h_fake_denom_dPhiDr_perlayer_[i] =
        iBooker.book1D("fake_denom_dPhiDr_" + name,
                       "Fake rate denominator (" + title + ");dPhiDr [rad/cm];Vector hits",
                       100,
                       -0.01,
                       0.01);
    h_fake_numer_eta_perlayer_[i] =
        iBooker.book1D("fake_numer_eta_" + name, "Fake rate numerator (" + title + ");#eta;VectorHits", 100, -4.2, 4.2);
    h_fake_denom_eta_perlayer_[i] = iBooker.book1D(
        "fake_denom_eta_" + name, "Fake rate denominator (" + title + ");#eta;VectorHits", 100, -4.2, 4.2);
  }

  iBooker.setCurrentFolder(topFolder_ + "/Diagnostics/dPhiDrPull");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_dPhiDr_res_perlayer_[i] =
        iBooker.book1D("dPhiDr_res_" + name,
                       "dPhiDr - dPhiDr_{true} genuine (" + title + ");dPhiDr - dPhiDr_{true} [rad/cm];Vector hits",
                       200,
                       -0.005,
                       0.005);
    h_dPhiDr_pull_perlayer_[i] =
        iBooker.book1D("dPhiDr_pull_" + name,
                       "dPhiDr pull genuine (" + title + ");(dPhiDr - dPhiDr_{true}) / #sigma_{dPhiDr};Vector hits",
                       200,
                       -10.0,
                       10.0);
  }
  h_dPhiDr_truthStatus_ = iBooker.book1D(
      "dPhiDr_truthStatus", "Truth for the dPhiDr pull (genuine vector hits);;Vector hits", 5, -0.5, 4.5);
  h_dPhiDr_truthStatus_->setBinLabel(1, "Used");
  h_dPhiDr_truthStatus_->setBinLabel(2, "No SimHit pair");
  h_dPhiDr_truthStatus_->setBinLabel(3, "No geometry");
  h_dPhiDr_truthStatus_->setBinLabel(4, "No bend");
  h_dPhiDr_truthStatus_->setBinLabel(5, "No SimHit collection");
  h_dPhiDr_replicaClosure_ = iBooker.book1D(
      "dPhiDr_replicaClosure",
      "Replica on the RecHits vs kernel (all vector hits);(dPhiDr_{replica} - dPhiDr) / #sigma_{dPhiDr};Vector hits",
      200,
      -0.01,
      0.01);

  iBooker.setCurrentFolder(topFolder_ + "/Diagnostics/VectorHitsPerLowerHit");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);
    h_vector_hits_per_lowerhit_genuine_perlayer_[i] = iBooker.book1D(
        "vector_hits_per_lowerhit_genuine_" + name,
        "Vector hits per lower-sensor hit (#geq1 genuine) (" + title + ");N vector hits;Lower-sensor hits",
        20,
        0.5,
        20.5);
    h_vector_hits_per_lowerhit_fake_perlayer_[i] =
        iBooker.book1D("vector_hits_per_lowerhit_fake_" + name,
                       "Vector hits per lower-sensor hit (all fake) (" + title + ");N vector hits;Lower-sensor hits",
                       20,
                       0.5,
                       20.5);
    h_vector_hits_per_lowerhit_all_perlayer_[i] =
        iBooker.book1D("vector_hits_per_lowerhit_all_" + name,
                       "Vector hits per lower-sensor hit (all) (" + title + ");N vector hits;Lower-sensor hits",
                       20,
                       0.5,
                       20.5);
  }

  h_mean_vector_hits_per_lowerhit_genuine_ = iBooker.bookProfile(
      "mean_vector_hits_per_lowerhit_genuine",
      "Mean vector hits per lower-sensor hit (#geq1 genuine) vs layer;Layer category;Mean N vector hits",
      NPerLayerCategories,
      -0.5,
      NPerLayerCategories - 0.5,
      0,
      100);
  h_mean_vector_hits_per_lowerhit_fake_ =
      iBooker.bookProfile("mean_vector_hits_per_lowerhit_fake",
                          "Mean vector hits per lower-sensor hit (all fake) vs layer;Layer category;Mean N vector hits",
                          NPerLayerCategories,
                          -0.5,
                          NPerLayerCategories - 0.5,
                          0,
                          100);
  h_mean_vector_hits_per_lowerhit_all_ =
      iBooker.bookProfile("mean_vector_hits_per_lowerhit_all",
                          "Mean vector hits per lower-sensor hit (all) vs layer;Layer category;Mean N vector hits",
                          NPerLayerCategories,
                          -0.5,
                          NPerLayerCategories - 0.5,
                          0,
                          100);
  for (int i = 0; i < NPerLayerCategories; ++i) {
    h_mean_vector_hits_per_lowerhit_genuine_->setBinLabel(i + 1, perLayerCategoryName(i));
    h_mean_vector_hits_per_lowerhit_fake_->setBinLabel(i + 1, perLayerCategoryName(i));
    h_mean_vector_hits_per_lowerhit_all_->setBinLabel(i + 1, perLayerCategoryName(i));
  }

  iBooker.setCurrentFolder(topFolder_ + "/SimTruth");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_width_sim_perlayer_[i] = iBooker.book1D(
        "width_sim_" + name, "|#Deltax_{local}| sim (" + title + ");|x_{lower} - x_{upper}| [cm];TP pairs", 200, 0, 0.3);
    h_width_sim_corr_perlayer_[i] = iBooker.book1D(
        "width_sim_corr_" + name, "|width| sim parallax-corr (" + title + ");|width_{corr}| [cm];TP pairs", 200, 0, 0.3);
  }

  h_sim_pairs_per_layer_ = iBooker.book1D(
      "sim_pairs_per_layer", "Simulated pairs able to form a vector hit per layer;Layer;TP pairs", 11, 0.5, 11.5);
  for (int i = 1; i <= 6; ++i) {
    h_sim_pairs_per_layer_->setBinLabel(i, "B" + std::to_string(i));
  }
  for (int i = 1; i <= 5; ++i) {
    h_sim_pairs_per_layer_->setBinLabel(6 + i, "D" + std::to_string(i));
  }

  h_sim_pt_ = iBooker.book1D("sim_pt", "TP p_{T} (with #geq1 OT pair);p_{T} [GeV];TPs", 100, 0, 50);
  h_sim_eta_ = iBooker.book1D("sim_eta", "TP #eta (with #geq1 OT pair);#eta;TPs", 100, -4.2, 4.2);
}

void Phase2OTValidateVectorHitsSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // getHandle() so that events where the HLT paths were not run are skipped
  const auto vectorHitsHandle = iEvent.getHandle(vectorHitsToken_);
  if (!vectorHitsHandle.isValid()) {
    return;
  }
  const auto otHitsHandle = iEvent.getHandle(otRecHitsToken_);
  if (!otHitsHandle.isValid()) {
    return;
  }
  const auto recHitsHandle = iEvent.getHandle(recHitCollectionToken_);
  if (!recHitsHandle.isValid()) {
    return;
  }
  const auto& vectorHits = *vectorHitsHandle;
  const auto& otHits = *otHitsHandle;
  const auto& recHits = *recHitsHandle;
  const auto clusterTPHandle = iEvent.getHandle(clusterTPToken_);
  if (!clusterTPHandle.isValid()) {
    return;
  }
  const auto& clusterTPs = *clusterTPHandle;
  const auto tpHandle = iEvent.getHandle(tpToken_);
  clusterTPs.checkMappedProductID(tpHandle);

  const auto& topo = iSetup.getData(topoToken_);
  const auto& trackerGeom = iSetup.getData(geomToken_);

  auto vectorHitsView = vectorHits.view().vectorHits();
  auto hitsView = otHits.view().otRecHits();
  uint32_t nVectorHits = vectorHits.nVectorHits();

  // TrackingParticles associated to the cluster of a RecHit, without duplicates. ClusterTPAssociation stores them in
  // descending key order; they are sorted here in the ascending order of edm::Ref, which std::set_intersection expects
  auto hitTPs = [&clusterTPs](const Phase2TrackerRecHit1D& recHit) {
    std::vector<TrackingParticleRef> tps;
    auto range = clusterTPs.equal_range(OmniClusterRef(recHit.cluster()));
    for (auto it = range.first; it != range.second; ++it) {
      tps.push_back(it->second);
    }
    std::sort(tps.begin(), tps.end());
    return tps;
  };

  // Inputs of the dPhiDr pull: the stacked-module geometry of the vector hit formation, indexed by stack DetId,
  // the beam spot subtracted by the OT RecHit converter, and the OT SimHits indexed by (sensor, SimTrack)
  auto const geomView = iSetup.getData(stackGeomToken_).const_view();
  std::unordered_map<uint32_t, uint32_t> stackDetIdToGeomIdx;
  for (int32_t iGeom = 0; iGeom < geomView.metadata().size(); ++iGeom) {
    stackDetIdToGeomIdx[geomView[iGeom].stackedDetId()] = iGeom;
  }

  const auto beamSpotHandle = iEvent.getHandle(beamSpotToken_);

  using SimHitKey = std::tuple<uint32_t, uint32_t, uint32_t>;  // (sensor DetId, SimTrack id, encoded event id)
  std::map<SimHitKey, std::vector<const PSimHit*>> sensorSimHits;
  unsigned int nSimHitCollections = 0;
  for (const auto& token : simHitTokens_) {
    const auto simHitHandle = iEvent.getHandle(token);
    if (!simHitHandle.isValid()) {
      continue;
    }
    ++nSimHitCollections;
    for (const auto& simHit : *simHitHandle) {
      sensorSimHits[{simHit.detUnitId(), simHit.trackId(), simHit.eventId().rawId()}].push_back(&simHit);
    }
  }

  // SimHit of the TrackingParticle on a sensor, the closest in local x to the RecHit if there are several
  auto findSimHit = [&sensorSimHits](const TrackingParticle& tp, uint32_t sensorDetId, float xLocalReco) {
    const PSimHit* best = nullptr;
    for (const auto& g4Track : tp.g4Tracks()) {
      auto it = sensorSimHits.find({sensorDetId, g4Track.trackId(), g4Track.eventId().rawId()});
      if (it == sensorSimHits.end()) {
        continue;
      }
      for (const PSimHit* simHit : it->second) {
        if (!best ||
            std::abs(simHit->localPosition().x() - xLocalReco) < std::abs(best->localPosition().x() - xLocalReco)) {
          best = simHit;
        }
      }
    }
    return best;
  };

  // Pass 0: flat vector of the original RecHit pointers
  std::vector<const Phase2TrackerRecHit1D*> origRecHits;
  for (const auto& detSet : recHits) {
    for (const auto& recHit : detSet) {
      origRecHits.push_back(&recHit);
    }
  }

  // reverse map: origRecHitIdx -> OTRecHitsSoA index
  uint32_t nOTHits = otHits.nHits();
  std::unordered_map<uint32_t, uint32_t> origIdxToSoAIdx;
  origIdxToSoAIdx.reserve(nOTHits);
  for (uint32_t iHit = 0; iHit < nOTHits; ++iHit) {
    origIdxToSoAIdx[hitsView[iHit].origRecHitIdx()] = iHit;
  }

  // Pass 1: classify each vector hit. A pair of hits on the two sensors is Genuine, Combinatoric or Unknown,
  // whether or not it has a bend measurement; a position-only entry (upper hit index UINT32_MAX) is PHitOnly
  enum Classification { Genuine = 0, Combinatoric = 1, Unknown = 2, PHitOnly = 3, BadHitIndex = 4 };

  struct VectorHitTruthInfo {
    Classification classification;
    TrackingParticleRef principalTP;
    // every TrackingParticle shared by the two hits of a Genuine pair, in ascending order
    std::vector<TrackingParticleRef> commonTPs;
  };

  std::vector<VectorHitTruthInfo> vectorHitTruth(nVectorHits);
  std::unordered_map<uint32_t, std::vector<uint32_t>> origHitToVectorHits;

  uint32_t nGenuine = 0, nCombinatoric = 0, nUnknown = 0, nPHitOnly = 0, nBadHitIndex = 0;

  for (uint32_t iVectorHit = 0; iVectorHit < nVectorHits; ++iVectorHit) {
    // An entry whose hit indices fall outside the OT RecHit SoA is skipped by every pass
    const uint32_t lowerHitIdx = vectorHitsView[iVectorHit].lowerHitIdx();
    const uint32_t upperHitIdx = vectorHitsView[iVectorHit].upperHitIdx();
    const bool isPositionOnly = (upperHitIdx == UINT32_MAX);
    if (lowerHitIdx >= nOTHits || (!isPositionOnly && upperHitIdx >= nOTHits) ||
        vectorHitsView[iVectorHit].posHitIdx() >= nOTHits) {
      vectorHitTruth[iVectorHit] = {BadHitIndex, TrackingParticleRef(), {}};
      nBadHitIndex++;
      continue;
    }

    if (isPositionOnly) {
      vectorHitTruth[iVectorHit] = {PHitOnly, TrackingParticleRef(), {}};
      nPHitOnly++;
      continue;
    }

    uint32_t innerSoAIdx = lowerHitIdx;
    uint32_t outerSoAIdx = upperHitIdx;

    uint32_t innerOrigIdx = hitsView[innerSoAIdx].origRecHitIdx();
    uint32_t outerOrigIdx = hitsView[outerSoAIdx].origRecHitIdx();

    origHitToVectorHits[innerOrigIdx].push_back(iVectorHit);
    origHitToVectorHits[outerOrigIdx].push_back(iVectorHit);

    std::vector<TrackingParticleRef> innerTPs, outerTPs;
    if (innerOrigIdx < origRecHits.size()) {
      innerTPs = hitTPs(*origRecHits[innerOrigIdx]);
    }
    if (outerOrigIdx < origRecHits.size()) {
      outerTPs = hitTPs(*origRecHits[outerOrigIdx]);
    }

    std::vector<TrackingParticleRef> commonTPs;
    std::set_intersection(
        innerTPs.begin(), innerTPs.end(), outerTPs.begin(), outerTPs.end(), std::back_inserter(commonTPs));

    if (innerTPs.empty() && outerTPs.empty()) {
      vectorHitTruth[iVectorHit] = {Unknown, TrackingParticleRef(), {}};
      nUnknown++;
    } else if (!commonTPs.empty()) {
      TrackingParticleRef principalTP = commonTPs.front();
      vectorHitTruth[iVectorHit] = {Genuine, principalTP, std::move(commonTPs)};
      nGenuine++;
    } else {
      vectorHitTruth[iVectorHit] = {Combinatoric, TrackingParticleRef(), {}};
      nCombinatoric++;
    }
  }

  h_nVectorHits_->Fill(nVectorHits);
  h_vectorHitCategory_->Fill(0.0, nGenuine);
  h_vectorHitCategory_->Fill(1.0, nCombinatoric);
  h_vectorHitCategory_->Fill(2.0, nUnknown);
  h_vectorHitCategory_->Fill(3.0, nPHitOnly);

  uint32_t nClassified = nGenuine + nCombinatoric + nUnknown;
  if (nClassified > 0) {
    h_fakeRatePerEvent_->Fill(float(nCombinatoric + nUnknown) / float(nClassified));
  }

  h_genuine_per_event_->Fill(nGenuine);
  h_combinatoric_per_event_->Fill(nCombinatoric);
  h_unknown_per_event_->Fill(nUnknown);
  h_phitonly_per_event_->Fill(nPHitOnly);

  if (nBadHitIndex > 0) {
    edm::LogWarning("Phase2OTValidateVectorHitsSoA")
        << "Event " << iEvent.id().event() << ": " << nBadHitIndex << " of " << nVectorHits
        << " vector hits have a hit index outside the " << nOTHits << " OT RecHits and are skipped";
  }

  for (uint32_t iVectorHit = 0; iVectorHit < nVectorHits; ++iVectorHit) {
    if (vectorHitTruth[iVectorHit].classification == BadHitIndex) {
      continue;
    }
    uint32_t innerSoAIdx = vectorHitsView[iVectorHit].lowerHitIdx();
    float x = hitsView[innerSoAIdx].xGlobal();
    float y = hitsView[innerSoAIdx].yGlobal();
    float z = hitsView[innerSoAIdx].zGlobal();
    float r = std::sqrt(x * x + y * y);
    h_vector_hit_rz_->Fill(z, r);

    if (vectorHitTruth[iVectorHit].classification == PHitOnly) {
      continue;
    }

    uint8_t flags = vectorHitsView[iVectorHit].flags();
    bool isBarrel = reco::VectorHitFlags::isBarrel(flags);
    uint8_t layer = reco::VectorHitFlags::layer(flags);

    int layerBin = isBarrel ? layer : layer + 6;
    h_total_per_layer_->Fill(layerBin);

    ModuleCategory cat = moduleCategory(flags, z);
    h_total_per_layer_cat_[cat]->Fill(layer);

    if (vectorHitTruth[iVectorHit].classification == Genuine) {
      h_genuine_per_layer_->Fill(layerBin);
      h_genuine_per_layer_cat_[cat]->Fill(layer);

      auto const& tp = vectorHitTruth[iVectorHit].principalTP;
      if (tp.isNonnull()) {
        h_genuine_tp_pt_->Fill(tp->pt());
        h_genuine_matched_tp_pt_->Fill(tp->pt());
      }
    }

    if (vectorHitTruth[iVectorHit].classification == Combinatoric ||
        vectorHitTruth[iVectorHit].classification == Unknown) {
      h_fake_per_layer_->Fill(layerBin);
      h_fake_per_layer_cat_[cat]->Fill(layer);
    }

    {
      // stack of the vector hit, in the stacked-module geometry of the vector hit formation
      const uint32_t lowerSoA = vectorHitsView[iVectorHit].lowerHitIdx();
      const uint32_t upperSoA = vectorHitsView[iVectorHit].upperHitIdx();
      const uint32_t posSoA = vectorHitsView[iVectorHit].posHitIdx();
      const uint32_t othSoA = (posSoA == lowerSoA) ? upperSoA : lowerSoA;
      const uint32_t posDetId = hitsView[posSoA].sensorDetId();
      const uint32_t othDetId = hitsView[othSoA].sensorDetId();
      auto geomIt = stackDetIdToGeomIdx.find(topo.stack(DetId(posDetId)));
      const bool hasGeom = (geomIt != stackDetIdToGeomIdx.end());
      const uint32_t iGeom = hasGeom ? geomIt->second : 0;
      const bool posIsInner = hasGeom && (posSoA == (geomView[iGeom].isFlipped() ? upperSoA : lowerSoA));

      float dPhiDr = vectorHitsView[iVectorHit].dPhiDr();
      float dPhiDrErr = vectorHitsView[iVectorHit].dPhiDrError();

      // closure of the replica, on every vector hit: on the RecHits it must give back the kernel's dPhiDr
      float dPhiDrReplica = 0.f;
      if (hasGeom && dPhiDrErr > 0.f &&
          replicaDPhiDr(geomView,
                        iGeom,
                        posIsInner,
                        hitsView[posSoA].xGlobal(),
                        hitsView[posSoA].yGlobal(),
                        hitsView[posSoA].zGlobal(),
                        hitsView[othSoA].xGlobal(),
                        hitsView[othSoA].yGlobal(),
                        hitsView[othSoA].zGlobal(),
                        dPhiDrReplica)) {
        h_dPhiDr_replicaClosure_->Fill((dPhiDrReplica - dPhiDr) / dPhiDrErr);
      }

      int plIdx = perLayerIndexFromVectorHit(flags, z);
      if (plIdx >= 0 && plIdx < NPerLayerCategories) {
        if (vectorHitTruth[iVectorHit].classification == Genuine) {
          // pull against the replica of the kernel formula evaluated on the TP's SimHits
          if (beamSpotHandle.isValid()) {
            const PSimHit* posSim =
                findSimHit(*vectorHitTruth[iVectorHit].principalTP, posDetId, hitsView[posSoA].xLocal());
            const PSimHit* othSim =
                findSimHit(*vectorHitTruth[iVectorHit].principalTP, othDetId, hitsView[othSoA].xLocal());
            if (!hasGeom) {
              h_dPhiDr_truthStatus_->Fill(2);
            } else if (nSimHitCollections == 0) {
              h_dPhiDr_truthStatus_->Fill(4);
            } else if (!posSim || !othSim) {
              h_dPhiDr_truthStatus_->Fill(1);
            } else if (!isVectorHit(vectorHitsView, iVectorHit)) {
              // a pair without a bend measurement
              h_dPhiDr_truthStatus_->Fill(3);
            } else {
              const auto& bs = *beamSpotHandle;
              const GlobalPoint posTrue =
                  trackerGeom.idToDetUnit(DetId(posDetId))->surface().toGlobal(posSim->localPosition());
              const GlobalPoint othTrue =
                  trackerGeom.idToDetUnit(DetId(othDetId))->surface().toGlobal(othSim->localPosition());
              float dPhiDrTrue = 0.f;
              if (!replicaDPhiDr(geomView,
                                 iGeom,
                                 posIsInner,
                                 posTrue.x() - bs.x0(),
                                 posTrue.y() - bs.y0(),
                                 posTrue.z() - bs.z0(),
                                 othTrue.x() - bs.x0(),
                                 othTrue.y() - bs.y0(),
                                 othTrue.z() - bs.z0(),
                                 dPhiDrTrue)) {
                h_dPhiDr_truthStatus_->Fill(3);
              } else {
                h_dPhiDr_truthStatus_->Fill(0);
                h_dPhiDr_res_perlayer_[plIdx]->Fill(dPhiDr - dPhiDrTrue);
                if (dPhiDrErr > 0.f) {
                  h_dPhiDr_pull_perlayer_[plIdx]->Fill((dPhiDr - dPhiDrTrue) / dPhiDrErr);
                }
              }
            }
          }
        }
      }
    }

    // fake rate in vector-hit-level quantities: eta of the reference hit seen from the origin, and dPhiDr
    {
      float vector_hit_eta = std::asinh(z / r);
      float vectorHit_dPhiDr = vectorHitsView[iVectorHit].dPhiDr();

      int plIdx = perLayerIndexFromVectorHit(flags, z);
      bool validPlIdx = (plIdx >= 0 && plIdx < NPerLayerCategories);

      if (isBarrel) {
        h_fake_denom_dPhiDr_barrel_->Fill(vectorHit_dPhiDr);
        h_fake_denom_eta_barrel_->Fill(vector_hit_eta);
      } else {
        h_fake_denom_dPhiDr_endcap_->Fill(vectorHit_dPhiDr);
        h_fake_denom_eta_endcap_->Fill(vector_hit_eta);
      }
      h_fake_denom_dPhiDr_all_->Fill(vectorHit_dPhiDr);
      h_fake_denom_eta_all_->Fill(vector_hit_eta);
      h_fake_denom_dPhiDr_cat_[cat]->Fill(vectorHit_dPhiDr);
      h_fake_denom_eta_cat_[cat]->Fill(vector_hit_eta);
      if (validPlIdx) {
        h_fake_denom_dPhiDr_perlayer_[plIdx]->Fill(vectorHit_dPhiDr);
        h_fake_denom_eta_perlayer_[plIdx]->Fill(vector_hit_eta);
      }

      if (vectorHitTruth[iVectorHit].classification == Genuine) {
        if (isBarrel) {
          h_fake_numer_dPhiDr_barrel_->Fill(vectorHit_dPhiDr);
          h_fake_numer_eta_barrel_->Fill(vector_hit_eta);
        } else {
          h_fake_numer_dPhiDr_endcap_->Fill(vectorHit_dPhiDr);
          h_fake_numer_eta_endcap_->Fill(vector_hit_eta);
        }
        h_fake_numer_dPhiDr_all_->Fill(vectorHit_dPhiDr);
        h_fake_numer_eta_all_->Fill(vector_hit_eta);
        h_fake_numer_dPhiDr_cat_[cat]->Fill(vectorHit_dPhiDr);
        h_fake_numer_eta_cat_[cat]->Fill(vector_hit_eta);
        if (validPlIdx) {
          h_fake_numer_dPhiDr_perlayer_[plIdx]->Fill(vectorHit_dPhiDr);
          h_fake_numer_eta_perlayer_[plIdx]->Fill(vector_hit_eta);
        }
      }
    }
  }

  {
    std::unordered_map<uint32_t, std::vector<uint32_t>> lowerHitToVectorHits;
    for (uint32_t iVectorHit = 0; iVectorHit < nVectorHits; ++iVectorHit) {
      if (vectorHitTruth[iVectorHit].classification == PHitOnly ||
          vectorHitTruth[iVectorHit].classification == BadHitIndex) {
        continue;
      }
      uint32_t groupId = vectorHitsView[iVectorHit].lowerHitIdx();
      lowerHitToVectorHits[groupId].push_back(iVectorHit);
    }

    for (const auto& [groupId, vectorHitIndices] : lowerHitToVectorHits) {
      int nVectorHitsForLowerHit = static_cast<int>(vectorHitIndices.size());
      bool hasGenuine = false;
      uint8_t firstFlags = vectorHitsView[vectorHitIndices[0]].flags();
      float firstZ = hitsView[groupId].zGlobal();
      int plIdx = perLayerIndexFromVectorHit(firstFlags, firstZ);
      if (plIdx < 0 || plIdx >= NPerLayerCategories) {
        continue;
      }

      for (uint32_t idx : vectorHitIndices) {
        if (vectorHitTruth[idx].classification == Genuine) {
          hasGenuine = true;
          break;
        }
      }

      h_vector_hits_per_lowerhit_all_perlayer_[plIdx]->Fill(nVectorHitsForLowerHit);
      h_mean_vector_hits_per_lowerhit_all_->Fill(plIdx, nVectorHitsForLowerHit);
      if (hasGenuine) {
        h_vector_hits_per_lowerhit_genuine_perlayer_[plIdx]->Fill(nVectorHitsForLowerHit);
        h_mean_vector_hits_per_lowerhit_genuine_->Fill(plIdx, nVectorHitsForLowerHit);
      } else {
        h_vector_hits_per_lowerhit_fake_perlayer_[plIdx]->Fill(nVectorHitsForLowerHit);
        h_mean_vector_hits_per_lowerhit_fake_->Fill(plIdx, nVectorHitsForLowerHit);
      }
    }
  }

  // Pass 2: efficiency
  struct TPHitInfo {
    uint32_t origRecHitIdx;
    int layerId;  // OT layer: barrel layer 1-6, disk 7-11 at z > 0, -7 to -11 at z < 0
    bool isBarrel;
    ModuleCategory category;
    int perLayerIdx;
    float eta;  // of the hit position seen from the origin, as for the fake rate
  };
  std::map<TrackingParticleRef, std::vector<TPHitInfo>> tpToLowerHits;

  struct TPSimHitInfo {
    uint32_t origRecHitIdx;
    uint32_t stackDetId;
    bool isLowerSensor;
    int perLayerIdx;
    int layerBin;
  };
  std::map<TrackingParticleRef, std::vector<TPSimHitInfo>> tpToOTHits;

  uint32_t flatIdx = 0;
  for (const auto& detSet : recHits) {
    DetId detId(detSet.detId());
    uint32_t stackId = topo.stack(detId);
    bool isOnStack = (stackId != 0);
    bool isLower = isOnStack && topo.isLower(detId);
    bool isBarrel = (detId.subdetId() == StripSubdetector::TOB);
    ModuleCategory cat = BarrelFlat;
    int plIdx = 0;
    int layerBin = 0;
    int layerId = 0;
    if (isOnStack) {
      cat = moduleCategoryFromDetId(detId, topo, trackerGeom);
      plIdx = perLayerIndexFromDetId(detId, topo, trackerGeom);
      int layerNum = topo.layer(detId);
      layerBin = isBarrel ? layerNum : layerNum + 6;
      layerId = (isBarrel || topo.side(detId) == 2) ? layerBin : -layerBin;
    }

    for (const auto& recHit : detSet) {
      if (isOnStack) {
        float hitEta = 0.f;
        if (isLower) {
          auto soaIt = origIdxToSoAIdx.find(flatIdx);
          if (soaIt != origIdxToSoAIdx.end()) {
            const auto hit = hitsView[soaIt->second];
            float hitR = std::sqrt(hit.xGlobal() * hit.xGlobal() + hit.yGlobal() * hit.yGlobal());
            hitEta = std::asinh(hit.zGlobal() / hitR);
          } else {
            hitEta = trackerGeom.idToDet(detId)->surface().toGlobal(recHit.localPosition()).eta();
          }
        }
        for (const auto& tp : hitTPs(recHit)) {
          if (isLower) {
            tpToLowerHits[tp].push_back({flatIdx, layerId, isBarrel, cat, plIdx, hitEta});
          }
          tpToOTHits[tp].push_back({flatIdx, stackId, isLower, plIdx, layerBin});
        }
      }
      flatIdx++;
    }
  }

  // TP selection shared by the efficiency and the simulation-truth passes; it requires lower-sensor RecHits of the TP
  // on at least TP_minNLayers distinct OT layers
  auto isSelectedTP = [this](const TrackingParticle& tp, const std::vector<TPHitInfo>& lowerHits) {
    if (tp.charge() == 0) {
      return false;
    }
    if (tp.pt() < tpMinPt_) {
      return false;
    }
    if (std::abs(tp.eta()) > tpMaxEta_) {
      return false;
    }
    if (std::abs(tp.vz()) > tpMaxVtxZ_) {
      return false;
    }
    if (std::abs(tp.d0()) > tpMaxD0_) {
      return false;
    }
    if (std::sqrt(tp.vx() * tp.vx() + tp.vy() * tp.vy()) > tpMaxLxy_) {
      return false;
    }
    std::vector<int> layerIds;
    layerIds.reserve(lowerHits.size());
    for (const auto& hitInfo : lowerHits) {
      layerIds.push_back(hitInfo.layerId);
    }
    std::sort(layerIds.begin(), layerIds.end());
    return std::unique(layerIds.begin(), layerIds.end()) - layerIds.begin() >= tpMinNLayers_;
  };

  for (const auto& [tp, lowerHits] : tpToLowerHits) {
    if (!isSelectedTP(*tp, lowerHits)) {
      continue;
    }

    float tp_pt = tp->pt();

    for (const auto& hitInfo : lowerHits) {
      if (hitInfo.isBarrel) {
        h_eff_denom_pt_barrel_->Fill(tp_pt);
        h_eff_denom_eta_barrel_->Fill(hitInfo.eta);
      } else {
        h_eff_denom_pt_endcap_->Fill(tp_pt);
        h_eff_denom_eta_endcap_->Fill(hitInfo.eta);
      }
      h_eff_denom_pt_all_->Fill(tp_pt);
      h_eff_denom_eta_all_->Fill(hitInfo.eta);
      h_eff_denom_pt_cat_[hitInfo.category]->Fill(tp_pt);
      h_eff_denom_eta_cat_[hitInfo.category]->Fill(hitInfo.eta);

      if (hitInfo.perLayerIdx >= 0 && hitInfo.perLayerIdx < NPerLayerCategories) {
        h_eff_denom_pt_perlayer_[hitInfo.perLayerIdx]->Fill(tp_pt);
        h_eff_denom_eta_perlayer_[hitInfo.perLayerIdx]->Fill(hitInfo.eta);
      }

      bool hasGenuineVectorHit = false;
      auto vectorHitIt = origHitToVectorHits.find(hitInfo.origRecHitIdx);
      if (vectorHitIt != origHitToVectorHits.end()) {
        for (uint32_t vectorHitIdx : vectorHitIt->second) {
          auto const& commonTPs = vectorHitTruth[vectorHitIdx].commonTPs;
          if (vectorHitTruth[vectorHitIdx].classification == Genuine &&
              std::binary_search(commonTPs.begin(), commonTPs.end(), tp)) {
            hasGenuineVectorHit = true;
            break;
          }
        }
      }

      if (hasGenuineVectorHit) {
        if (hitInfo.isBarrel) {
          h_eff_numer_pt_barrel_->Fill(tp_pt);
          h_eff_numer_eta_barrel_->Fill(hitInfo.eta);
        } else {
          h_eff_numer_pt_endcap_->Fill(tp_pt);
          h_eff_numer_eta_endcap_->Fill(hitInfo.eta);
        }
        h_eff_numer_pt_all_->Fill(tp_pt);
        h_eff_numer_eta_all_->Fill(hitInfo.eta);
        h_eff_numer_pt_cat_[hitInfo.category]->Fill(tp_pt);
        h_eff_numer_eta_cat_[hitInfo.category]->Fill(hitInfo.eta);

        if (hitInfo.perLayerIdx >= 0 && hitInfo.perLayerIdx < NPerLayerCategories) {
          h_eff_numer_pt_perlayer_[hitInfo.perLayerIdx]->Fill(tp_pt);
          h_eff_numer_eta_perlayer_[hitInfo.perLayerIdx]->Fill(hitInfo.eta);
        }
      }
    }
  }

  // Pass 3: simulation-truth diagnostics
  for (const auto& [tp, tpOTHits] : tpToOTHits) {
    auto lowerIt = tpToLowerHits.find(tp);
    if (lowerIt == tpToLowerHits.end() || !isSelectedTP(*tp, lowerIt->second)) {
      continue;
    }

    struct StackHitGroup {
      std::vector<uint32_t> lowerOrigIdx;
      std::vector<uint32_t> upperOrigIdx;
      int perLayerIdx;
      int layerBin;
    };
    std::map<uint32_t, StackHitGroup> stackGroups;
    for (const auto& simHit : tpOTHits) {
      auto& group = stackGroups[simHit.stackDetId];
      group.perLayerIdx = simHit.perLayerIdx;
      group.layerBin = simHit.layerBin;
      if (simHit.isLowerSensor) {
        group.lowerOrigIdx.push_back(simHit.origRecHitIdx);
      } else {
        group.upperOrigIdx.push_back(simHit.origRecHitIdx);
      }
    }

    bool hasSimPair = false;

    for (const auto& [stkId, group] : stackGroups) {
      if (group.lowerOrigIdx.empty() || group.upperOrigIdx.empty()) {
        continue;
      }

      for (uint32_t lowerOrig : group.lowerOrigIdx) {
        auto lowerSoAIt = origIdxToSoAIdx.find(lowerOrig);
        if (lowerSoAIt == origIdxToSoAIdx.end()) {
          continue;
        }
        uint32_t lowerSoA = lowerSoAIt->second;

        for (uint32_t upperOrig : group.upperOrigIdx) {
          auto upperSoAIt = origIdxToSoAIdx.find(upperOrig);
          if (upperSoAIt == origIdxToSoAIdx.end()) {
            continue;
          }
          uint32_t upperSoA = upperSoAIt->second;

          float dx = std::abs(hitsView[lowerSoA].xLocal() - hitsView[upperSoA].xLocal());

          float dx_corr = dx;
          {
            DetId lowerDetId(hitsView[lowerSoA].sensorDetId());
            DetId upperDetId(hitsView[upperSoA].sensorDetId());
            auto const* lowerDet = trackerGeom.idToDetUnit(lowerDetId);
            auto const* upperDet = trackerGeom.idToDetUnit(upperDetId);
            if (lowerDet && upperDet) {
              float gx = hitsView[lowerSoA].xGlobal();
              float gy = hitsView[lowerSoA].yGlobal();
              float gz = hitsView[lowerSoA].zGlobal();
              GlobalVector originToHit(gx, gy, gz);
              LocalVector localDir = lowerDet->toLocal(originToHit);
              LocalPoint upperInLower = lowerDet->toLocal(upperDet->position());
              float sepZ = std::abs(upperInLower.z());
              float pC_abs = 0.f;
              if (std::abs(localDir.z()) > 1e-6f) {
                pC_abs = std::abs(localDir.x() / localDir.z()) * sepZ;
                if (!std::isfinite(pC_abs)) {
                  pC_abs = 0.f;
                }
              }
              dx_corr = std::max(0.f, dx - pC_abs);
            }
          }

          int plIdx = group.perLayerIdx;
          if (plIdx >= 0 && plIdx < NPerLayerCategories) {
            h_width_sim_perlayer_[plIdx]->Fill(dx);
            h_width_sim_corr_perlayer_[plIdx]->Fill(dx_corr);
          }
          h_sim_pairs_per_layer_->Fill(group.layerBin);
          hasSimPair = true;
        }
      }
    }

    if (hasSimPair) {
      h_sim_pt_->Fill(tp->pt());
      h_sim_eta_->Fill(tp->eta());
    }
  }

  edm::LogInfo("Phase2OTValidateVectorHitsSoA")
      << "Event " << iEvent.id().event() << ": nVectorHits=" << nVectorHits << " genuine=" << nGenuine << " ("
      << (nClassified > 0 ? 100.0f * nGenuine / nClassified : 0.f) << "%)"
      << " combinatoric=" << nCombinatoric << " unknown=" << nUnknown << " PHitOnly=" << nPHitOnly;
}

void Phase2OTValidateVectorHitsSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("TopFolderName", "HLT/TrackerPhase2OTVectorHitsSoAV");

  desc.add<edm::InputTag>("vectorHitsSrc", edm::InputTag("phase2OTVectorHitsSoA"));
  desc.add<edm::InputTag>("otRecHitsSrc", edm::InputTag("pixelSeedingOTRecHitsSoA"));
  desc.add<edm::InputTag>("recHitSrc", edm::InputTag("siPhase2RecHits"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));

  desc.add<double>("TP_minPt", 0.75);
  desc.add<double>("TP_maxEta", 4.2);
  desc.add<double>("TP_maxVtxZ", 60.0);
  desc.add<double>("TP_maxD0", 5.0);
  desc.add<double>("TP_maxLxy", 60.0);
  desc.add<int>("TP_minNLayers", 4)->setComment("Minimum number of OT layers with a lower-sensor RecHit of the TP");

  desc.add<edm::InputTag>("clusterTPAssociationSrc", edm::InputTag("tpClusterProducer"));
  // beam spot subtracted from the OT RecHit global positions by the SoA converter
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  // Outer Tracker SimHits, for the true dPhiDr of the pull. The Phase-2 simulation writes the SimHits of the
  // whole tracker, Outer Tracker included, into the TrackerHitsPixel* collections; the TOB/TID ones stay empty.
  desc.add<std::vector<edm::InputTag>>("simHitSrc",
                                       {edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof")});

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Phase2OTValidateVectorHitsSoA);
