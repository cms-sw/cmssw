// DQM validation of VectorHitStyle stubs: fake rate (stubs that are combinatoric or unknown),
// efficiency (TrackingParticle lower-sensor RecHits with a genuine stub), position residuals and
// per-layer diagnostics, broken down into barrel-flat, barrel-tilted, forward+ and forward- modules.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
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
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

class Phase2OTValidateVectorHitStyleStub : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateVectorHitStyleStub(const edm::ParameterSet& iConfig);
  ~Phase2OTValidateVectorHitStyleStub() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  enum ModuleCategory { BarrelFlat = 0, BarrelTilted = 1, ForwardPos = 2, ForwardNeg = 3, NCategories = 4 };
  static constexpr const char* categoryNames_[NCategories] = {"barrel_flat", "barrel_tilted", "fwd_pos", "fwd_neg"};
  static constexpr const char* categoryTitles_[NCategories] = {"Barrel Flat", "Barrel Tilted", "Forward+", "Forward-"};

  ModuleCategory moduleCategory(uint8_t flags, float z) const {
    if (reco::StubFlags::isBarrel(flags)) {
      return reco::StubFlags::isFlat(flags) ? BarrelFlat : BarrelTilted;
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
      if (layer < 1 || layer > 6)
        return -1;
      return (layer - 1) * 2 + (isFlat ? 0 : 1);
    }
    if (layer < 1 || layer > 5)
      return -1;
    if (zPositive) {
      return 12 + (layer - 1);
    }
    return 17 + (layer - 1);
  }

  static int perLayerIndexFromStub(uint8_t flags, float z) {
    bool barrel = reco::StubFlags::isBarrel(flags);
    bool flat = reco::StubFlags::isFlat(flags);
    int layer = reco::StubFlags::layer(flags);
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

  std::string topFolder_;

  edm::EDGetTokenT<reco::StubsHost> stubsToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitCollectionToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> tpToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  TrackerHitAssociator::Config hitAssocConfig_;

  double tpMinPt_;
  double tpMaxEta_;
  double tpMaxVtxZ_;
  double tpMaxD0_;
  double tpMaxLxy_;
  int tpMinNLayers_;

  MonitorElement* h_nStubs_;
  MonitorElement* h_stubCategory_;
  MonitorElement* h_fakeRatePerEvent_;
  MonitorElement* h_stub_rz_;

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

  MonitorElement* h_fake_denom_pt_barrel_;
  MonitorElement* h_fake_numer_pt_barrel_;
  MonitorElement* h_fake_denom_pt_endcap_;
  MonitorElement* h_fake_numer_pt_endcap_;
  MonitorElement* h_fake_denom_pt_all_;
  MonitorElement* h_fake_numer_pt_all_;

  MonitorElement* h_fake_denom_pt_cat_[NCategories];
  MonitorElement* h_fake_numer_pt_cat_[NCategories];

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

  MonitorElement* h_fake_numer_pt_perlayer_[NPerLayerCategories];
  MonitorElement* h_fake_denom_pt_perlayer_[NPerLayerCategories];
  MonitorElement* h_fake_numer_eta_perlayer_[NPerLayerCategories];
  MonitorElement* h_fake_denom_eta_perlayer_[NPerLayerCategories];

  MonitorElement* h_phi_res_barrel_;
  MonitorElement* h_phi_res_endcap_;
  MonitorElement* h_z_res_barrel_;
  MonitorElement* h_r_res_endcap_;

  MonitorElement* h_dy_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_dy_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_dypull_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_dypull_fake_perlayer_[NPerLayerCategories];

  MonitorElement* h_dPhiDr_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDr_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDrErr_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDrErr_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDr_signif_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_dPhiDr_signif_fake_perlayer_[NPerLayerCategories];

  MonitorElement* h_clusterSize_lower_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSize_lower_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSize_upper_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSize_upper_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeDiff_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeDiff_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSize_2d_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSize_2d_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeSum_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeSum_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeAsymmetry_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeAsymmetry_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeMin_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_clusterSizeMin_fake_perlayer_[NPerLayerCategories];

  MonitorElement* h_stubs_per_phit_genuine_;
  MonitorElement* h_stubs_per_phit_fake_;
  static constexpr int NBarrelLayers = 6;
  MonitorElement* h_stubs_per_phit_genuine_barrel_[NBarrelLayers];
  MonitorElement* h_stubs_per_phit_fake_barrel_[NBarrelLayers];
  MonitorElement* h_stubs_per_phit_genuine_perlayer_[NPerLayerCategories];
  MonitorElement* h_stubs_per_phit_fake_perlayer_[NPerLayerCategories];
  MonitorElement* h_stubs_per_phit_all_perlayer_[NPerLayerCategories];
  MonitorElement* h_mean_stubs_per_phit_genuine_;
  MonitorElement* h_mean_stubs_per_phit_fake_;
  MonitorElement* h_mean_stubs_per_phit_all_;

  MonitorElement* h_width_sim_perlayer_[NPerLayerCategories];
  MonitorElement* h_width_sim_corr_perlayer_[NPerLayerCategories];
  MonitorElement* h_dy_sim_perlayer_[NPerLayerCategories];
  MonitorElement* h_dypull_sim_perlayer_[NPerLayerCategories];
  MonitorElement* h_sim_pairs_per_layer_;
  MonitorElement* h_sim_pt_;
  MonitorElement* h_sim_eta_;

  MonitorElement* h_genuine_per_event_;
  MonitorElement* h_combinatoric_per_event_;
  MonitorElement* h_unknown_per_event_;
  MonitorElement* h_phitonly_per_event_;
  MonitorElement* h_genuine_matched_tp_pt_;
};

constexpr const char* Phase2OTValidateVectorHitStyleStub::categoryNames_[];
constexpr const char* Phase2OTValidateVectorHitStyleStub::categoryTitles_[];

Phase2OTValidateVectorHitStyleStub::Phase2OTValidateVectorHitStyleStub(const edm::ParameterSet& iConfig)
    : topFolder_(iConfig.getParameter<std::string>("TopFolderName")),
      stubsToken_(consumes<reco::StubsHost>(iConfig.getParameter<edm::InputTag>("stubsSrc"))),
      otRecHitsToken_(consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"))),
      recHitCollectionToken_(
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("recHitSrc"))),
      tpToken_(consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("trackingParticleSrc"))),
      geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      hitAssocConfig_(iConfig.getParameter<edm::ParameterSet>("hitAssociatorConfig"), consumesCollector()),
      tpMinPt_(iConfig.getParameter<double>("TP_minPt")),
      tpMaxEta_(iConfig.getParameter<double>("TP_maxEta")),
      tpMaxVtxZ_(iConfig.getParameter<double>("TP_maxVtxZ")),
      tpMaxD0_(iConfig.getParameter<double>("TP_maxD0")),
      tpMaxLxy_(iConfig.getParameter<double>("TP_maxLxy")),
      tpMinNLayers_(iConfig.getParameter<int>("TP_minNLayers")) {}

void Phase2OTValidateVectorHitStyleStub::bookHistograms(DQMStore::IBooker& iBooker,
                                                        edm::Run const&,
                                                        edm::EventSetup const&) {
  iBooker.setCurrentFolder(topFolder_ + "/Summary");

  h_nStubs_ = iBooker.book1D("nStubs", "Number of stubs per event;N stubs;Events", 200, 0, 200000);
  h_stubCategory_ = iBooker.book1D("stubCategory", "Stub classification;Category;Stubs", 4, -0.5, 3.5);
  h_stubCategory_->setBinLabel(1, "Genuine");
  h_stubCategory_->setBinLabel(2, "Combinatoric");
  h_stubCategory_->setBinLabel(3, "Unknown");
  h_stubCategory_->setBinLabel(4, "PHitOnly");
  h_fakeRatePerEvent_ = iBooker.book1D("fakeRatePerEvent", "Fake rate per event;Fake rate;Events", 100, 0, 1);
  h_stub_rz_ = iBooker.book2D("stub_rz", "Stub R-Z;z [cm];R [cm]", 600, -300, 300, 240, 0, 120);

  // per-layer bins: 6 barrel + 5 endcap
  h_genuine_per_layer_ = iBooker.book1D("genuine_per_layer", "Genuine stubs per layer;Layer;Stubs", 11, 0.5, 11.5);
  h_total_per_layer_ = iBooker.book1D("total_per_layer", "Total stubs per layer;Layer;Stubs", 11, 0.5, 11.5);
  h_fake_per_layer_ = iBooker.book1D("fake_per_layer", "Non-genuine stubs per layer;Layer;Stubs", 11, 0.5, 11.5);
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
                       "Genuine stubs per layer (" + std::string(categoryTitles_[c]) + ");Layer;Stubs",
                       nLayers,
                       0.5,
                       nLayers + 0.5);
    h_total_per_layer_cat_[c] =
        iBooker.book1D("total_per_layer_" + std::string(categoryNames_[c]),
                       "Total stubs per layer (" + std::string(categoryTitles_[c]) + ");Layer;Stubs",
                       nLayers,
                       0.5,
                       nLayers + 0.5);
    h_fake_per_layer_cat_[c] =
        iBooker.book1D("fake_per_layer_" + std::string(categoryNames_[c]),
                       "Non-genuine stubs per layer (" + std::string(categoryTitles_[c]) + ");Layer;Stubs",
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

  h_genuine_tp_pt_ = iBooker.book1D("genuine_tp_pt", "TP p_{T} for genuine stubs;p_{T} [GeV];Stubs", 100, 0, 100);

  h_genuine_per_event_ =
      iBooker.book1D("genuine_per_event", "Genuine stubs per event;N genuine;Events", 200, 0, 100000);
  h_combinatoric_per_event_ =
      iBooker.book1D("combinatoric_per_event", "Combinatoric stubs per event;N combinatoric;Events", 200, 0, 100000);
  h_unknown_per_event_ =
      iBooker.book1D("unknown_per_event", "Unknown stubs per event;N unknown;Events", 200, 0, 100000);
  h_phitonly_per_event_ =
      iBooker.book1D("phitonly_per_event", "PHitOnly stubs per event;N PHitOnly;Events", 200, 0, 50000);
  h_genuine_matched_tp_pt_ = iBooker.book1D(
      "genuine_matched_tp_pt", "Matched TP p_{T} for genuine stubs;p_{T} [GeV];Genuine stubs", 200, 0, 100);

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

  h_fake_denom_pt_barrel_ =
      iBooker.book1D("fake_denom_pt_barrel", "Fake rate denominator (barrel);p_{T} [GeV];Stubs", 100, 0, 50);
  h_fake_numer_pt_barrel_ =
      iBooker.book1D("fake_numer_pt_barrel", "Fake rate numerator (barrel);p_{T} [GeV];Stubs", 100, 0, 50);
  h_fake_denom_pt_endcap_ =
      iBooker.book1D("fake_denom_pt_endcap", "Fake rate denominator (endcap);p_{T} [GeV];Stubs", 100, 0, 50);
  h_fake_numer_pt_endcap_ =
      iBooker.book1D("fake_numer_pt_endcap", "Fake rate numerator (endcap);p_{T} [GeV];Stubs", 100, 0, 50);
  h_fake_denom_pt_all_ =
      iBooker.book1D("fake_denom_pt_all", "Fake rate denominator (all);p_{T} [GeV];Stubs", 100, 0, 50);
  h_fake_numer_pt_all_ = iBooker.book1D("fake_numer_pt_all", "Fake rate numerator (all);p_{T} [GeV];Stubs", 100, 0, 50);

  for (int c = 0; c < NCategories; ++c) {
    h_fake_denom_pt_cat_[c] =
        iBooker.book1D("fake_denom_pt_" + std::string(categoryNames_[c]),
                       "Fake rate denominator (" + std::string(categoryTitles_[c]) + ");p_{T} [GeV];Stubs",
                       100,
                       0,
                       50);
    h_fake_numer_pt_cat_[c] =
        iBooker.book1D("fake_numer_pt_" + std::string(categoryNames_[c]),
                       "Fake rate numerator (" + std::string(categoryTitles_[c]) + ");p_{T} [GeV];Stubs",
                       100,
                       0,
                       50);
  }

  h_fake_denom_eta_barrel_ =
      iBooker.book1D("fake_denom_eta_barrel", "Fake rate denominator (barrel);#eta;Stubs", 100, -4.2, 4.2);
  h_fake_numer_eta_barrel_ =
      iBooker.book1D("fake_numer_eta_barrel", "Fake rate numerator (barrel);#eta;Stubs", 100, -4.2, 4.2);
  h_fake_denom_eta_endcap_ =
      iBooker.book1D("fake_denom_eta_endcap", "Fake rate denominator (endcap);#eta;Stubs", 100, -4.2, 4.2);
  h_fake_numer_eta_endcap_ =
      iBooker.book1D("fake_numer_eta_endcap", "Fake rate numerator (endcap);#eta;Stubs", 100, -4.2, 4.2);
  h_fake_denom_eta_all_ =
      iBooker.book1D("fake_denom_eta_all", "Fake rate denominator (all);#eta;Stubs", 100, -4.2, 4.2);
  h_fake_numer_eta_all_ = iBooker.book1D("fake_numer_eta_all", "Fake rate numerator (all);#eta;Stubs", 100, -4.2, 4.2);

  for (int c = 0; c < NCategories; ++c) {
    h_fake_denom_eta_cat_[c] =
        iBooker.book1D("fake_denom_eta_" + std::string(categoryNames_[c]),
                       "Fake rate denominator (" + std::string(categoryTitles_[c]) + ");#eta;Stubs",
                       100,
                       -4.2,
                       4.2);
    h_fake_numer_eta_cat_[c] =
        iBooker.book1D("fake_numer_eta_" + std::string(categoryNames_[c]),
                       "Fake rate numerator (" + std::string(categoryTitles_[c]) + ");#eta;Stubs",
                       100,
                       -4.2,
                       4.2);
  }

  iBooker.setCurrentFolder(topFolder_ + "/FakeRate/PerLayer");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_fake_numer_pt_perlayer_[i] =
        iBooker.book1D("fake_numer_pt_" + name, "Fake rate numerator (" + title + ");p_{T} [GeV];Stubs", 100, 0, 50);
    h_fake_denom_pt_perlayer_[i] =
        iBooker.book1D("fake_denom_pt_" + name, "Fake rate denominator (" + title + ");p_{T} [GeV];Stubs", 100, 0, 50);
    h_fake_numer_eta_perlayer_[i] =
        iBooker.book1D("fake_numer_eta_" + name, "Fake rate numerator (" + title + ");#eta;Stubs", 100, -4.2, 4.2);
    h_fake_denom_eta_perlayer_[i] =
        iBooker.book1D("fake_denom_eta_" + name, "Fake rate denominator (" + title + ");#eta;Stubs", 100, -4.2, 4.2);
  }

  iBooker.setCurrentFolder(topFolder_ + "/Residuals");

  h_phi_res_barrel_ =
      iBooker.book1D("phi_res_barrel", "#Delta#phi barrel (genuine);#phi_{stub} - #phi_{TP};Stubs", 200, -0.01, 0.01);
  h_phi_res_endcap_ =
      iBooker.book1D("phi_res_endcap", "#Delta#phi endcap (genuine);#phi_{stub} - #phi_{TP};Stubs", 200, -0.01, 0.01);
  h_z_res_barrel_ = iBooker.book1D("z_res_barrel", "#Deltaz barrel (genuine);z_{stub} - z_{TP} [cm];Stubs", 200, -2, 2);
  h_r_res_endcap_ = iBooker.book1D("r_res_endcap", "#DeltaR endcap (genuine);R_{stub} - R_{TP} [cm];Stubs", 200, -2, 2);

  iBooker.setCurrentFolder(topFolder_ + "/Diagnostics/DeltaY");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_dy_genuine_perlayer_[i] = iBooker.book1D(
        "dy_genuine_" + name, "|#Deltay| genuine (" + title + ");|y_{lower} - y_{upper}| [cm];Stubs", 200, 0, 10.0);
    h_dy_fake_perlayer_[i] = iBooker.book1D(
        "dy_fake_" + name, "|#Deltay| fake (" + title + ");|y_{lower} - y_{upper}| [cm];Stubs", 200, 0, 10.0);

    h_dypull_genuine_perlayer_[i] =
        iBooker.book1D("dypull_genuine_" + name,
                       "|#Deltay| pull genuine (" + title + ");|#Deltay| / #sigma_{#Deltay};Stubs",
                       200,
                       0,
                       20.0);
    h_dypull_fake_perlayer_[i] = iBooker.book1D(
        "dypull_fake_" + name, "|#Deltay| pull fake (" + title + ");|#Deltay| / #sigma_{#Deltay};Stubs", 200, 0, 20.0);
  }

  iBooker.setCurrentFolder(topFolder_ + "/Diagnostics/ClusterSize");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_clusterSize_lower_genuine_perlayer_[i] =
        iBooker.book1D("clusterSize_lower_genuine_" + name,
                       "Cluster size lower genuine (" + title + ");Cluster size;Stubs",
                       20,
                       0.5,
                       20.5);
    h_clusterSize_lower_fake_perlayer_[i] = iBooker.book1D(
        "clusterSize_lower_fake_" + name, "Cluster size lower fake (" + title + ");Cluster size;Stubs", 20, 0.5, 20.5);
    h_clusterSize_upper_genuine_perlayer_[i] =
        iBooker.book1D("clusterSize_upper_genuine_" + name,
                       "Cluster size upper genuine (" + title + ");Cluster size;Stubs",
                       20,
                       0.5,
                       20.5);
    h_clusterSize_upper_fake_perlayer_[i] = iBooker.book1D(
        "clusterSize_upper_fake_" + name, "Cluster size upper fake (" + title + ");Cluster size;Stubs", 20, 0.5, 20.5);
    h_clusterSizeDiff_genuine_perlayer_[i] =
        iBooker.book1D("clusterSizeDiff_genuine_" + name,
                       "|#DeltaClusterSize| genuine (" + title + ");|size_{lower} - size_{upper}|;Stubs",
                       20,
                       -0.5,
                       19.5);
    h_clusterSizeDiff_fake_perlayer_[i] =
        iBooker.book1D("clusterSizeDiff_fake_" + name,
                       "|#DeltaClusterSize| fake (" + title + ");|size_{lower} - size_{upper}|;Stubs",
                       20,
                       -0.5,
                       19.5);
    h_clusterSize_2d_genuine_perlayer_[i] =
        iBooker.book2D("clusterSize_2d_genuine_" + name,
                       "Cluster size lower vs upper genuine (" + title + ");Cluster size (lower);Cluster size (upper)",
                       20,
                       0.5,
                       20.5,
                       20,
                       0.5,
                       20.5);
    h_clusterSize_2d_fake_perlayer_[i] =
        iBooker.book2D("clusterSize_2d_fake_" + name,
                       "Cluster size lower vs upper fake (" + title + ");Cluster size (lower);Cluster size (upper)",
                       20,
                       0.5,
                       20.5,
                       20,
                       0.5,
                       20.5);
    h_clusterSizeSum_genuine_perlayer_[i] =
        iBooker.book1D("clusterSizeSum_genuine_" + name,
                       "Cluster size sum genuine (" + title + ");cs_{lower} + cs_{upper};Stubs",
                       39,
                       1.5,
                       40.5);
    h_clusterSizeSum_fake_perlayer_[i] =
        iBooker.book1D("clusterSizeSum_fake_" + name,
                       "Cluster size sum fake (" + title + ");cs_{lower} + cs_{upper};Stubs",
                       39,
                       1.5,
                       40.5);
    h_clusterSizeAsymmetry_genuine_perlayer_[i] = iBooker.book1D(
        "clusterSizeAsymmetry_genuine_" + name,
        "|#Deltacs| / #Sigmacs genuine (" + title + ");|cs_{lower} - cs_{upper}| / (cs_{lower} + cs_{upper});Stubs",
        50,
        0.0,
        1.0);
    h_clusterSizeAsymmetry_fake_perlayer_[i] = iBooker.book1D(
        "clusterSizeAsymmetry_fake_" + name,
        "|#Deltacs| / #Sigmacs fake (" + title + ");|cs_{lower} - cs_{upper}| / (cs_{lower} + cs_{upper});Stubs",
        50,
        0.0,
        1.0);
    h_clusterSizeMin_genuine_perlayer_[i] =
        iBooker.book1D("clusterSizeMin_genuine_" + name,
                       "Cluster size min genuine (" + title + ");min(cs_{lower}, cs_{upper});Stubs",
                       20,
                       0.5,
                       20.5);
    h_clusterSizeMin_fake_perlayer_[i] =
        iBooker.book1D("clusterSizeMin_fake_" + name,
                       "Cluster size min fake (" + title + ");min(cs_{lower}, cs_{upper});Stubs",
                       20,
                       0.5,
                       20.5);
  }

  iBooker.setCurrentFolder(topFolder_ + "/Diagnostics/VectorHitCompat");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_dPhiDr_genuine_perlayer_[i] = iBooker.book1D(
        "dPhiDr_genuine_" + name, "dPhiDr genuine (" + title + ");dPhiDr [rad/cm];Stubs", 200, -0.005, 0.005);
    h_dPhiDr_fake_perlayer_[i] =
        iBooker.book1D("dPhiDr_fake_" + name, "dPhiDr fake (" + title + ");dPhiDr [rad/cm];Stubs", 200, -0.005, 0.005);

    h_dPhiDrErr_genuine_perlayer_[i] =
        iBooker.book1D("dPhiDrErr_genuine_" + name,
                       "dPhiDr error genuine (" + title + ");#sigma_{dPhiDr} [rad/cm];Stubs",
                       200,
                       0,
                       0.005);
    h_dPhiDrErr_fake_perlayer_[i] = iBooker.book1D(
        "dPhiDrErr_fake_" + name, "dPhiDr error fake (" + title + ");#sigma_{dPhiDr} [rad/cm];Stubs", 200, 0, 0.005);

    h_dPhiDr_signif_genuine_perlayer_[i] =
        iBooker.book1D("dPhiDr_signif_genuine_" + name,
                       "|dPhiDr| / #sigma genuine (" + title + ");|dPhiDr| / #sigma;Stubs",
                       200,
                       0,
                       20.0);
    h_dPhiDr_signif_fake_perlayer_[i] = iBooker.book1D(
        "dPhiDr_signif_fake_" + name, "|dPhiDr| / #sigma fake (" + title + ");|dPhiDr| / #sigma;Stubs", 200, 0, 20.0);
  }

  iBooker.setCurrentFolder(topFolder_ + "/Diagnostics/StubsPerPHit");

  h_stubs_per_phit_genuine_ = iBooker.book1D(
      "stubs_per_phit_genuine", "Stubs per P-hit (P-hits with #geq1 genuine);N stubs;P-hits", 20, 0.5, 20.5);
  h_stubs_per_phit_fake_ =
      iBooker.book1D("stubs_per_phit_fake", "Stubs per P-hit (all-fake P-hits);N stubs;P-hits", 20, 0.5, 20.5);

  for (int i = 0; i < NBarrelLayers; ++i) {
    std::string layerStr = "B" + std::to_string(i + 1);
    h_stubs_per_phit_genuine_barrel_[i] = iBooker.book1D("stubs_per_phit_genuine_" + layerStr,
                                                         "Stubs per P-hit genuine (" + layerStr + ");N stubs;P-hits",
                                                         20,
                                                         0.5,
                                                         20.5);
    h_stubs_per_phit_fake_barrel_[i] = iBooker.book1D(
        "stubs_per_phit_fake_" + layerStr, "Stubs per P-hit all-fake (" + layerStr + ");N stubs;P-hits", 20, 0.5, 20.5);
  }

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);
    h_stubs_per_phit_genuine_perlayer_[i] =
        iBooker.book1D("stubs_per_phit_genuine_" + name,
                       "Stubs per P-hit (#geq1 genuine) (" + title + ");N stubs;P-hits",
                       20,
                       0.5,
                       20.5);
    h_stubs_per_phit_fake_perlayer_[i] = iBooker.book1D(
        "stubs_per_phit_fake_" + name, "Stubs per P-hit (all fake) (" + title + ");N stubs;P-hits", 20, 0.5, 20.5);
    h_stubs_per_phit_all_perlayer_[i] = iBooker.book1D(
        "stubs_per_phit_all_" + name, "Stubs per P-hit (all) (" + title + ");N stubs;P-hits", 20, 0.5, 20.5);
  }

  h_mean_stubs_per_phit_genuine_ =
      iBooker.bookProfile("mean_stubs_per_phit_genuine",
                          "Mean stubs per P-hit (#geq1 genuine) vs layer;Layer category;Mean N stubs",
                          NPerLayerCategories,
                          -0.5,
                          NPerLayerCategories - 0.5,
                          0,
                          100);
  h_mean_stubs_per_phit_fake_ =
      iBooker.bookProfile("mean_stubs_per_phit_fake",
                          "Mean stubs per P-hit (all fake) vs layer;Layer category;Mean N stubs",
                          NPerLayerCategories,
                          -0.5,
                          NPerLayerCategories - 0.5,
                          0,
                          100);
  h_mean_stubs_per_phit_all_ = iBooker.bookProfile("mean_stubs_per_phit_all",
                                                   "Mean stubs per P-hit (all) vs layer;Layer category;Mean N stubs",
                                                   NPerLayerCategories,
                                                   -0.5,
                                                   NPerLayerCategories - 0.5,
                                                   0,
                                                   100);
  for (int i = 0; i < NPerLayerCategories; ++i) {
    h_mean_stubs_per_phit_genuine_->setBinLabel(i + 1, perLayerCategoryName(i));
    h_mean_stubs_per_phit_fake_->setBinLabel(i + 1, perLayerCategoryName(i));
    h_mean_stubs_per_phit_all_->setBinLabel(i + 1, perLayerCategoryName(i));
  }

  iBooker.setCurrentFolder(topFolder_ + "/SimTruth");

  for (int i = 0; i < NPerLayerCategories; ++i) {
    std::string name = perLayerCategoryName(i);
    std::string title = perLayerCategoryTitle(i);

    h_width_sim_perlayer_[i] = iBooker.book1D(
        "width_sim_" + name, "|#Deltax_{local}| sim (" + title + ");|x_{lower} - x_{upper}| [cm];TP pairs", 200, 0, 0.3);
    h_width_sim_corr_perlayer_[i] = iBooker.book1D(
        "width_sim_corr_" + name, "|width| sim parallax-corr (" + title + ");|width_{corr}| [cm];TP pairs", 200, 0, 0.3);
    h_dy_sim_perlayer_[i] = iBooker.book1D(
        "dy_sim_" + name, "|#Deltay| sim (" + title + ");|y_{lower} - y_{upper}| [cm];TP pairs", 200, 0, 10.0);
    h_dypull_sim_perlayer_[i] = iBooker.book1D(
        "dypull_sim_" + name, "|#Deltay| pull sim (" + title + ");|#Deltay| / #sigma_{#Deltay};TP pairs", 200, 0, 20.0);
  }

  h_sim_pairs_per_layer_ =
      iBooker.book1D("sim_pairs_per_layer", "Simulated stub-able pairs per layer;Layer;TP pairs", 11, 0.5, 11.5);
  for (int i = 1; i <= 6; ++i) {
    h_sim_pairs_per_layer_->setBinLabel(i, "B" + std::to_string(i));
  }
  for (int i = 1; i <= 5; ++i) {
    h_sim_pairs_per_layer_->setBinLabel(6 + i, "D" + std::to_string(i));
  }

  h_sim_pt_ = iBooker.book1D("sim_pt", "TP p_{T} (with #geq1 OT pair);p_{T} [GeV];TPs", 100, 0, 50);
  h_sim_eta_ = iBooker.book1D("sim_eta", "TP #eta (with #geq1 OT pair);#eta;TPs", 100, -4.2, 4.2);
}

void Phase2OTValidateVectorHitStyleStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // getHandle() so that events where the HLT paths were not run are skipped
  const auto stubsHandle = iEvent.getHandle(stubsToken_);
  if (!stubsHandle.isValid())
    return;
  const auto otHitsHandle = iEvent.getHandle(otRecHitsToken_);
  if (!otHitsHandle.isValid())
    return;
  const auto recHitsHandle = iEvent.getHandle(recHitCollectionToken_);
  if (!recHitsHandle.isValid())
    return;
  const auto& stubs = *stubsHandle;
  const auto& otHits = *otHitsHandle;
  const auto& recHits = *recHitsHandle;
  edm::Handle<std::vector<TrackingParticle>> tpHandle;
  iEvent.getByToken(tpToken_, tpHandle);

  const auto& topo = iSetup.getData(topoToken_);
  const auto& trackerGeom = iSetup.getData(geomToken_);

  auto stubsView = stubs.view().stubs();
  auto hitsView = otHits.view().otRecHits();
  uint32_t nStubs = stubs.nStubs();

  TrackerHitAssociator hitAssociator(iEvent, hitAssocConfig_);

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

  // Pass 1: (simTrackId, eventId) -> TrackingParticle maps
  std::map<SimHitIdpr, edm::Ptr<TrackingParticle>> simTrackToTP;
  std::map<SimHitIdpr, edm::Ptr<TrackingParticle>> primarySimTrackToTP;
  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    edm::Ptr<TrackingParticle> tpPtr(tpHandle, iTP);
    for (const auto& g4Track : tpPtr->g4Tracks()) {
      simTrackToTP[{g4Track.trackId(), g4Track.eventId()}] = tpPtr;
    }
    if (!tpPtr->g4Tracks().empty()) {
      const auto& primary = tpPtr->g4Tracks().front();
      primarySimTrackToTP[{primary.trackId(), primary.eventId()}] = tpPtr;
    }
  }

  // Pass 2: classify each stub
  enum Classification { Genuine = 0, Combinatoric = 1, Unknown = 2, PHitOnly = 3 };

  struct StubTruthInfo {
    Classification classification;
    edm::Ptr<TrackingParticle> principalTP;
    edm::Ptr<TrackingParticle> bestTP;
  };

  std::vector<StubTruthInfo> stubTruth(nStubs);
  std::unordered_map<uint32_t, std::vector<uint32_t>> origHitToStubs;

  uint32_t nGenuine = 0, nCombinatoric = 0, nUnknown = 0, nPHitOnly = 0;

  auto findBestTP = [&simTrackToTP](const std::set<SimHitIdpr>& ids) -> edm::Ptr<TrackingParticle> {
    edm::Ptr<TrackingParticle> best;
    for (const auto& id : ids) {
      auto it = simTrackToTP.find(id);
      if (it != simTrackToTP.end()) {
        if (best.isNull() || it->second->pt() > best->pt()) {
          best = it->second;
        }
      }
    }
    return best;
  };

  for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
    if (!isStub(stubsView, iStub)) {
      stubTruth[iStub] = {PHitOnly, edm::Ptr<TrackingParticle>(), edm::Ptr<TrackingParticle>()};
      nPHitOnly++;
      continue;
    }

    uint32_t innerSoAIdx = stubsView[iStub].lowerHitIdx();
    uint32_t outerSoAIdx = stubsView[iStub].upperHitIdx();

    uint32_t innerOrigIdx = hitsView[innerSoAIdx].origRecHitIdx();
    uint32_t outerOrigIdx = hitsView[outerSoAIdx].origRecHitIdx();

    origHitToStubs[innerOrigIdx].push_back(iStub);
    origHitToStubs[outerOrigIdx].push_back(iStub);

    std::set<SimHitIdpr> innerIds, outerIds;

    if (innerOrigIdx < origRecHits.size()) {
      std::vector<SimHitIdpr> ids;
      hitAssociator.associatePhase2TrackerRecHit(origRecHits[innerOrigIdx], ids);
      innerIds.insert(ids.begin(), ids.end());
    }
    if (outerOrigIdx < origRecHits.size()) {
      std::vector<SimHitIdpr> ids;
      hitAssociator.associatePhase2TrackerRecHit(origRecHits[outerOrigIdx], ids);
      outerIds.insert(ids.begin(), ids.end());
    }

    std::set<edm::Ptr<TrackingParticle>> innerTPs, outerTPs;
    for (const auto& id : innerIds) {
      auto it = simTrackToTP.find(id);
      if (it != simTrackToTP.end())
        innerTPs.insert(it->second);
    }
    for (const auto& id : outerIds) {
      auto it = simTrackToTP.find(id);
      if (it != simTrackToTP.end())
        outerTPs.insert(it->second);
    }

    std::vector<edm::Ptr<TrackingParticle>> commonTPs;
    std::set_intersection(
        innerTPs.begin(), innerTPs.end(), outerTPs.begin(), outerTPs.end(), std::back_inserter(commonTPs));

    if (innerIds.empty() && outerIds.empty()) {
      stubTruth[iStub] = {Unknown, edm::Ptr<TrackingParticle>(), edm::Ptr<TrackingParticle>()};
      nUnknown++;
    } else if (!commonTPs.empty()) {
      edm::Ptr<TrackingParticle> principalTP = commonTPs.front();
      stubTruth[iStub] = {Genuine, principalTP, principalTP};
      nGenuine++;
    } else {
      std::set<SimHitIdpr> allIds;
      allIds.insert(innerIds.begin(), innerIds.end());
      allIds.insert(outerIds.begin(), outerIds.end());
      edm::Ptr<TrackingParticle> bestTP = findBestTP(allIds);
      stubTruth[iStub] = {Combinatoric, edm::Ptr<TrackingParticle>(), bestTP};
      nCombinatoric++;
    }
  }

  h_nStubs_->Fill(nStubs);
  h_stubCategory_->Fill(0.0, nGenuine);
  h_stubCategory_->Fill(1.0, nCombinatoric);
  h_stubCategory_->Fill(2.0, nUnknown);
  h_stubCategory_->Fill(3.0, nPHitOnly);

  uint32_t nClassified = nGenuine + nCombinatoric + nUnknown;
  if (nClassified > 0) {
    h_fakeRatePerEvent_->Fill(float(nCombinatoric + nUnknown) / float(nClassified));
  }

  h_genuine_per_event_->Fill(nGenuine);
  h_combinatoric_per_event_->Fill(nCombinatoric);
  h_unknown_per_event_->Fill(nUnknown);
  h_phitonly_per_event_->Fill(nPHitOnly);

  for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
    uint32_t innerSoAIdx = stubsView[iStub].lowerHitIdx();
    float x = hitsView[innerSoAIdx].xGlobal();
    float y = hitsView[innerSoAIdx].yGlobal();
    float z = hitsView[innerSoAIdx].zGlobal();
    float r = std::sqrt(x * x + y * y);
    h_stub_rz_->Fill(z, r);

    if (!isStub(stubsView, iStub))
      continue;

    uint8_t flags = stubsView[iStub].flags();
    bool isBarrel = reco::StubFlags::isBarrel(flags);
    uint8_t layer = reco::StubFlags::layer(flags);

    int layerBin = isBarrel ? layer : layer + 6;
    h_total_per_layer_->Fill(layerBin);

    ModuleCategory cat = moduleCategory(flags, z);
    h_total_per_layer_cat_[cat]->Fill(layer);

    if (stubTruth[iStub].classification == Genuine) {
      h_genuine_per_layer_->Fill(layerBin);
      h_genuine_per_layer_cat_[cat]->Fill(layer);

      auto const& tp = stubTruth[iStub].principalTP;
      if (tp.isNonnull()) {
        h_genuine_tp_pt_->Fill(tp->pt());
        h_genuine_matched_tp_pt_->Fill(tp->pt());

        float stub_phi = std::atan2(y, x);
        float tp_phi = tp->phi();
        float dphi = stub_phi - tp_phi;
        if (dphi > M_PI)
          dphi -= 2 * M_PI;
        if (dphi < -M_PI)
          dphi += 2 * M_PI;

        if (isBarrel) {
          h_phi_res_barrel_->Fill(dphi);
          float tp_tanl = tp->pz() / tp->pt();
          float tp_z_at_r = tp->vz() + r * tp_tanl;
          h_z_res_barrel_->Fill(z - tp_z_at_r);
        } else {
          h_phi_res_endcap_->Fill(dphi);
          float tp_tanl = tp->pz() / tp->pt();
          float tp_r_at_z = (tp_tanl != 0.f) ? (z - tp->vz()) / tp_tanl : 0.f;
          h_r_res_endcap_->Fill(r - std::abs(tp_r_at_z));
        }
      }
    }

    if (stubTruth[iStub].classification == Combinatoric || stubTruth[iStub].classification == Unknown) {
      h_fake_per_layer_->Fill(layerBin);
      h_fake_per_layer_cat_[cat]->Fill(layer);
    }

    {
      int plIdx = perLayerIndexFromStub(flags, z);
      if (plIdx >= 0 && plIdx < NPerLayerCategories) {
        uint32_t innerSoAIdx = stubsView[iStub].lowerHitIdx();
        uint32_t outerSoAIdx = stubsView[iStub].upperHitIdx();
        float dy = std::abs(hitsView[innerSoAIdx].yLocal() - hitsView[outerSoAIdx].yLocal());
        float dyErr2 = hitsView[innerSoAIdx].yerrLocal() + hitsView[outerSoAIdx].yerrLocal();
        float dyPull = (dyErr2 > 0.f) ? dy / std::sqrt(dyErr2) : 0.f;

        float dPhiDr = stubsView[iStub].dPhiDr();
        float dPhiDrErr = stubsView[iStub].dPhiDrError();
        float dPhiDrSignif = (dPhiDrErr > 0.f) ? std::abs(dPhiDr) / dPhiDrErr : 0.f;

        // Within a stack the converter stores lower-sensor hits before upper-sensor hits,
        // so the smaller SoA index of the pair is always the lower-sensor hit.
        uint32_t lowerSoAIdx = std::min(innerSoAIdx, outerSoAIdx);
        uint32_t upperSoAIdx = std::max(innerSoAIdx, outerSoAIdx);
        uint16_t csLower = hitsView[lowerSoAIdx].clusterSize();
        uint16_t csUpper = hitsView[upperSoAIdx].clusterSize();
        int csDiff = std::abs(int(csLower) - int(csUpper));
        int csSum = int(csLower) + int(csUpper);
        int csMin = std::min(int(csLower), int(csUpper));
        float csAsymmetry = (csSum > 0) ? float(csDiff) / float(csSum) : 0.f;

        if (stubTruth[iStub].classification == Genuine) {
          h_dy_genuine_perlayer_[plIdx]->Fill(dy);
          h_dypull_genuine_perlayer_[plIdx]->Fill(dyPull);
          h_dPhiDr_genuine_perlayer_[plIdx]->Fill(dPhiDr);
          h_dPhiDrErr_genuine_perlayer_[plIdx]->Fill(dPhiDrErr);
          h_dPhiDr_signif_genuine_perlayer_[plIdx]->Fill(dPhiDrSignif);
          h_clusterSize_lower_genuine_perlayer_[plIdx]->Fill(csLower);
          h_clusterSize_upper_genuine_perlayer_[plIdx]->Fill(csUpper);
          h_clusterSizeDiff_genuine_perlayer_[plIdx]->Fill(csDiff);
          h_clusterSize_2d_genuine_perlayer_[plIdx]->Fill(csLower, csUpper);
          h_clusterSizeSum_genuine_perlayer_[plIdx]->Fill(csSum);
          h_clusterSizeAsymmetry_genuine_perlayer_[plIdx]->Fill(csAsymmetry);
          h_clusterSizeMin_genuine_perlayer_[plIdx]->Fill(csMin);
        } else {
          h_dy_fake_perlayer_[plIdx]->Fill(dy);
          h_dypull_fake_perlayer_[plIdx]->Fill(dyPull);
          h_dPhiDr_fake_perlayer_[plIdx]->Fill(dPhiDr);
          h_dPhiDrErr_fake_perlayer_[plIdx]->Fill(dPhiDrErr);
          h_dPhiDr_signif_fake_perlayer_[plIdx]->Fill(dPhiDrSignif);
          h_clusterSize_lower_fake_perlayer_[plIdx]->Fill(csLower);
          h_clusterSize_upper_fake_perlayer_[plIdx]->Fill(csUpper);
          h_clusterSizeDiff_fake_perlayer_[plIdx]->Fill(csDiff);
          h_clusterSize_2d_fake_perlayer_[plIdx]->Fill(csLower, csUpper);
          h_clusterSizeSum_fake_perlayer_[plIdx]->Fill(csSum);
          h_clusterSizeAsymmetry_fake_perlayer_[plIdx]->Fill(csAsymmetry);
          h_clusterSizeMin_fake_perlayer_[plIdx]->Fill(csMin);
        }
      }
    }

    auto const& bestTP = stubTruth[iStub].bestTP;
    if (bestTP.isNonnull()) {
      float tp_pt = bestTP->pt();
      float tp_eta = bestTP->eta();

      int plIdx = perLayerIndexFromStub(flags, z);
      bool validPlIdx = (plIdx >= 0 && plIdx < NPerLayerCategories);

      if (isBarrel) {
        h_fake_denom_pt_barrel_->Fill(tp_pt);
        h_fake_denom_eta_barrel_->Fill(tp_eta);
      } else {
        h_fake_denom_pt_endcap_->Fill(tp_pt);
        h_fake_denom_eta_endcap_->Fill(tp_eta);
      }
      h_fake_denom_pt_all_->Fill(tp_pt);
      h_fake_denom_eta_all_->Fill(tp_eta);
      h_fake_denom_pt_cat_[cat]->Fill(tp_pt);
      h_fake_denom_eta_cat_[cat]->Fill(tp_eta);
      if (validPlIdx) {
        h_fake_denom_pt_perlayer_[plIdx]->Fill(tp_pt);
        h_fake_denom_eta_perlayer_[plIdx]->Fill(tp_eta);
      }

      if (stubTruth[iStub].classification == Genuine) {
        if (isBarrel) {
          h_fake_numer_pt_barrel_->Fill(tp_pt);
          h_fake_numer_eta_barrel_->Fill(tp_eta);
        } else {
          h_fake_numer_pt_endcap_->Fill(tp_pt);
          h_fake_numer_eta_endcap_->Fill(tp_eta);
        }
        h_fake_numer_pt_all_->Fill(tp_pt);
        h_fake_numer_eta_all_->Fill(tp_eta);
        h_fake_numer_pt_cat_[cat]->Fill(tp_pt);
        h_fake_numer_eta_cat_[cat]->Fill(tp_eta);
        if (validPlIdx) {
          h_fake_numer_pt_perlayer_[plIdx]->Fill(tp_pt);
          h_fake_numer_eta_perlayer_[plIdx]->Fill(tp_eta);
        }
      }
    }
  }

  {
    std::unordered_map<uint32_t, std::vector<uint32_t>> phitToStubs;
    for (uint32_t iStub = 0; iStub < nStubs; ++iStub) {
      if (!isStub(stubsView, iStub))
        continue;
      uint32_t groupId = stubsView[iStub].lowerHitIdx();
      phitToStubs[groupId].push_back(iStub);
    }

    for (const auto& [groupId, stubIndices] : phitToStubs) {
      int nStubsForPhit = static_cast<int>(stubIndices.size());
      bool hasGenuine = false;
      uint8_t firstFlags = stubsView[stubIndices[0]].flags();
      float firstZ = hitsView[groupId].zGlobal();
      bool isBarrelPhit = reco::StubFlags::isBarrel(firstFlags);
      int barrelLayer = isBarrelPhit ? static_cast<int>(reco::StubFlags::layer(firstFlags)) : 0;
      int plIdx = perLayerIndexFromStub(firstFlags, firstZ);
      bool validPlIdx = (plIdx >= 0 && plIdx < NPerLayerCategories);

      for (uint32_t idx : stubIndices) {
        if (stubTruth[idx].classification == Genuine) {
          hasGenuine = true;
          break;
        }
      }

      if (validPlIdx) {
        h_stubs_per_phit_all_perlayer_[plIdx]->Fill(nStubsForPhit);
        h_mean_stubs_per_phit_all_->Fill(plIdx, nStubsForPhit);
      }

      if (hasGenuine) {
        h_stubs_per_phit_genuine_->Fill(nStubsForPhit);
        if (isBarrelPhit && barrelLayer >= 1 && barrelLayer <= NBarrelLayers) {
          h_stubs_per_phit_genuine_barrel_[barrelLayer - 1]->Fill(nStubsForPhit);
        }
        if (validPlIdx) {
          h_stubs_per_phit_genuine_perlayer_[plIdx]->Fill(nStubsForPhit);
          h_mean_stubs_per_phit_genuine_->Fill(plIdx, nStubsForPhit);
        }
      } else {
        h_stubs_per_phit_fake_->Fill(nStubsForPhit);
        if (isBarrelPhit && barrelLayer >= 1 && barrelLayer <= NBarrelLayers) {
          h_stubs_per_phit_fake_barrel_[barrelLayer - 1]->Fill(nStubsForPhit);
        }
        if (validPlIdx) {
          h_stubs_per_phit_fake_perlayer_[plIdx]->Fill(nStubsForPhit);
          h_mean_stubs_per_phit_fake_->Fill(plIdx, nStubsForPhit);
        }
      }
    }
  }

  // Pass 3: efficiency
  struct TPHitInfo {
    uint32_t origRecHitIdx;
    bool isBarrel;
    ModuleCategory category;
    int perLayerIdx;
  };
  std::map<edm::Ptr<TrackingParticle>, std::vector<TPHitInfo>> tpToLowerHits;

  struct TPSimHitInfo {
    uint32_t origRecHitIdx;
    uint32_t stackDetId;
    bool isLowerSensor;
    int perLayerIdx;
    int layerBin;
  };
  std::map<edm::Ptr<TrackingParticle>, std::vector<TPSimHitInfo>> tpToOTHits;

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
    if (isOnStack) {
      cat = moduleCategoryFromDetId(detId, topo, trackerGeom);
      plIdx = perLayerIndexFromDetId(detId, topo, trackerGeom);
      int layerNum = topo.layer(detId);
      layerBin = isBarrel ? layerNum : layerNum + 6;
    }

    for (const auto& recHit : detSet) {
      if (isOnStack) {
        std::vector<SimHitIdpr> ids;
        hitAssociator.associatePhase2TrackerRecHit(&recHit, ids);
        for (const auto& id : ids) {
          if (isLower) {
            auto it = primarySimTrackToTP.find(id);
            if (it != primarySimTrackToTP.end()) {
              tpToLowerHits[it->second].push_back({flatIdx, isBarrel, cat, plIdx});
            }
          }
          auto it = simTrackToTP.find(id);
          if (it != simTrackToTP.end()) {
            tpToOTHits[it->second].push_back({flatIdx, stackId, isLower, plIdx, layerBin});
          }
        }
      }
      flatIdx++;
    }
  }

  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    edm::Ptr<TrackingParticle> tpPtr(tpHandle, iTP);

    if (tpPtr->charge() == 0)
      continue;
    if (tpPtr->pt() < tpMinPt_)
      continue;
    if (std::abs(tpPtr->eta()) > tpMaxEta_)
      continue;
    if (std::abs(tpPtr->vz()) > tpMaxVtxZ_)
      continue;
    float d0 = tpPtr->d0();
    if (std::abs(d0) > tpMaxD0_)
      continue;
    float lxy = std::sqrt(tpPtr->vx() * tpPtr->vx() + tpPtr->vy() * tpPtr->vy());
    if (lxy > tpMaxLxy_)
      continue;

    auto hitIt = tpToLowerHits.find(tpPtr);
    if (hitIt == tpToLowerHits.end())
      continue;

    if (static_cast<int>(hitIt->second.size()) < tpMinNLayers_)
      continue;

    float tp_pt = tpPtr->pt();
    float tp_eta = tpPtr->eta();

    for (const auto& hitInfo : hitIt->second) {
      if (hitInfo.isBarrel) {
        h_eff_denom_pt_barrel_->Fill(tp_pt);
        h_eff_denom_eta_barrel_->Fill(tp_eta);
      } else {
        h_eff_denom_pt_endcap_->Fill(tp_pt);
        h_eff_denom_eta_endcap_->Fill(tp_eta);
      }
      h_eff_denom_pt_all_->Fill(tp_pt);
      h_eff_denom_eta_all_->Fill(tp_eta);
      h_eff_denom_pt_cat_[hitInfo.category]->Fill(tp_pt);
      h_eff_denom_eta_cat_[hitInfo.category]->Fill(tp_eta);

      if (hitInfo.perLayerIdx >= 0 && hitInfo.perLayerIdx < NPerLayerCategories) {
        h_eff_denom_pt_perlayer_[hitInfo.perLayerIdx]->Fill(tp_pt);
        h_eff_denom_eta_perlayer_[hitInfo.perLayerIdx]->Fill(tp_eta);
      }

      bool hasGenuineStub = false;
      auto stubIt = origHitToStubs.find(hitInfo.origRecHitIdx);
      if (stubIt != origHitToStubs.end()) {
        for (uint32_t stubIdx : stubIt->second) {
          if (stubTruth[stubIdx].classification == Genuine && stubTruth[stubIdx].principalTP == tpPtr) {
            hasGenuineStub = true;
            break;
          }
        }
      }

      if (hasGenuineStub) {
        if (hitInfo.isBarrel) {
          h_eff_numer_pt_barrel_->Fill(tp_pt);
          h_eff_numer_eta_barrel_->Fill(tp_eta);
        } else {
          h_eff_numer_pt_endcap_->Fill(tp_pt);
          h_eff_numer_eta_endcap_->Fill(tp_eta);
        }
        h_eff_numer_pt_all_->Fill(tp_pt);
        h_eff_numer_eta_all_->Fill(tp_eta);
        h_eff_numer_pt_cat_[hitInfo.category]->Fill(tp_pt);
        h_eff_numer_eta_cat_[hitInfo.category]->Fill(tp_eta);

        if (hitInfo.perLayerIdx >= 0 && hitInfo.perLayerIdx < NPerLayerCategories) {
          h_eff_numer_pt_perlayer_[hitInfo.perLayerIdx]->Fill(tp_pt);
          h_eff_numer_eta_perlayer_[hitInfo.perLayerIdx]->Fill(tp_eta);
        }
      }
    }
  }

  // Pass 4: simulation-truth diagnostics
  for (size_t iTP = 0; iTP < tpHandle->size(); ++iTP) {
    edm::Ptr<TrackingParticle> tpPtr(tpHandle, iTP);

    if (tpPtr->charge() == 0)
      continue;
    if (tpPtr->pt() < tpMinPt_)
      continue;
    if (std::abs(tpPtr->eta()) > tpMaxEta_)
      continue;
    if (std::abs(tpPtr->vz()) > tpMaxVtxZ_)
      continue;
    if (std::abs(tpPtr->d0()) > tpMaxD0_)
      continue;
    float lxy_sim = std::sqrt(tpPtr->vx() * tpPtr->vx() + tpPtr->vy() * tpPtr->vy());
    if (lxy_sim > tpMaxLxy_)
      continue;

    auto otHitIt = tpToOTHits.find(tpPtr);
    if (otHitIt == tpToOTHits.end())
      continue;

    int nLowerHits = 0;
    for (const auto& simHit : otHitIt->second) {
      if (simHit.isLowerSensor)
        nLowerHits++;
    }
    if (nLowerHits < tpMinNLayers_)
      continue;

    struct StackHitGroup {
      std::vector<uint32_t> lowerOrigIdx;
      std::vector<uint32_t> upperOrigIdx;
      int perLayerIdx;
      int layerBin;
    };
    std::map<uint32_t, StackHitGroup> stackGroups;
    for (const auto& simHit : otHitIt->second) {
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
      if (group.lowerOrigIdx.empty() || group.upperOrigIdx.empty())
        continue;

      for (uint32_t lowerOrig : group.lowerOrigIdx) {
        auto lowerSoAIt = origIdxToSoAIdx.find(lowerOrig);
        if (lowerSoAIt == origIdxToSoAIdx.end())
          continue;
        uint32_t lowerSoA = lowerSoAIt->second;

        for (uint32_t upperOrig : group.upperOrigIdx) {
          auto upperSoAIt = origIdxToSoAIdx.find(upperOrig);
          if (upperSoAIt == origIdxToSoAIdx.end())
            continue;
          uint32_t upperSoA = upperSoAIt->second;

          float dx = std::abs(hitsView[lowerSoA].xLocal() - hitsView[upperSoA].xLocal());
          float dy = std::abs(hitsView[lowerSoA].yLocal() - hitsView[upperSoA].yLocal());
          float dyErr2 = hitsView[lowerSoA].yerrLocal() + hitsView[upperSoA].yerrLocal();
          float dyPull = (dyErr2 > 0.f) ? dy / std::sqrt(dyErr2) : 0.f;

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
                if (!std::isfinite(pC_abs))
                  pC_abs = 0.f;
              }
              dx_corr = std::max(0.f, dx - pC_abs);
            }
          }

          int plIdx = group.perLayerIdx;
          if (plIdx >= 0 && plIdx < NPerLayerCategories) {
            h_width_sim_perlayer_[plIdx]->Fill(dx);
            h_width_sim_corr_perlayer_[plIdx]->Fill(dx_corr);
            h_dy_sim_perlayer_[plIdx]->Fill(dy);
            h_dypull_sim_perlayer_[plIdx]->Fill(dyPull);
          }
          h_sim_pairs_per_layer_->Fill(group.layerBin);
          hasSimPair = true;
        }
      }
    }

    if (hasSimPair) {
      h_sim_pt_->Fill(tpPtr->pt());
      h_sim_eta_->Fill(tpPtr->eta());
    }
  }

  edm::LogInfo("Phase2OTValidateVectorHitStyleStub")
      << "Event " << iEvent.id().event() << ": nStubs=" << nStubs << " genuine=" << nGenuine << " ("
      << (nClassified > 0 ? 100.0f * nGenuine / nClassified : 0.f) << "%)"
      << " combinatoric=" << nCombinatoric << " unknown=" << nUnknown << " PHitOnly=" << nPHitOnly;
}

void Phase2OTValidateVectorHitStyleStub::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("TopFolderName", "HLT/TrackerPhase2OTStubV");

  desc.add<edm::InputTag>("stubsSrc", edm::InputTag("hltOTStubProducer"));
  desc.add<edm::InputTag>("otRecHitsSrc", edm::InputTag("hltPixelSeedingOTRecHitsSoA"));
  desc.add<edm::InputTag>("recHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));

  desc.add<double>("TP_minPt", 0.75);
  desc.add<double>("TP_maxEta", 4.2);
  desc.add<double>("TP_maxVtxZ", 60.0);
  desc.add<double>("TP_maxD0", 5.0);
  desc.add<double>("TP_maxLxy", 60.0);
  desc.add<int>("TP_minNLayers", 4);

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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Phase2OTValidateVectorHitStyleStub);
