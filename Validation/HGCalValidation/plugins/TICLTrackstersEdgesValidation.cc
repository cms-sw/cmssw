#include <string>
#include <unordered_map>
#include <numeric>

// user include files
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

using namespace ticl;

struct Histogram_TICLTrackstersEdgesValidation {
  dqm::reco::MonitorElement* number_;
  dqm::reco::MonitorElement *raw_energy_, *raw_energy_1plusLC_;
  dqm::reco::MonitorElement *regr_energy_, *regr_energy_1plusLC_;
  dqm::reco::MonitorElement *raw_energy_vs_regr_energy_, *raw_energy_vs_regr_energy_1plusLC_;
  dqm::reco::MonitorElement *id_prob_, *id_prob_1plusLC_;
  dqm::reco::MonitorElement* delta_energy_;
  dqm::reco::MonitorElement* delta_energy_relative_;
  dqm::reco::MonitorElement* delta_energy_vs_energy_;
  dqm::reco::MonitorElement* delta_energy_vs_layer_;
  dqm::reco::MonitorElement* delta_energy_relative_vs_layer_;
  dqm::reco::MonitorElement* delta_layer_;
  dqm::reco::MonitorElement* ingoing_links_vs_layer_;
  dqm::reco::MonitorElement* outgoing_links_vs_layer_;
  // For the definition of the angles, read http://hgcal.web.cern.ch/hgcal/Reconstruction/Tutorial/
  dqm::reco::MonitorElement* angle_alpha_;
  dqm::reco::MonitorElement* angle_alpha_alternative_;
  dqm::reco::MonitorElement* angle_beta_;
  std::vector<dqm::reco::MonitorElement*> angle_beta_byLayer_;
  std::vector<dqm::reco::MonitorElement*> angle_beta_w_byLayer_;
};

using Histograms_TICLTrackstersEdgesValidation =
    std::unordered_map<unsigned int, Histogram_TICLTrackstersEdgesValidation>;

class TICLTrackstersEdgesValidation : public DQMGlobalEDAnalyzer<Histograms_TICLTrackstersEdgesValidation> {
public:
  explicit TICLTrackstersEdgesValidation(const edm::ParameterSet&);
  ~TICLTrackstersEdgesValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms_TICLTrackstersEdgesValidation&) const override;

  void dqmAnalyze(edm::Event const&,
                  edm::EventSetup const&,
                  Histograms_TICLTrackstersEdgesValidation const&) const override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&, Histograms_TICLTrackstersEdgesValidation&) const override;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  std::string folder_;
  std::vector<std::string> trackstersCollectionsNames_;
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> tracksterTokens_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  edm::EDGetTokenT<std::vector<TICLSeedingRegion>> ticlSeedingGlobalToken_;
  edm::EDGetTokenT<std::vector<TICLSeedingRegion>> ticlSeedingTrkToken_;
  mutable hgcal::RecHitTools rhtools_;
};

TICLTrackstersEdgesValidation::TICLTrackstersEdgesValidation(const edm::ParameterSet& iConfig)
    : caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      folder_(iConfig.getParameter<std::string>("folder")) {
  tracksterTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("tracksterCollections"),
                                           [this](edm::InputTag const& tag) {
                                             trackstersCollectionsNames_.emplace_back(tag.label());
                                             return consumes<std::vector<Trackster>>(tag);
                                           });
  layerClustersToken_ = consumes<std::vector<reco::CaloCluster>>(iConfig.getParameter<edm::InputTag>("layerClusters"));
  ticlSeedingGlobalToken_ =
      consumes<std::vector<TICLSeedingRegion>>(iConfig.getParameter<edm::InputTag>("ticlSeedingGlobal"));
  ticlSeedingTrkToken_ =
      consumes<std::vector<TICLSeedingRegion>>(iConfig.getParameter<edm::InputTag>("ticlSeedingTrk"));
}

TICLTrackstersEdgesValidation::~TICLTrackstersEdgesValidation() {}

void TICLTrackstersEdgesValidation::dqmAnalyze(edm::Event const& iEvent,
                                               edm::EventSetup const& iSetup,
                                               Histograms_TICLTrackstersEdgesValidation const& histos) const {
  edm::Handle<std::vector<reco::CaloCluster>> layerClustersH;
  iEvent.getByToken(layerClustersToken_, layerClustersH);
  auto const& layerClusters = *layerClustersH.product();

  edm::Handle<std::vector<TICLSeedingRegion>> ticlSeedingGlobalH;
  iEvent.getByToken(ticlSeedingGlobalToken_, ticlSeedingGlobalH);
  auto const& ticlSeedingGlobal = *ticlSeedingGlobalH.product();

  edm::Handle<std::vector<TICLSeedingRegion>> ticlSeedingTrkH;
  iEvent.getByToken(ticlSeedingTrkToken_, ticlSeedingTrkH);
  auto const& ticlSeedingTrk = *ticlSeedingTrkH.product();

  for (const auto& trackster_token : tracksterTokens_) {
    edm::Handle<std::vector<Trackster>> trackster_h;
    iEvent.getByToken(trackster_token, trackster_h);
    auto numberOfTracksters = trackster_h->size();
    //using .at() as [] is not const
    const auto& histo = histos.at(trackster_token.index());
    histo.number_->Fill(numberOfTracksters);
    for (unsigned int i = 0; i < numberOfTracksters; ++i) {
      const auto& thisTrackster = trackster_h->at(i);

      // The following plots should be moved to HGVHistoProducerAlgo
      // when we get rid of the MultiClusters and use only Tracksters
      histo.raw_energy_->Fill(thisTrackster.raw_energy());
      histo.regr_energy_->Fill(thisTrackster.regressed_energy());
      histo.raw_energy_vs_regr_energy_->Fill(thisTrackster.regressed_energy(), thisTrackster.raw_energy());
      const auto& probs = thisTrackster.id_probabilities();
      std::vector<int> sorted_probs_idx(probs.size());
      std::iota(begin(sorted_probs_idx), end(sorted_probs_idx), 0);
      std::sort(begin(sorted_probs_idx), end(sorted_probs_idx), [&probs](int i, int j) { return probs[i] > probs[j]; });
      histo.id_prob_->Fill(sorted_probs_idx[0]);
      if (!thisTrackster.vertices().empty()) {
        histo.raw_energy_1plusLC_->Fill(thisTrackster.raw_energy());
        histo.regr_energy_1plusLC_->Fill(thisTrackster.regressed_energy());
        histo.raw_energy_vs_regr_energy_1plusLC_->Fill(thisTrackster.regressed_energy(), thisTrackster.raw_energy());
        histo.id_prob_1plusLC_->Fill(sorted_probs_idx[0]);
      }

      // Plots on edges
      for (const auto& edge : thisTrackster.edges()) {
        auto& ic = layerClusters[edge[0]];
        auto& oc = layerClusters[edge[1]];
        auto const& cl_in = ic.hitsAndFractions()[0].first;
        auto const& cl_out = oc.hitsAndFractions()[0].first;
        auto const layer_in = rhtools_.getLayerWithOffset(cl_in);
        auto const layer_out = rhtools_.getLayerWithOffset(cl_out);
        histo.delta_energy_->Fill(oc.energy() - ic.energy());
        histo.delta_energy_relative_->Fill((oc.energy() - ic.energy()) / ic.energy());
        histo.delta_energy_vs_energy_->Fill(oc.energy() - ic.energy(), ic.energy());
        histo.delta_energy_vs_layer_->Fill(layer_in, oc.energy() - ic.energy());
        histo.delta_energy_relative_vs_layer_->Fill(layer_in, (oc.energy() - ic.energy()) / ic.energy());
        histo.delta_layer_->Fill(layer_out - layer_in);

        // Alpha angles
        const auto& outer_outer_pos = oc.position();
        const auto& outer_inner_pos = ic.position();
        const auto& seed = thisTrackster.seedIndex();
        auto seedGlobalPos = math::XYZPoint(
            ticlSeedingGlobal[0].origin.x(), ticlSeedingGlobal[0].origin.y(), ticlSeedingGlobal[0].origin.z());
        auto seedDirectionPos = outer_inner_pos;
        if (thisTrackster.seedID().id() != 0) {
          // Seed to trackster association is, at present, rather convoluted.
          for (auto const& s : ticlSeedingTrk) {
            if (s.index == seed) {
              seedGlobalPos = math::XYZPoint(s.origin.x(), s.origin.y(), s.origin.z());
              seedDirectionPos =
                  math::XYZPoint(s.directionAtOrigin.x(), s.directionAtOrigin.y(), s.directionAtOrigin.z());
              break;
            }
          }
        }

        auto alpha = (outer_inner_pos - seedGlobalPos).Dot(outer_outer_pos - outer_inner_pos) /
                     sqrt((outer_inner_pos - seedGlobalPos).Mag2() * (outer_outer_pos - outer_inner_pos).Mag2());
        auto alpha_alternative = (outer_outer_pos - seedGlobalPos).Dot(seedDirectionPos) /
                                 sqrt((outer_outer_pos - seedGlobalPos).Mag2() * seedDirectionPos.Mag2());
        histo.angle_alpha_->Fill(alpha);
        histo.angle_alpha_alternative_->Fill(alpha_alternative);

        // Beta angle is usually computed using 2 edges. Another inner loop
        // is therefore needed.
        std::vector<std::array<unsigned int, 2>> innerDoublets;
        std::vector<std::array<unsigned int, 2>> outerDoublets;
        for (const auto& otherEdge : thisTrackster.edges()) {
          if (otherEdge[1] == edge[0]) {
            innerDoublets.push_back(otherEdge);
          }
          if (edge[1] == otherEdge[0]) {
            outerDoublets.push_back(otherEdge);
          }
        }

        histo.ingoing_links_vs_layer_->Fill(layer_in, innerDoublets.size());
        histo.outgoing_links_vs_layer_->Fill(layer_out, outerDoublets.size());
        for (const auto& inner : innerDoublets) {
          const auto& inner_ic = layerClusters[inner[0]];
          const auto& inner_inner_pos = inner_ic.position();
          auto beta = (outer_inner_pos - inner_inner_pos).Dot(outer_outer_pos - inner_inner_pos) /
                      sqrt((outer_inner_pos - inner_inner_pos).Mag2() * (outer_outer_pos - inner_inner_pos).Mag2());
          histo.angle_beta_->Fill(beta);
          histo.angle_beta_byLayer_[layer_in]->Fill(beta);
          histo.angle_beta_w_byLayer_[layer_in]->Fill(beta, ic.energy());
        }
      }
    }
  }
}

void TICLTrackstersEdgesValidation::bookHistograms(DQMStore::IBooker& ibook,
                                                   edm::Run const& run,
                                                   edm::EventSetup const& iSetup,
                                                   Histograms_TICLTrackstersEdgesValidation& histos) const {
  float eMin = 0., eThresh = 70., eMax = 500;
  float eWidth[] = {0.5, 2.};
  std::vector<float> eBins;
  float eVal = eMin;
  while (eVal <= eThresh) {
    eBins.push_back(eVal);
    eVal += eWidth[0];
  }
  while (eVal < eMax) {
    eVal += eWidth[1];
    eBins.push_back(eVal);
  }
  int eNBins = eBins.size() - 1;

  TString onePlusLC[] = {"1plus LC", "for tracksters with at least one LC"};
  TString trkers = "Tracksters";
  static const char* particle_kind[] = {
      "photon", "electron", "muon", "neutral_pion", "charged_hadron", "neutral_hadron", "ambiguous", "unknown"};
  auto nCategory = sizeof(particle_kind) / sizeof(*particle_kind);
  int labelIndex = 0;
  for (const auto& trackster_token : tracksterTokens_) {
    auto& histo = histos[trackster_token.index()];
    ibook.setCurrentFolder(folder_ + "HGCalValidator/" + trackstersCollectionsNames_[labelIndex]);
    histo.number_ = ibook.book1D(
        "Number of Tracksters per Event", "Number of Tracksters per Event;# Tracksters;Events", 250, 0., 250.);
    // The following plots should be moved to HGVHistoProducerAlgo
    // when we get rid of the MultiClusters and use only Tracksters
    histo.raw_energy_ = ibook.book1D("Raw Energy", "Raw Energy;Raw Energy [GeV];" + trkers, eNBins, &eBins[0]);
    histo.regr_energy_ =
        ibook.book1D("Regressed Energy", "Regressed Energy;Regressed Energy [GeV];" + trkers, eNBins, &eBins[0]);
    histo.raw_energy_vs_regr_energy_ = ibook.book2D("Raw Energy vs Regressed Energy",
                                                    "Raw vs Regressed Energy;Regressed Energy [GeV];Raw Energy [GeV]",
                                                    eNBins,
                                                    &eBins[0],
                                                    eNBins,
                                                    &eBins[0]);
    histo.id_prob_ =
        ibook.book1D("ID probability", "ID probability;category;Max ID probability", nCategory, 0, nCategory);
    histo.raw_energy_1plusLC_ = ibook.book1D(
        "Raw Energy " + onePlusLC[0], "Raw Energy " + onePlusLC[1] + ";Raw Energy [GeV];" + trkers, eNBins, &eBins[0]);
    histo.regr_energy_1plusLC_ = ibook.book1D("Regressed Energy " + onePlusLC[0],
                                              "Regressed Energy " + onePlusLC[1] + ";Regressed Energy [GeV];" + trkers,
                                              eNBins,
                                              &eBins[0]);
    histo.raw_energy_vs_regr_energy_1plusLC_ =
        ibook.book2D("Raw Energy vs Regressed Energy " + onePlusLC[0],
                     "Raw vs Regressed Energy " + onePlusLC[1] + ";Regressed Energy [GeV];Raw Energy [GeV]",
                     eNBins,
                     &eBins[0],
                     eNBins,
                     &eBins[0]);
    histo.id_prob_1plusLC_ = ibook.book1D("ID probability " + onePlusLC[0],
                                          "ID probability " + onePlusLC[1] + ";category;Max ID probability",
                                          nCategory,
                                          0,
                                          nCategory);
    for (int iBin = 0; iBin < histo.id_prob_->getNbinsX(); iBin++) {
      histo.id_prob_->setBinLabel(iBin + 1, particle_kind[iBin]);
      histo.id_prob_1plusLC_->setBinLabel(iBin + 1, particle_kind[iBin]);
    }
    // Plots on edges
    histo.delta_energy_ = ibook.book1D("Delta energy", "Delta Energy (O-I)", 800, -20., 20.);
    histo.delta_energy_relative_ =
        ibook.book1D("Relative Delta energy", "Relative Delta Energy (O-I)/I", 200, -10., 10.);
    histo.delta_energy_vs_energy_ =
        ibook.book2D("Energy vs Delta Energy", "Energy (I) vs Delta Energy (O-I)", 800, -20., 20., 200, 0., 20.);
    histo.delta_energy_vs_layer_ = ibook.book2D(
        "Delta Energy (O-I) vs Layer Number (I)", "Delta Energy (O-I) vs Layer Number (I)", 50, 0., 50., 800, -20., 20.);
    histo.delta_energy_relative_vs_layer_ = ibook.book2D("Relative Delta Energy (O-I)_I vs Layer Number (I)",
                                                         "Relative Delta Energy (O-I)_I vs Layer Number (I)",
                                                         50,
                                                         0.,
                                                         50.,
                                                         200,
                                                         -10.,
                                                         10.);
    histo.ingoing_links_vs_layer_ =
        ibook.book2D("Ingoing links Layer Number", "Ingoing links vs Layer Number", 50, 0., 50., 40, 0., 40.);
    histo.outgoing_links_vs_layer_ =
        ibook.book2D("Outgoing links vs Layer Number", "Outgoing links vs Layer Number", 50, 0., 50., 40, 0., 40.);
    histo.delta_layer_ = ibook.book1D("Delta Layer", "Delta Layer", 10, 0., 10.);
    histo.angle_alpha_ = ibook.book1D("cosAngle Alpha", "cosAngle Alpha", 200, -1., 1.);
    histo.angle_beta_ = ibook.book1D("cosAngle Beta", "cosAngle Beta", 200, -1., 1.);
    histo.angle_alpha_alternative_ = ibook.book1D("cosAngle Alpha Alternative", "Angle Alpha Alternative", 200, 0., 1.);
    for (int layer = 0; layer < 50; layer++) {
      auto layerstr = std::to_string(layer + 1);
      if (layerstr.length() < 2)
        layerstr.insert(0, 2 - layerstr.length(), '0');
      histo.angle_beta_byLayer_.push_back(
          ibook.book1D("cosAngle Beta on Layer " + layerstr, "cosAngle Beta on Layer " + layerstr, 200, -1., 1.));
      histo.angle_beta_w_byLayer_.push_back(ibook.book1D(
          "cosAngle Beta Weighted on Layer " + layerstr, "cosAngle Beta Weighted on Layer " + layerstr, 200, -1., 1.));
    }
    labelIndex++;
  }
}

void TICLTrackstersEdgesValidation::dqmBeginRun(edm::Run const& run,
                                                edm::EventSetup const& iSetup,
                                                Histograms_TICLTrackstersEdgesValidation& histograms) const {
  edm::ESHandle<CaloGeometry> geom = iSetup.getHandle(caloGeomToken_);
  rhtools_.setGeometry(*geom);
}

void TICLTrackstersEdgesValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<edm::InputTag> source_vector{edm::InputTag("ticlTrackstersTrk"),
                                           edm::InputTag("ticlTrackstersTrkEM"),
                                           edm::InputTag("ticlTrackstersEM"),
                                           edm::InputTag("ticlTrackstersHAD"),
                                           edm::InputTag("ticlTrackstersMerge")};
  desc.add<std::vector<edm::InputTag>>("tracksterCollections", source_vector);
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("ticlSeedingGlobal", edm::InputTag("ticlSeedingGlobal"));
  desc.add<edm::InputTag>("ticlSeedingTrk", edm::InputTag("ticlSeedingTrk"));
  desc.add<std::string>("folder", "HGCAL/");
  descriptions.add("ticlTrackstersEdgesValidationDefault", desc);
}

DEFINE_FWK_MODULE(TICLTrackstersEdgesValidation);
