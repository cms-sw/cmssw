#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

template <typename RecoClusterCollection>
class PFTesterT : public DQMEDAnalyzer {
public:
  explicit PFTesterT(const edm::ParameterSet&);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::string doubleToString(double x) const;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  edm::EDGetTokenT<reco::PFCandidateCollection> PFCandToken_;
  edm::EDGetTokenT<reco::PFRecHitCollection> RechitToken_;
  edm::EDGetTokenT<RecoClusterCollection> RecoClusterToken_;
  edm::EDGetTokenT<CaloParticleCollection> CaloParticleToken_;
  edm::EDGetTokenT<SimClusterCollection> SimClusterToken_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionWithSimClustersT<RecoClusterCollection>> RecoToSimAssociatorToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionWithSimClustersT<RecoClusterCollection>> SimToRecoAssociatorToken_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionT<RecoClusterCollection>> RecoToCpAssociatorToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionT<RecoClusterCollection>> CpToRecoAssociatorToken_;

  MonitorElement* h_PFCandEt_;
  MonitorElement* h_PFCandEta_;
  MonitorElement* h_PFCandPhi_;
  MonitorElement* h_PFCandCharge_;
  MonitorElement* h_PFCandPdgId_;
  MonitorElement* h_PFCandType_;

  MonitorElement* h_NumElements_;
  MonitorElement* h_NumTrackElements_;
  MonitorElement* h_NumMuonElements_;
  MonitorElement* h_NumPS1Elements_;
  MonitorElement* h_NumPS2Elements_;
  MonitorElement* h_NumECALElements_;
  MonitorElement* h_NumHCALElements_;
  MonitorElement* h_NumHGCALElements_;

  MonitorElement* h_TrackCharge_;
  MonitorElement* h_TrackNumPoints_;
  MonitorElement* h_TrackNumMeasurements_;
  MonitorElement* h_TrackImpactParameter_;

  MonitorElement* h_PFClusterE_;
  MonitorElement* h_PFClusterEta_;
  MonitorElement* h_PFClusterPhi_;
  MonitorElement* h_PFClusterDepth_;
  MonitorElement* h_PFClusterNHits_;
  MonitorElement* h_PFClusterType_;
  MonitorElement* h_PFClusterHitFraction_;
  MonitorElement* h_PFClusterHitDetId_;

  MonitorElement* h_CPToSCEnergyFraction_;
  MonitorElement* h_CPToSHEnergyFraction_;
  MonitorElement* h_CP_recoToSimScore_;
  MonitorElement* h_CP_simToRecoScore_;
  MonitorElement* h_CP_simToRecoShEnF_;
  MonitorElement* h_CP_simToRecoShEnF_Score_;

  MonitorElement* h_nPFClusters_;
  MonitorElement* h_nSimClusters_;
  MonitorElement* h_nSimClustersPrimary_;
  MonitorElement* h_recoToSimScore_;
  MonitorElement* h_simToRecoScore_;
  MonitorElement* h_simToRecoShEnF_;
  MonitorElement* h_simToRecoShEnF_Score_;
  MonitorElement* h_simToRecoShEnF_En_;
  MonitorElement* h_simToRecoShEnF_EnFrac_;
  MonitorElement* h_simToRecoShEnF_EnSimTrack_;
  MonitorElement* h_simToRecoShEnF_Mult_;
  MonitorElement* h_simToRecoScore_En_;
  MonitorElement* h_simToRecoScore_EnFrac_;
  MonitorElement* h_simToRecoScore_EnSimTrack_;
  MonitorElement* h_simToRecoScore_Mult_;
  MonitorElement* h_SimTrackToSimHitsEnergyFraction_;

  std::vector<double> assocScoreThresholds_;
  uint nAssocScoreThresholds_;
  double enFracCut_;
  double ptCut_;
  double etaCut_;
  bool doMatchByScore_;
  std::string outFolder_;

  const std::unordered_map<std::string, std::tuple<unsigned, float, float>> histoVarsReco = {
      {"En", std::make_tuple(100, 0., 100.)},
      {"Pt", std::make_tuple(200, 0., 100.)},
      {"PtLow", std::make_tuple(100, 0., 10.)},
      {"Eta", std::make_tuple(50, -6.5, 6.5)},
      {"Phi", std::make_tuple(50, -3.5, 3.5)},
      {"Mult", std::make_tuple(200, 0., 200.)},
  };
  const std::unordered_map<std::string, std::tuple<unsigned, float, float>> histoVarsSim = {
      {"En", std::make_tuple(100, 0., 100.)},
      {"EnFrac", std::make_tuple(220, 0., 1.1)},
      {"EnSimTrack", std::make_tuple(100, 0., 100.)},
      {"Pt", std::make_tuple(200, 0., 100.)},
      {"PtLow", std::make_tuple(100, 0., 10.)},
      {"Eta", std::make_tuple(50, -6.5, 6.5)},
      {"Phi", std::make_tuple(50, -3.5, 3.5)},
      {"Mult", std::make_tuple(200, 0., 200.)},
  };

  using UMap = std::unordered_map<std::string, MonitorElement*>;
  using VUMap = std::vector<UMap>;
  UMap h_simClusters_;
  VUMap h_simClustersMatchedRecoClusters_;
  VUMap h_simClustersMultiMatchedRecoClusters_;
  UMap h_recoClusters_;
  VUMap h_recoClustersMatchedSimClusters_;
  VUMap h_recoClustersMultiMatchedSimClusters_;
  std::vector<MonitorElement*> h_nSimMatchedToOneReco_;
  std::vector<MonitorElement*> h_nRecoMatchedToOneSim_;

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> histo2dVarsReco = {
      {"En_Eta", std::make_tuple(100, 0., 100., 50, -6.5, 6.5)},
      {"En_Phi", std::make_tuple(100, 0., 100., 50, -3.5, 3.5)},
      {"En_Mult", std::make_tuple(100, 0., 100., 200, 0., 200.)},
      {"Pt_Eta", std::make_tuple(100, 0., 40., 50, -6.5, 6.5)},
      {"Pt_Phi", std::make_tuple(100, 0., 40., 50, -3.5, 3.5)},
      {"Pt_Mult", std::make_tuple(100, 0., 40., 200, 0., 200.)},
      {"Mult_Eta", std::make_tuple(200, 0., 200., 50, -6.5, 6.5)},
      {"Mult_Phi", std::make_tuple(200, 0., 200., 50, -3.5, 3.5)},
  };

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> histo2dVarsSim = {
      {"En_Eta", std::make_tuple(100, 0., 100., 50, -6.5, 6.5)},
      {"En_Phi", std::make_tuple(100, 0., 100., 50, -3.5, 3.5)},
      {"En_Mult", std::make_tuple(100, 0., 100., 200, 0., 200.)},
      {"EnFrac_Eta", std::make_tuple(220, 0., 1.1, 50, -6.5, 6.5)},
      {"EnFrac_Phi", std::make_tuple(220, 0., 1.1, 50, -3.5, 3.5)},
      {"EnFrac_Mult", std::make_tuple(220, 0., 1.1, 200, 0., 200.)},
      {"EnSimTrack_Eta", std::make_tuple(100, 0., 100., 50, -6.5, 6.5)},
      {"EnSimTrack_Phi", std::make_tuple(100, 0., 100., 50, -3.5, 3.5)},
      {"EnSimTrack_Mult", std::make_tuple(100, 0., 100., 200, 0., 200.)},
      {"Pt_Eta", std::make_tuple(100, 0., 40., 50, -6.5, 6.5)},
      {"Pt_Phi", std::make_tuple(100, 0., 40., 50, -3.5, 3.5)},
      {"Pt_Mult", std::make_tuple(100, 0., 40., 200, 0., 200.)},
      {"Mult_Eta", std::make_tuple(200, 0., 200., 50, -6.5, 6.5)},
      {"Mult_Phi", std::make_tuple(200, 0., 200., 50, -3.5, 3.5)},
  };

  using U2Map = std::unordered_map<std::string, MonitorElement*>;
  using VU2Map = std::vector<std::unordered_map<std::string, MonitorElement*>>;
  U2Map h2d_simClusters_;
  VU2Map h2d_simClustersMatchedRecoClusters_;
  U2Map h2d_recoClusters_;
  VU2Map h2d_recoClustersMatchedSimClusters_;

  VU2Map h2d_responsePt_;
  VU2Map h2d_responseE_;
};

template <typename RecoClusterCollection>
PFTesterT<RecoClusterCollection>::PFTesterT(const edm::ParameterSet& iConfig)
    : geometry_token_(esConsumes()),
      PFCandToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCand"))),
      RechitToken_(consumes<reco::PFRecHitCollection>(iConfig.getParameter<edm::InputTag>("Rechit"))),
      RecoClusterToken_(consumes<RecoClusterCollection>(iConfig.getParameter<edm::InputTag>("RecoCluster"))),
      CaloParticleToken_(consumes<CaloParticleCollection>(iConfig.getParameter<edm::InputTag>("CaloParticle"))),
      SimClusterToken_(consumes<SimClusterCollection>(iConfig.getParameter<edm::InputTag>("SimCluster"))),
      RecoToSimAssociatorToken_(consumes<ticl::RecoToSimCollectionWithSimClustersT<RecoClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("ClusterSimClusterAssociator"))),
      SimToRecoAssociatorToken_(consumes<ticl::SimToRecoCollectionWithSimClustersT<RecoClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("ClusterSimClusterAssociator"))),
      RecoToCpAssociatorToken_(consumes<ticl::RecoToSimCollectionT<RecoClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("ClusterCaloParticleAssociator"))),
      CpToRecoAssociatorToken_(consumes<ticl::SimToRecoCollectionT<RecoClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("ClusterCaloParticleAssociator"))),
      assocScoreThresholds_(iConfig.getParameter<std::vector<double>>("assocScoreThresholds")),
      enFracCut_(iConfig.getParameter<double>("enFracCut")),
      ptCut_(iConfig.getParameter<double>("ptCut")),
      etaCut_(iConfig.getParameter<double>("etaCut")),
      doMatchByScore_(iConfig.getParameter<bool>("doMatchByScore")),
      outFolder_(iConfig.getParameter<std::string>("outFolder")) {
  nAssocScoreThresholds_ = assocScoreThresholds_.size();
  h_simClustersMatchedRecoClusters_.resize(nAssocScoreThresholds_);
  h_simClustersMultiMatchedRecoClusters_.resize(nAssocScoreThresholds_);
  h_recoClustersMatchedSimClusters_.resize(nAssocScoreThresholds_);
  h_recoClustersMultiMatchedSimClusters_.resize(nAssocScoreThresholds_);
  h2d_simClustersMatchedRecoClusters_.resize(nAssocScoreThresholds_);
  h2d_recoClustersMatchedSimClusters_.resize(nAssocScoreThresholds_);
  h2d_responsePt_.resize(nAssocScoreThresholds_);
  h2d_responseE_.resize(nAssocScoreThresholds_);
  h_nSimMatchedToOneReco_.resize(nAssocScoreThresholds_);
  h_nRecoMatchedToOneSim_.resize(nAssocScoreThresholds_);
}

template <typename RecoClusterCollection>
void PFTesterT<RecoClusterCollection>::bookHistograms(DQMStore::IBooker& ibook,
                                                      edm::Run const&,
                                                      edm::EventSetup const&) {
  std::string matching = doMatchByScore_ ? "MatchByScore" : "MatchByShEnF";
  ibook.setCurrentFolder(outFolder_ + "/" + matching + "/CaloParticles");
  h_CPToSCEnergyFraction_ =
      ibook.book1D("CaloParticleToSimClusterEnergyFraction",
                   "CaloParticleToSimClusterEnergyFraction;CaloParticle to SimCluster energy fraction",
                   100,
                   0,
                   2);
  h_CPToSHEnergyFraction_ =
      ibook.book1D("CaloParticleToSimHitsEnergyFraction",
                   "CaloParticleToSimHitsEnergyFraction;CaloParticle to SimHits energy fraction",
                   100,
                   0,
                   2);
  h_CP_recoToSimScore_ =
      ibook.book1D("CP_recoToSimScore", "CPrecoToSimScore;CaloParticle Reco #rightarrow Sim score", 51, 0, 1.02);
  h_CP_simToRecoScore_ =
      ibook.book1D("CP_simToRecoScore", "CPsimToRecoScore;CaloParticle Sim #rightarrow Reco score", 51, 0, 1.02);
  h_CP_simToRecoShEnF_ = ibook.book1D("CP_simToRecoShEnF",
                                      "simToRecoSharedEnergy;CaloParticle Sim #rightarrow Reco shared energy fraction",
                                      51,
                                      0,
                                      1.02);
  h_CP_simToRecoShEnF_Score_ = ibook.book2D("CP_simToRecoShEnF_Score",
                                            "CaloParticle #rightarrow RecoCluster simToRecoSharedEnergy_Score;Sim "
                                            "#rightarrow Reco shared energy fraction;Sim #rightarrow Reco score",
                                            51,
                                            0,
                                            1.02,
                                            51,
                                            0,
                                            1.02);

  std::string pfValidFolder = outFolder_ + "/" + matching + "/PFClusterValidation";
  ibook.setCurrentFolder(pfValidFolder);
  h_nSimClusters_ = ibook.book1D("nSimClusters", "Number of SimClusters;Number of SimClusters per event", 100, 0, 100);
  h_nSimClustersPrimary_ = ibook.book1D(
      "nSimClustersPrimary", "Number of Primary SimClusters;Number of Primary SimClusters per event", 100, 0, 100);
  h_nPFClusters_ = ibook.book1D("nPFClusters", "Number of PFClusters per PFCandidate", 100, 0, 100);
  h_recoToSimScore_ = ibook.book1D("recoToSimScore", "recoToSimScore;Reco #rightarrow Sim score", 51, 0, 1.02);
  h_simToRecoScore_ = ibook.book1D("simToRecoScore", "simToRecoScore;Sim #rightarrow Reco score", 51, 0, 1.02);
  h_simToRecoShEnF_ =
      ibook.book1D("simToRecoShEnF", "simToRecoSharedEnergy;Sim #rightarrow Reco shared energy fraction", 51, 0, 1.02);
  h_simToRecoShEnF_Score_ =
      ibook.book2D("simToRecoShEnF_Score",
                   "simToRecoSharedEnergy_Score;Sim #rightarrow Reco shared energy fraction;Sim #rightarrow Reco score",
                   51,
                   0,
                   1.02,
                   51,
                   0,
                   1.02);
  h_simToRecoShEnF_En_ =
      ibook.book2D("simToRecoShEnF_En",
                   "simToRecoSharedEnergy vs Energy;Sim #rightarrow Reco shared energy fraction;Energy_{hits}",
                   51,
                   0,
                   1.02,
                   100,
                   0.,
                   100.);
  h_simToRecoShEnF_EnFrac_ =
      ibook.book2D("simToRecoShEnF_EnFrac",
                   "simToRecoSharedEnergy vs Energy Fraction;Sim #rightarrow Reco shared energy fraction;EnFrac",
                   51,
                   0,
                   1.02,
                   220,
                   0.,
                   1.1);
  h_simToRecoShEnF_EnSimTrack_ =
      ibook.book2D("simToRecoShEnF_EnSimTrack",
                   "simToRecoSharedEnergy vs SimTrack Energy;Sim #rightarrow Reco shared energy fraction;SimTrack Energy",
                   51,
                   0,
                   1.02,
                   100,
                   0.,
                   100.);
  h_simToRecoShEnF_Mult_ =
      ibook.book2D("simToRecoShEnF_Mult",
                   "simToRecoSharedEnergy vs Multiplicity;Sim #rightarrow Reco shared energy fraction;Multiplicity",
                   51,
                   0,
                   1.02,
                   200,
                   0.,
                   200.);
  h_simToRecoScore_En_ = ibook.book2D("simToRecoScore_En",
                                          "simToRecoScore vs Energy;Sim #rightarrow Reco score;Energy_{hits}",
                                          51,
                                          0,
                                          1.02,
                                          100,
                                          0.,
                                          100.);
  h_simToRecoScore_EnFrac_ = ibook.book2D("simToRecoScore_EnFrac",
                                          "simToRecoScore vs Energy Fraction;Sim #rightarrow Reco score;EnFrac",
                                          51,
                                          0,
                                          1.02,
                                          220,
                                          0.,
                                          1.1);
  h_simToRecoScore_EnSimTrack_ = ibook.book2D(
      "simToRecoScore_EnSimTrack", "simToRecoScore vs SimTrack Energy;Sim #rightarrow Reco score;SimTrack Energy", 51, 0, 1.02, 100, 0., 100.);
  h_simToRecoScore_Mult_ = ibook.book2D("simToRecoScore_Mult",
                                        "simToRecoScore vs Multiplicity;Sim #rightarrow Reco score;Multiplicity",
                                        51,
                                        0,
                                        1.02,
                                        200,
                                        0.,
                                        200.);
  h_SimTrackToSimHitsEnergyFraction_ =
      ibook.book1D("SimTrackToSimHitsEnergyFraction",
                   "SimTrackToSimHitsEnergyFraction;SimTrack to SimHits energy fraction",
                   110,
                   0,
                   1.1);

  for (auto& hVar : histoVarsSim) {
    auto [nBins, hMin, hMax] = hVar.second;
    h_simClusters_[hVar.first] =
        ibook.book1D("SimClusters" + hVar.first, "SimClusters;" + hVar.first, nBins, hMin, hMax);
  }

  for (auto& hVar : histoVarsReco) {
    auto [nBins, hMin, hMax] = hVar.second;
    h_recoClusters_[hVar.first] =
        ibook.book1D("RecoClusters" + hVar.first, "RecoClusters;" + hVar.first, nBins, hMin, hMax);
  }

  for (auto& h2dVar : histo2dVarsSim) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = h2dVar.second;
    auto x_title = h2dVar.first.substr(0, h2dVar.first.find("_"));
    auto y_title = h2dVar.first.substr(h2dVar.first.find("_") + 1);
    h2d_simClusters_[h2dVar.first] = ibook.book2D("SimClusters" + h2dVar.first,
                                                  "SimClusters;" + x_title + ";" + y_title,
                                                  nBinsX,
                                                  hMinX,
                                                  hMaxX,
                                                  nBinsY,
                                                  hMinY,
                                                  hMaxY);
  }
  for (auto& h2dVar : histo2dVarsReco) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = h2dVar.second;
    auto x_title = h2dVar.first.substr(0, h2dVar.first.find("_"));
    auto y_title = h2dVar.first.substr(h2dVar.first.find("_") + 1);
    h2d_recoClusters_[h2dVar.first] = ibook.book2D("RecoClusters" + h2dVar.first,
                                                   "RecoClusters;" + x_title + ";" + y_title,
                                                   nBinsX,
                                                   hMinX,
                                                   hMaxX,
                                                   nBinsY,
                                                   hMinY,
                                                   hMaxY);
  }
  
  for (unsigned ithr = 0; ithr < nAssocScoreThresholds_; ++ithr) {
    std::string threshStr = "Score" + doubleToString(assocScoreThresholds_[ithr]);
    ibook.setCurrentFolder(pfValidFolder + "/" + threshStr);
    h_nSimMatchedToOneReco_[ithr] = ibook.book1D(
      "nSimMatchedToOneReco",
      "Number of SimClusters matched to a RecoCluster;Number of RecoClusters; Number of matched SimClusters",
      10,
      0,
      10);
    h_nRecoMatchedToOneSim_[ithr] = ibook.book1D(
      "nRecoMatchedToOneSim",
      "Number of RecoClusters matched to a SimCluster;Number of SimClusters; Number of matched RecoClusters",
      10,
      0,
      10);
    for (auto& hVar : histoVarsSim) {
      auto [nBins, hMin, hMax] = hVar.second;
      h_simClustersMatchedRecoClusters_[ithr][hVar.first] =
          ibook.book1D("SimClustersMatchedRecoClusters" + hVar.first,
                       "SimClusters matched to RecoClusters;" + hVar.first,
                       nBins,
                       hMin,
                       hMax);
      h_simClustersMultiMatchedRecoClusters_[ithr][hVar.first] =
          ibook.book1D("SimClustersMultiMatchedRecoClusters" + hVar.first,
                       "SimClusters multi-matched to RecoClusters;" + hVar.first,
                       nBins,
                       hMin,
                       hMax);
      h2d_responsePt_[ithr][hVar.first] =
          ibook.book2D("ResponsePt_" + hVar.first, "Response p_T;" + hVar.first, nBins, hMin, hMax, 50, 0., 2.);
      h2d_responseE_[ithr][hVar.first] =
          ibook.book2D("ResponseE_" + hVar.first, "Response Energy;" + hVar.first, nBins, hMin, hMax, 50, 0., 2.);
    }
    for (auto& hVar : histoVarsReco) {
      auto [nBins, hMin, hMax] = hVar.second;
      h_recoClustersMatchedSimClusters_[ithr][hVar.first] =
          ibook.book1D("RecoClustersMatchedSimClusters" + hVar.first,
                       "RecoClusters matched to SimClusters;" + hVar.first,
                       nBins,
                       hMin,
                       hMax);
      h_recoClustersMultiMatchedSimClusters_[ithr][hVar.first] =
          ibook.book1D("RecoClustersMultiMatchedSimClusters" + hVar.first,
                       "RecoClusters multi-matched to SimClusters;" + hVar.first,
                       nBins,
                       hMin,
                       hMax);
    }
    for (auto& h2dVar : histo2dVarsSim) {
      auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = h2dVar.second;
      auto x_title = h2dVar.first.substr(0, h2dVar.first.find("_"));
      auto y_title = h2dVar.first.substr(h2dVar.first.find("_") + 1);
      h2d_simClustersMatchedRecoClusters_[ithr][h2dVar.first] =
          ibook.book2D("SimClustersMatchedRecoClusters" + h2dVar.first,
                       "SimClusters matched to RecoClusters;" + x_title + ";" + y_title,
                       nBinsX,
                       hMinX,
                       hMaxX,
                       nBinsY,
                       hMinY,
                       hMaxY);
    }
    for (auto& h2dVar : histo2dVarsReco) {
      auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = h2dVar.second;
      auto x_title = h2dVar.first.substr(0, h2dVar.first.find("_"));
      auto y_title = h2dVar.first.substr(h2dVar.first.find("_") + 1);
      h2d_recoClustersMatchedSimClusters_[ithr][h2dVar.first] =
          ibook.book2D("RecoClustersMatchedSimClusters" + h2dVar.first,
                       "RecoClusters matched to SimClusters;" + x_title + ";" + y_title,
                       nBinsX,
                       hMinX,
                       hMaxX,
                       nBinsY,
                       hMinY,
                       hMaxY);
    }
  }

  for (auto& hVar : histoVarsSim) {
    auto [nBins, hMin, hMax] = hVar.second;
    for (unsigned ithr = 0; ithr < nAssocScoreThresholds_; ++ithr) {
      std::string threshStr = "Score" + doubleToString(assocScoreThresholds_[ithr]);
      ibook.setCurrentFolder(pfValidFolder + "/" + threshStr);
      h2d_responsePt_[ithr][hVar.first] =
          ibook.book2D("ResponsePt_" + hVar.first, "Response p_T;" + hVar.first, nBins, hMin, hMax, 50, 0., 1.5);
      h2d_responseE_[ithr][hVar.first] =
          ibook.book2D("ResponseE_" + hVar.first, "Response Energy;" + hVar.first, nBins, hMin, hMax, 50, 0., 1.5);
    }
  }

  ibook.setCurrentFolder(outFolder_ + "/PFCandidates");

  h_PFCandEt_ = ibook.book1D("PFCandEt", "PFCandEt", 1000, 0, 1000);
  h_PFCandEta_ = ibook.book1D("PFCandEta", "PFCandEta", 200, -5, 5);
  h_PFCandPhi_ = ibook.book1D("PFCandPhi", "PFCandPhi", 200, -M_PI, M_PI);
  h_PFCandCharge_ = ibook.book1D("PFCandCharge", "PFCandCharge", 5, -2, 2);
  h_PFCandPdgId_ = ibook.book1D("PFCandPdgId", "PFCandPdgId", 44, -22, 22);
  h_PFCandType_ = ibook.book1D("PFCandidateType", "PFCandidateType", 10, 0, 10);

  ibook.setCurrentFolder(outFolder_ + "/" + matching + "/PFBlocks");
  h_NumElements_ = ibook.book1D("NumElements", "NumElements", 25, 0, 25);
  h_NumTrackElements_ = ibook.book1D("NumTrackElements", "NumTrackElements", 5, 0, 5);
  h_NumMuonElements_ = ibook.book1D("NumMuonElements", "NumMuonElements", 5, 0, 5);
  h_NumPS1Elements_ = ibook.book1D("NumPS1Elements", "NumPS1Elements", 5, 0, 5);
  h_NumPS2Elements_ = ibook.book1D("NumPS2Elements", "NumPS2Elements", 5, 0, 5);
  h_NumECALElements_ = ibook.book1D("NumECALElements", "NumECALElements", 5, 0, 5);
  h_NumHCALElements_ = ibook.book1D("NumHCALElements", "NumHCALElements", 5, 0, 5);
  h_NumHGCALElements_ = ibook.book1D("NumHGCALElements", "NumHGCALElements", 5, 0, 5);

  ibook.setCurrentFolder(outFolder_ + "/" + matching + "/PFTracks");
  h_TrackCharge_ = ibook.book1D("TrackCharge", "TrackCharge", 5, -2, 2);
  h_TrackNumPoints_ = ibook.book1D("TrackNumPoints", "TrackNumPoints", 100, 0, 100);
  h_TrackNumMeasurements_ = ibook.book1D("TrackNumMeasurements", "TrackNumMeasurements", 100, 0, 100);
  h_TrackImpactParameter_ = ibook.book1D("TrackImpactParameter", "TrackImpactParameter", 1000, 0, 1);

  ibook.setCurrentFolder(outFolder_ + "/" + matching + "/PFClusters");
  h_PFClusterE_ = ibook.book1D("PFClusterE", "RecoCluster Energy;E [GeV]", 100, 0, 100);
  h_PFClusterEta_ = ibook.book1D("PFClusterEta", "RecoCluster Eta;#eta", 120, -6, 6);
  h_PFClusterPhi_ = ibook.book1D("PFClusterPhi", "RecoCluster Phi;#phi", 128, -3.2, 3.2);
  h_PFClusterDepth_ = ibook.book1D("PFClusterDepth", "RecoCluster Depth;Depth", 10, 0, 10);
  h_PFClusterNHits_ = ibook.book1D("PFClusterNHits", "RecoCluster Number of Hits", 100, 0, 100);
  h_PFClusterType_ = ibook.book1D("PFClusterEtaWidth", "RecoCluster Eta Width;#sigma_{#eta}", 20, 0, 20);
  h_PFClusterHitFraction_ = ibook.book1D("PFClusterHitFraction", "RecoCluster Hit Fraction;Fraction", 100, 0.0, 1.1);
  h_PFClusterHitDetId_ =
      ibook.book1D("PFClusterHitDetId", "RecoCluster Hit DetId modulo 10000;DetId mod 10000", 100, 0, 10000);
}

template <typename RecoClusterCollection>
void PFTesterT<RecoClusterCollection>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // --------------------------------------------------------------------
  // ---------------- PF Clusters and associators -----------------------
  // --------------------------------------------------------------------
  // std::cout << std::endl;
  // std::cout << "--- Analyze ---" << std::endl;
  edm::Handle<reco::PFRecHitCollection> Rechit;
  iEvent.getByToken(RechitToken_, Rechit);
  if (!Rechit.isValid()) {
    edm::LogPrint("PFTester") << "Input Rechit collection not found.";
    return;
  }
  auto pfRechit = *Rechit;

  edm::Handle<RecoClusterCollection> RecoCluster;
  iEvent.getByToken(RecoClusterToken_, RecoCluster);
  if (!RecoCluster.isValid()) {
    edm::LogPrint("PFTester") << "Input RecoCluster collection not found.";
    return;
  }
  auto recoClusters = *RecoCluster;

  edm::Handle<SimClusterCollection> SimCluster;
  iEvent.getByToken(SimClusterToken_, SimCluster);
  if (!SimCluster.isValid()) {
    edm::LogPrint("PFTester") << "Input SimCluster collection not found.";
    return;
  }
  auto simClusters = *SimCluster;

  edm::Handle<ticl::SimToRecoCollectionWithSimClustersT<RecoClusterCollection>> SimToRecoAssociatorCollection;
  iEvent.getByToken(SimToRecoAssociatorToken_, SimToRecoAssociatorCollection);
  if (!SimToRecoAssociatorCollection.isValid()) {
    edm::LogPrint("PFTester") << "Input ClusterSimClusterAssociator SimToReco collection not found.";
    return;
  }
  auto simToRecoAssoc = *SimToRecoAssociatorCollection;

  edm::Handle<ticl::RecoToSimCollectionWithSimClustersT<RecoClusterCollection>> RecoToSimAssociatorCollection;
  iEvent.getByToken(RecoToSimAssociatorToken_, RecoToSimAssociatorCollection);
  if (!RecoToSimAssociatorCollection.isValid()) {
    edm::LogPrint("PFTester") << "Input ClusterSimClusterAssociator RecoToSim collection not found.";
    return;
  }
  auto recoToSimAssoc = *RecoToSimAssociatorCollection;

  // --------------------------------------------------------------------
  // ---------------- Calo Particles and associators --------------------
  // --------------------------------------------------------------------

  edm::Handle<CaloParticleCollection> CaloParticle;
  iEvent.getByToken(CaloParticleToken_, CaloParticle);
  if (!CaloParticle.isValid()) {
    edm::LogPrint("PFTester") << "Input CaloParticle collection not found.";
    return;
  }
  auto caloParticles = *CaloParticle;

  edm::Handle<ticl::SimToRecoCollectionT<RecoClusterCollection>> CpToRecoAssociatorCollection;
  iEvent.getByToken(CpToRecoAssociatorToken_, CpToRecoAssociatorCollection);
  if (!CpToRecoAssociatorCollection.isValid()) {
    edm::LogPrint("PFTester") << "Input ClusterCaloParticleAssociator SimToReco collection not found.";
    return;
  }
  auto cpToRecoAssoc = *CpToRecoAssociatorCollection;

  edm::Handle<ticl::RecoToSimCollectionT<RecoClusterCollection>> RecoToCpAssociatorCollection;
  iEvent.getByToken(RecoToCpAssociatorToken_, RecoToCpAssociatorCollection);
  if (!RecoToCpAssociatorCollection.isValid()) {
    edm::LogPrint("PFTester") << "Input ClusterCaloParticleAssociator RecoToSim collection not found.";
    return;
  }
  auto recoToCpAssoc = *RecoToCpAssociatorCollection;

  // --------------------------------------------------------------------
  // ----- Calo Particles plots -----------------------------------------
  // --------------------------------------------------------------------

  std::unordered_map<uint, double> simClusterToCPEnergyMap;
  for (unsigned int cpId = 0; cpId < caloParticles.size(); ++cpId) {
    // Fill map: for each simCluster, the energy of the caloParticle computed as the sum of all simClusters arising from it
    double energySumSimClusters = 0;
    double energySumSimHits = 0;
    double energyFracSumSimHits = 0;

    for (const auto& scRef : caloParticles[cpId].simClusters()) {
      auto const& sc = *(scRef);
      // Compute energy of caloParticle as sum of simClusters energies (from SimTrack energy)
      energySumSimClusters += sc.energy();
      // Compute energy of caloParticle as sum of all hits from all simClusters
      for (auto hit_energy : sc.hits_and_energies()) {
        energySumSimHits += hit_energy.second;
      }
      // Compute energy of caloParticle as sum of all rechits energy multiplied by sim fraction from all simClusters
      for (auto hit_fraction : sc.hits_and_fractions()) {
        DetId id(hit_fraction.first);
        auto rechitIt =
            std::find_if(pfRechit.begin(), pfRechit.end(), [id](const reco::PFRecHit& rh) { return rh.detId() == id; });
        if (rechitIt == pfRechit.end()) {
          continue;
        } else {
          energyFracSumSimHits += rechitIt->energy() * hit_fraction.second;
        }
      }
    }
    for (const auto& scRef : caloParticles[cpId].simClusters()) {
      simClusterToCPEnergyMap[scRef.key()] = energySumSimHits;
    }
#ifdef debug
    edm::LogPrint("PFTester") << " caloParticle [" << cpId << "]: energy=" << caloParticles[cpId].energy()
                              << ", energySumSimClusters=" << energySumSimClusters
                              << ", energySumSimHits=" << energySumSimHits
                              << ", energyFracSumSimHits=" << energyFracSumSimHits << std::endl;
#endif

    h_CPToSCEnergyFraction_->Fill(energySumSimClusters / caloParticles[cpId].energy());
    h_CPToSHEnergyFraction_->Fill(energySumSimHits / caloParticles[cpId].energy());

    // SimToReco association for caloParticles
    const edm::Ref<CaloParticleCollection> caloParticleRef(CaloParticle, cpId);
    const auto& cpToRecoIt = cpToRecoAssoc.find(caloParticleRef);
    if (cpToRecoIt == cpToRecoAssoc.end())
      continue;
    const auto& cpToRecoMatched = cpToRecoIt->val;
    if (cpToRecoMatched.empty())
      continue;

    for (const auto& recoPair : cpToRecoMatched) {
#ifdef debug
      edm::LogPrint("PFTester") << " caloParticle [" << cpId << "] matched to RecoCluster [" << recoPair.first.index()
                                << "] with shared energy: " << recoPair.second.first
                                << ", shared energy fraction: " << recoPair.second.first / energyFracSumSimHits
                                << ", score: " << recoPair.second.second << std::endl;
#endif
      h_CP_simToRecoScore_->Fill(recoPair.second.second);
      h_CP_simToRecoShEnF_->Fill(recoPair.second.first / energyFracSumSimHits);
      h_CP_simToRecoShEnF_Score_->Fill(recoPair.second.first / energyFracSumSimHits, recoPair.second.second);
    }
  }

  // RecoToSim association for caloParticles
  for (unsigned int recoId = 0; recoId < recoClusters.size(); ++recoId) {
    const edm::Ref<RecoClusterCollection> recoClusterRef(RecoCluster, recoId);
    const auto& recoToCpIt = recoToCpAssoc.find(recoClusterRef);
    if (recoToCpIt == recoToCpAssoc.end())
      continue;
    const auto& recoToCpMatched = recoToCpIt->val;
    if (recoToCpMatched.empty())
      continue;
    for (const auto& cpPair : recoToCpMatched) {
      h_CP_recoToSimScore_->Fill(cpPair.second);
    }
  }

  // --------------------------------------------------------------------
  // ----- Efficiency and split computation at cluster level ------------
  // --------------------------------------------------------------------

  uint nSimClusters = 0;
  uint nSimClustersPrimary = 0;
  for (unsigned int simId = 0; simId < simClusters.size(); ++simId) {
    double energySumSimHits = 0;
    for (auto hit_energy : simClusters[simId].hits_and_energies()) {
      energySumSimHits += hit_energy.second;
    }
    h_SimTrackToSimHitsEnergyFraction_->Fill(energySumSimHits / simClusters[simId].energy());

    double energyFracSumSimHits = 0;
    for (auto hit_energy : simClusters[simId].hits_and_fractions()) {
      DetId id(hit_energy.first);
      auto rechitIt =
          std::find_if(pfRechit.begin(), pfRechit.end(), [id](const reco::PFRecHit& rh) { return rh.detId() == id; });
      if (rechitIt == pfRechit.end()) {
        continue;
      } else {
        energyFracSumSimHits += rechitIt->energy() * hit_energy.second;
      }
    }

    // apply cut on energy fraction (sim cluster energy wrt all sim clusters from same calo particle)
    double SimClusterToCPEnergyFraction = energySumSimHits / simClusterToCPEnergyMap[simId];
    if (SimClusterToCPEnergyFraction < enFracCut_)
      continue;
    // apply cut on pt of the sim track
    if (simClusters[simId].pt() < ptCut_)
      continue;

    // filter all sim clusters produced by a sim track which crossed the
    // tracker/calorimeter boundary outside the barrel
    auto const scTrack = simClusters[simId].g4Tracks()[0];
    const math::XYZTLorentzVectorF& pos = scTrack.getPositionAtBoundary();
    auto const simTrackEtaAtBoundary = pos.Eta();
    if (abs(simTrackEtaAtBoundary) > etaCut_)  // simTrack does not cross the barrel
      continue;

    ++nSimClusters;
    if (simClusters[simId].g4Tracks()[0].isPrimary())
      ++nSimClustersPrimary;

    // efficiency and split denominator
    h_simClusters_["En"]->Fill(energySumSimHits);
    h_simClusters_["EnFrac"]->Fill(SimClusterToCPEnergyFraction);
    h_simClusters_["EnSimTrack"]->Fill(simClusters[simId].energy());
    h_simClusters_["Pt"]->Fill(simClusters[simId].pt());
    h_simClusters_["PtLow"]->Fill(simClusters[simId].pt());
    h_simClusters_["Eta"]->Fill(simTrackEtaAtBoundary);
    h_simClusters_["Phi"]->Fill(simClusters[simId].phi());
    h_simClusters_["Mult"]->Fill(simClusters[simId].numberOfRecHits());

    h2d_simClusters_["En_Eta"]->Fill(energySumSimHits, simTrackEtaAtBoundary);
    h2d_simClusters_["En_Phi"]->Fill(energySumSimHits, simClusters[simId].phi());
    h2d_simClusters_["En_Mult"]->Fill(energySumSimHits, simClusters[simId].numberOfRecHits());
    h2d_simClusters_["EnFrac_Eta"]->Fill(SimClusterToCPEnergyFraction, simTrackEtaAtBoundary);
    h2d_simClusters_["EnFrac_Phi"]->Fill(SimClusterToCPEnergyFraction, simClusters[simId].phi());
    h2d_simClusters_["EnFrac_Mult"]->Fill(SimClusterToCPEnergyFraction, simClusters[simId].numberOfRecHits());
    h2d_simClusters_["EnSimTrack_Eta"]->Fill(simClusters[simId].energy(), simTrackEtaAtBoundary);
    h2d_simClusters_["EnSimTrack_Phi"]->Fill(simClusters[simId].energy(), simClusters[simId].phi());
    h2d_simClusters_["EnSimTrack_Mult"]->Fill(simClusters[simId].energy(), simClusters[simId].numberOfRecHits());
    h2d_simClusters_["Pt_Eta"]->Fill(simClusters[simId].pt(), simTrackEtaAtBoundary);
    h2d_simClusters_["Pt_Phi"]->Fill(simClusters[simId].pt(), simClusters[simId].phi());
    h2d_simClusters_["Pt_Mult"]->Fill(simClusters[simId].pt(), simClusters[simId].numberOfRecHits());
    h2d_simClusters_["Mult_Eta"]->Fill(simClusters[simId].numberOfRecHits(), simTrackEtaAtBoundary);
    h2d_simClusters_["Mult_Phi"]->Fill(simClusters[simId].numberOfRecHits(), simClusters[simId].phi());

    const edm::Ref<SimClusterCollection> simClusterRef(SimCluster, simId);
    const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
    if (simToRecoIt == simToRecoAssoc.end())
      continue;
    const auto& simToRecoMatched = simToRecoIt->val;
    if (simToRecoMatched.empty())
      continue;

    for (unsigned ithr = 0; ithr < nAssocScoreThresholds_; ++ithr) {
      const double& thresh = assocScoreThresholds_[ithr];

      unsigned nRecoMatchedToOneSim = 0;
      for (const auto& recoPair : simToRecoMatched) {
#ifdef debug
        const CaloGeometry& caloGeom = iSetup.getData(geometry_token_);

        auto ev = simClusters[simId].g4Tracks()[0].eventId().event();
        auto bx = simClusters[simId].g4Tracks()[0].eventId().bunchCrossing();
        edm::LogPrint("PFTester") << "  SimCluster[" << simId << "], ev=" << ev << ", bx=" << bx
                                  << ", en=" << energySumSimHits << ", hits=";
        const auto& hits_fractions = simClusters[simId].hits_and_fractions();
        const auto& hits_energies = simClusters[simId].hits_and_energies();

        auto itF = hits_fractions.begin();
        auto itE = hits_energies.begin();
        for (; itF != hits_fractions.end() && itE != hits_energies.end(); ++itF, ++itE) {
          DetId id(itF->first);
          const GlobalPoint pos = caloGeom.getPosition(id);
          edm::LogPrint("PFTester") << "    DetId=" << itF->first << ", eta=" << pos.eta() << ", phi=" << pos.phi()
                                    << ", en=" << itE->second << ", fr=" << itF->second;
        }
        edm::LogPrint("PFTester") << "   Matched to RecoCluster[" << recoPair.first.index()
                                  << "], en=" << recoClusters[recoPair.first.index()].energy()
                                  << ", with shared energy: " << recoPair.second.first
                                  << ", shared energy fraction: " << recoPair.second.first / energyFracSumSimHits
                                  << ", score: " << recoPair.second.second << ", hits=";
        for (auto const& hit_energy : recoClusters[recoPair.first.index()].recHitFractions()) {
          DetId id(hit_energy.recHitRef()->detId());
          const GlobalPoint pos = caloGeom.getPosition(id);
          edm::LogPrint("PFTester") << "     DetId=" << hit_energy.recHitRef()->detId() << ", eta=" << pos.eta()
                                    << ", phi=" << pos.phi() << ", en=" << hit_energy.recHitRef()->energy()
                                    << ", fr=" << hit_energy.fraction();
        }
#endif

        auto score = recoPair.second.second;
        auto shared_energy = recoPair.second.first;
        auto shared_energy_frac = shared_energy / energyFracSumSimHits;

        h_simToRecoScore_->Fill(score);
        h_simToRecoShEnF_->Fill(shared_energy_frac);
        h_simToRecoShEnF_Score_->Fill(shared_energy_frac, score);
        h_simToRecoShEnF_En_->Fill(shared_energy_frac, energySumSimHits);
        h_simToRecoShEnF_EnFrac_->Fill(shared_energy_frac, SimClusterToCPEnergyFraction);
        h_simToRecoShEnF_EnSimTrack_->Fill(shared_energy_frac, simClusters[simId].energy());
        h_simToRecoShEnF_Mult_->Fill(shared_energy_frac, simClusters[simId].numberOfRecHits());
        h_simToRecoScore_En_->Fill(score, energySumSimHits);
        h_simToRecoScore_EnFrac_->Fill(score, SimClusterToCPEnergyFraction);
        h_simToRecoScore_EnSimTrack_->Fill(score, simClusters[simId].energy());
        h_simToRecoScore_Mult_->Fill(score, simClusters[simId].numberOfRecHits());

        if (doMatchByScore_) {
          // cut on score
          if (score < thresh) {
            nRecoMatchedToOneSim++;
          }
        } else {
          // cut on shared energy fraction
          if (shared_energy_frac > thresh) {
            nRecoMatchedToOneSim++;
          }
        }
      }

      // efficiency numerator
      if (nRecoMatchedToOneSim > 0) {
        h_simClustersMatchedRecoClusters_[ithr]["En"]->Fill(energySumSimHits);
        h_simClustersMatchedRecoClusters_[ithr]["EnFrac"]->Fill(SimClusterToCPEnergyFraction);
        h_simClustersMatchedRecoClusters_[ithr]["EnSimTrack"]->Fill(simClusters[simId].energy());
        h_simClustersMatchedRecoClusters_[ithr]["Pt"]->Fill(simClusters[simId].pt());
        h_simClustersMatchedRecoClusters_[ithr]["PtLow"]->Fill(simClusters[simId].pt());
        h_simClustersMatchedRecoClusters_[ithr]["Eta"]->Fill(simTrackEtaAtBoundary);
        h_simClustersMatchedRecoClusters_[ithr]["Phi"]->Fill(simClusters[simId].phi());
        h_simClustersMatchedRecoClusters_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits());

        h2d_simClustersMatchedRecoClusters_[ithr]["En_Eta"]->Fill(energySumSimHits, simTrackEtaAtBoundary);
        h2d_simClustersMatchedRecoClusters_[ithr]["En_Phi"]->Fill(energySumSimHits, simClusters[simId].phi());
        h2d_simClustersMatchedRecoClusters_[ithr]["En_Mult"]->Fill(energySumSimHits,
                                                                       simClusters[simId].numberOfRecHits());
        h2d_simClustersMatchedRecoClusters_[ithr]["EnFrac_Eta"]->Fill(SimClusterToCPEnergyFraction,
                                                                      simTrackEtaAtBoundary);
        h2d_simClustersMatchedRecoClusters_[ithr]["EnFrac_Phi"]->Fill(SimClusterToCPEnergyFraction,
                                                                      simClusters[simId].phi());
        h2d_simClustersMatchedRecoClusters_[ithr]["EnFrac_Mult"]->Fill(SimClusterToCPEnergyFraction,
                                                                       simClusters[simId].numberOfRecHits());
        h2d_simClustersMatchedRecoClusters_[ithr]["EnSimTrack_Eta"]->Fill(simClusters[simId].energy(), simTrackEtaAtBoundary);
        h2d_simClustersMatchedRecoClusters_[ithr]["EnSimTrack_Phi"]->Fill(simClusters[simId].energy(),
                                                                  simClusters[simId].phi());
        h2d_simClustersMatchedRecoClusters_[ithr]["EnSimTrack_Mult"]->Fill(simClusters[simId].energy(),
                                                                   simClusters[simId].numberOfRecHits());
        h2d_simClustersMatchedRecoClusters_[ithr]["Pt_Eta"]->Fill(simClusters[simId].pt(), simTrackEtaAtBoundary);
        h2d_simClustersMatchedRecoClusters_[ithr]["Pt_Phi"]->Fill(simClusters[simId].pt(), simClusters[simId].phi());
        h2d_simClustersMatchedRecoClusters_[ithr]["Pt_Mult"]->Fill(simClusters[simId].pt(),
                                                                   simClusters[simId].numberOfRecHits());
        h2d_simClustersMatchedRecoClusters_[ithr]["Mult_Eta"]->Fill(simClusters[simId].numberOfRecHits(),
                                                                    simTrackEtaAtBoundary);
        h2d_simClustersMatchedRecoClusters_[ithr]["Mult_Phi"]->Fill(simClusters[simId].numberOfRecHits(),
                                                                    simClusters[simId].phi());

        // split numerator
        if (nRecoMatchedToOneSim > 1) {
          h_simClustersMultiMatchedRecoClusters_[ithr]["En"]->Fill(energySumSimHits);
          h_simClustersMultiMatchedRecoClusters_[ithr]["EnFrac"]->Fill(SimClusterToCPEnergyFraction);
          h_simClustersMultiMatchedRecoClusters_[ithr]["EnSimTrack"]->Fill(simClusters[simId].energy());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Pt"]->Fill(simClusters[simId].pt());
          h_simClustersMultiMatchedRecoClusters_[ithr]["PtLow"]->Fill(simClusters[simId].pt());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Eta"]->Fill(simTrackEtaAtBoundary);
          h_simClustersMultiMatchedRecoClusters_[ithr]["Phi"]->Fill(simClusters[simId].phi());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits());
        }
      }

      h_nRecoMatchedToOneSim_[ithr]->Fill(nRecoMatchedToOneSim);
    }
  }

  h_nSimClusters_->Fill(nSimClusters);
  h_nSimClustersPrimary_->Fill(nSimClustersPrimary);

  // --------------------------------------------------------------------
  // ----- Fakes and merge computation at cluster level -----------------
  // --------------------------------------------------------------------

  h_nPFClusters_->Fill(recoClusters.size());
  for (unsigned int recoId = 0; recoId < recoClusters.size(); ++recoId) {
    // fake and merge denominator
    h_recoClusters_["En"]->Fill(recoClusters[recoId].energy());
    // h_recoClusters_["Pt"]->Fill(recoClusters[recoId].pt());
    // h_recoClusters_["PtLow"]->Fill(recoClusters[recoId].pt());
    h_recoClusters_["Eta"]->Fill(recoClusters[recoId].eta());
    h_recoClusters_["Phi"]->Fill(recoClusters[recoId].phi());
    h_recoClusters_["Mult"]->Fill(recoClusters[recoId].size());

    h2d_recoClusters_["En_Eta"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].eta());
    h2d_recoClusters_["En_Phi"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].phi());
    h2d_recoClusters_["En_Mult"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].size());
    // h2d_recoClusters_["Pt_Eta"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].eta());
    // h2d_recoClusters_["Pt_Phi"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].phi());
    // h2d_recoClusters_["Pt_Mult"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].size());
    h2d_recoClusters_["Mult_Eta"]->Fill(recoClusters[recoId].size(), recoClusters[recoId].eta());
    h2d_recoClusters_["Mult_Phi"]->Fill(recoClusters[recoId].size(), recoClusters[recoId].phi());

    const edm::Ref<RecoClusterCollection> recoClusterRef(RecoCluster, recoId);
    const auto& recoToSimIt = recoToSimAssoc.find(recoClusterRef);
    if (recoToSimIt == recoToSimAssoc.end())
      continue;
    const auto& recoToSimMatched = recoToSimIt->val;
    if (recoToSimMatched.empty())
      continue;

    for (unsigned ithr = 0; ithr < nAssocScoreThresholds_; ++ithr) {
      const double& thresh = assocScoreThresholds_[ithr];

      unsigned nSimMatchedToOneReco = 0;

      for (const auto& simPair : recoToSimMatched) {
        const auto simPairIdx = simPair.first.index();

#ifdef debug
        edm::LogPrint("PFTester") << " recoToSimAssoc recoCluster id " << recoId
                                  << " : matched simCluster id = " << simPairIdx << " score = " << simPair.second
                                  << std::endl;
#endif

        double energySumSimHits = 0;
        for (auto hit_energy : simClusters[simPairIdx].hits_and_energies()) {
          energySumSimHits += hit_energy.second;
        }

        // apply cut on energy fraction (sim cluster energy wrt all sim clusters from same calo particle)
        double SimClusterToCPEnergyFraction = energySumSimHits / simClusterToCPEnergyMap[simPairIdx];
        if (SimClusterToCPEnergyFraction < enFracCut_)
          continue;
        // apply cut on pt of the sim track
        if (simClusters[simPairIdx].pt() < ptCut_)
          continue;

        // filter all sim clusters produced by a sim track which crossed the
        // tracker/calorimeter boundary outside the barrel
        auto const scTrack = simClusters[simPairIdx].g4Tracks()[0];
        const math::XYZTLorentzVectorF pos = scTrack.getPositionAtBoundary();
        if (abs(pos.Eta()) > etaCut_)  // simTrack does not cross the barrel
          continue;

        h_recoToSimScore_->Fill(simPair.second);

        // cut on score
        if (simPair.second < thresh) {
          nSimMatchedToOneReco++;
        }
      }

      // fake numerator
      if (nSimMatchedToOneReco > 0) {
        h_recoClustersMatchedSimClusters_[ithr]["En"]->Fill(recoClusters[recoId].energy());
        // h_recoClustersMatchedSimClusters_[ithr]["Pt"]->Fill(recoClusters[recoId].pt());
        // h_recoClustersMatchedSimClusters_[ithr]["PtLow"]->Fill(recoClusters[recoId].pt());
        h_recoClustersMatchedSimClusters_[ithr]["Eta"]->Fill(recoClusters[recoId].eta());
        h_recoClustersMatchedSimClusters_[ithr]["Phi"]->Fill(recoClusters[recoId].phi());
        h_recoClustersMatchedSimClusters_[ithr]["Mult"]->Fill(recoClusters[recoId].size());

        h2d_recoClustersMatchedSimClusters_[ithr]["En_Eta"]->Fill(recoClusters[recoId].energy(),
                                                                  recoClusters[recoId].eta());
        h2d_recoClustersMatchedSimClusters_[ithr]["En_Phi"]->Fill(recoClusters[recoId].energy(),
                                                                  recoClusters[recoId].phi());
        h2d_recoClustersMatchedSimClusters_[ithr]["En_Mult"]->Fill(recoClusters[recoId].energy(),
                                                                   recoClusters[recoId].size());
        // h2d_recoClustersMatchedSimClusters_[ithr]["Pt_Eta"]->Fill(recoClusters[recoId].pt(),
        //                                                           recoClusters[recoId].eta());
        // h2d_recoClustersMatchedSimClusters_[ithr]["Pt_Phi"]->Fill(recoClusters[recoId].pt(),
        //                                                           recoClusters[recoId].phi());
        // h2d_recoClustersMatchedSimClusters_[ithr]["Pt_Mult"]->Fill(recoClusters[recoId].pt(),
        //                                                            recoClusters[recoId].size());
        h2d_recoClustersMatchedSimClusters_[ithr]["Mult_Eta"]->Fill(recoClusters[recoId].size(),
                                                                    recoClusters[recoId].eta());
        h2d_recoClustersMatchedSimClusters_[ithr]["Mult_Phi"]->Fill(recoClusters[recoId].size(),
                                                                    recoClusters[recoId].phi());

        // merge numerator
        if (nSimMatchedToOneReco > 1) {
          h_recoClustersMultiMatchedSimClusters_[ithr]["En"]->Fill(recoClusters[recoId].energy());
          // h_recoClustersMultiMatchedSimClusters_[ithr]["Pt"]->Fill(recoClusters[recoId].pt());
          // h_recoClustersMultiMatchedSimClusters_[ithr]["PtLow"]->Fill(recoClusters[recoId].pt());
          h_recoClustersMultiMatchedSimClusters_[ithr]["Eta"]->Fill(recoClusters[recoId].eta());
          h_recoClustersMultiMatchedSimClusters_[ithr]["Phi"]->Fill(recoClusters[recoId].phi());
          h_recoClustersMultiMatchedSimClusters_[ithr]["Mult"]->Fill(recoClusters[recoId].size());
        }
      }

      h_nSimMatchedToOneReco_[ithr]->Fill(nSimMatchedToOneReco);
    }
  }

  // --------------------------------------------------------------------
  // ----- Cluster response computation ---------------------------------
  // --------------------------------------------------------------------
  // std::cout << std::endl;
  // std::cout << "--- Event " << iEvent.eventAuxiliary().event() << " ---" << std::endl;
  for (unsigned int simId = 0; simId < simClusters.size(); ++simId) {
    double energySumSimHits = 0;
    for (auto hit_energy : simClusters[simId].hits_and_energies()) {
      energySumSimHits += hit_energy.second;
    }
    // apply cut on energy fraction (sim cluster energy wrt all sim clusters from same calo particle)
    double SimClusterToCPEnergyFraction = energySumSimHits / simClusterToCPEnergyMap[simId];
    if (SimClusterToCPEnergyFraction < enFracCut_)
      continue;
    // apply cut on pt of the sim track
    if (simClusters[simId].pt() < ptCut_)
      continue;

    // filter all sim clusters produced by a sim track which crossed the
    // tracker/calorimeter boundary outside the barrel
    auto const scTrack = simClusters[simId].g4Tracks()[0];
    const math::XYZTLorentzVectorF& pos = scTrack.getPositionAtBoundary();
    auto const simTrackEtaAtBoundary = pos.Eta();
    if (abs(simTrackEtaAtBoundary) > etaCut_)  // simTrack does not cross the barrel
      continue;

    const edm::Ref<SimClusterCollection> simClusterRef(SimCluster, simId);
    const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
    if (simToRecoIt == simToRecoAssoc.end())
      continue;
    const auto& simToRecoMatched = simToRecoIt->val;
    if (simToRecoMatched.empty())
      continue;

    // they should already be sorted by score
    std::vector simToRecoMatchedSorted(simToRecoMatched.begin(), simToRecoMatched.end());
    std::sort(simToRecoMatchedSorted.begin(), simToRecoMatchedSorted.end(), [](const auto& a, const auto& b) {
      return a.second.second < b.second.second;
    });

    for (unsigned ithr = 0; ithr < nAssocScoreThresholds_; ++ithr) {
      const double& thresh = assocScoreThresholds_[ithr];

      // fill only the best matched (lowest score) reco cluster, regardless split or merge
      for (const auto& recoPair : simToRecoMatchedSorted) {
        auto recoId = recoPair.first.index();
		
        bool passMatch = false;
        if (doMatchByScore_) {
          // cut on score
          passMatch = recoPair.second.second < thresh;
        } else {
          // cut on shared energy fraction
          double shared_energy = recoPair.second.first;
          double shared_energy_frac = shared_energy / energySumSimHits;
          passMatch = shared_energy_frac > thresh;
        }

		// std::cout << "===============================" << std::endl;
		// std::cout << "matchByScore? " << doMatchByScore_ << std::endl;
		// std::cout << "passMatch: " << passMatch << ", recoId: " << recoId << std::endl;
		// std::cout << "sim en: " << energySumSimHits << ", reco en: " << recoClusters[recoId].energy() << std::endl;
		// std::cout << "sim eta: " << simClusters[simId].eta() << ", reco eta: " << recoClusters[recoId].eta()  << ", sim track eta: " << simTrackEtaAtBoundary << std::endl;
		// std::cout << "sim phi: " << simClusters[simId].phi() << ", reco phi: " << recoClusters[recoId].phi() << std::endl;
		// std::cout << "score: " << recoPair.second.second << std::endl;
		// std::cout << "shared en frac: " << recoPair.second.first / energySumSimHits << std::endl;
		// std::cout << "n sim clusters: " << simClusters.size() << std::endl;
		// std::cout << "n matched reco clusters: " << simToRecoMatchedSorted.size() << std::endl;
		// for (const auto& recoPairDebug : simToRecoMatchedSorted) {
		//   std::cout << "- score: " << recoPairDebug.second.second << ", share en frac: " << recoPairDebug.second.first / energySumSimHits << ", en: " << recoClusters[recoPairDebug.first.index()].energy() << std::endl;
		// }
		// std::cout << "threshold: " << thresh << std::endl;
		// std::cout << "===============================" << std::endl;
		
        if (passMatch) {
          // h2d_responsePt_[ithr]["En"]->Fill(energySumSimHits, 
          //                                   recoClusters[recoId].pt() / simClusters[simId].pt());
          // h2d_responsePt_[ithr]["EnFrac"]->Fill(SimClusterToCPEnergyFraction,
          //                                       recoClusters[recoId].pt() / simClusters[simId].pt());
          // h2d_responsePt_[ithr]["EnSimTrack"]->Fill(simClusters[simId].energy(),
          //                                   recoClusters[recoId].pt() / simClusters[simId].pt());
          // h2d_responsePt_[ithr]["Pt"]->Fill(simClusters[simId].pt(),
          //                                   recoClusters[recoId].pt() / simClusters[simId].pt());
          // h2d_responsePt_[ithr]["Eta"]->Fill(simTrackEtaAtBoundary,
          //                                    recoClusters[recoId].pt() / simClusters[simId].pt());
          // h2d_responsePt_[ithr]["Phi"]->Fill(simClusters[simId].phi(),
          //                                    recoClusters[recoId].pt() / simClusters[simId].pt());
          // h2d_responsePt_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits(),
          //                                     recoClusters[recoId].pt() / simClusters[simId].pt());

          h2d_responseE_[ithr]["En"]->Fill(energySumSimHits, 
                                           recoClusters[recoId].energy() / energySumSimHits);
          h2d_responseE_[ithr]["EnFrac"]->Fill(SimClusterToCPEnergyFraction,
                                               recoClusters[recoId].energy() / energySumSimHits);
          h2d_responseE_[ithr]["EnSimTrack"]->Fill(simClusters[simId].energy(),
                                           recoClusters[recoId].energy() / energySumSimHits);
          h2d_responseE_[ithr]["Pt"]->Fill(simClusters[simId].pt(), recoClusters[recoId].energy() / energySumSimHits);
          h2d_responseE_[ithr]["Eta"]->Fill(simTrackEtaAtBoundary, recoClusters[recoId].energy() / energySumSimHits);
          h2d_responseE_[ithr]["Phi"]->Fill(simClusters[simId].phi(), recoClusters[recoId].energy() / energySumSimHits);
          h2d_responseE_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits(),
                                             recoClusters[recoId].energy() / energySumSimHits);
		  // std::cout << "fill response: " << recoClusters[recoId].energy() / energySumSimHits << std::endl;
		  // std::cout << "============== break =================" << std::endl;
          break;
        }
      }
    }
  }

  // --------------------------------------------------------------------
  // ---------------- PF Candidates -------------------------------------
  // --------------------------------------------------------------------

  const reco::PFCandidateCollection* pf_candidates;
  edm::Handle<reco::PFCandidateCollection> PFCand;
  iEvent.getByToken(PFCandToken_, PFCand);
  if (!PFCand.isValid()) {
    edm::LogPrint("PFTester") << "Input PFCand collection not found.";
    return;
  }

  pf_candidates = PFCand.product();
  if (!pf_candidates) {
    edm::LogPrint("PFTester") << " Failed to retrieve data required by PFTester.cc";
    return;
  }

  // --------------------------------------------------------------------
  // -------------------- PF Blocks and Elements ------------------------
  // --------------------------------------------------------------------

  // Loop Over Particle Flow Candidates
  for (size_t i = 0; i < pf_candidates->size(); ++i) {
    const auto& particle = (*pf_candidates)[i];

    h_PFCandEt_->Fill(particle.et());
    h_PFCandEta_->Fill(particle.eta());
    h_PFCandPhi_->Fill(particle.phi());
    h_PFCandCharge_->Fill(particle.charge());
    h_PFCandPdgId_->Fill(particle.pdgId());
    h_PFCandType_->Fill(particle.particleId());

    // Get the PFBlock and Elements
    const reco::PFCandidate::ElementsInBlocks& elementsInBlocks = particle.elementsInBlocks();
    int numElements = elementsInBlocks.size();
    int numTrackElements = 0;
    int numMuonElements = 0;
    int numPS1Elements = 0;
    int numPS2Elements = 0;
    int numECALElements = 0;
    int numHCALElements = 0;
    int numHGCALElements = 0;
    int numPFClusters = 0;

    // Loop over Elements in Block
    for (const auto& elemBlockPair : elementsInBlocks) {
      reco::PFBlockRef blockRef = elemBlockPair.first;
      unsigned elementIndex = elemBlockPair.second;
      const reco::PFBlockElement& element = blockRef->elements()[elementIndex];
      int element_type = element.type();

      // Element is a Tracker Track
      if (element_type == reco::PFBlockElement::TRACK) {
        // Get General Information about the Track
        reco::PFRecTrack track = *(element.trackRefPF());
        h_TrackCharge_->Fill(track.charge());
        h_TrackNumPoints_->Fill(track.nTrajectoryPoints());
        h_TrackNumMeasurements_->Fill(track.nTrajectoryMeasurements());

        // Loop Over Points in the Track
        std::vector<reco::PFTrajectoryPoint> points = track.trajectoryPoints();
        std::vector<reco::PFTrajectoryPoint>::iterator point;
        for (point = points.begin(); point != points.end(); point++) {
          int point_layer = point->layer();
          double x = point->position().x();
          double y = point->position().y();
          double z = point->position().z();
          if (point_layer == reco::PFTrajectoryPoint::ClosestApproach) {
            h_TrackImpactParameter_->Fill(sqrt(x * x + y * y + z * z));  // [FIXME]
          }
        }
        numTrackElements++;
      } else if (element_type == reco::PFBlockElement::MUON) {
        numMuonElements++;  // Element is a Muon Track
      } else {
        if (element_type == reco::PFBlockElement::PS1)
          numPS1Elements++;  // Element is a PreShower1 Cluster
        if (element_type == reco::PFBlockElement::PS2)
          numPS2Elements++;  // Element is a PreShower2 Cluster
        if (element_type == reco::PFBlockElement::ECAL)
          numECALElements++;  // Element is an ECAL Cluster
        if (element_type == reco::PFBlockElement::HCAL)
          numHCALElements++;  // Element is a HCAL Cluster
        if (element_type == reco::PFBlockElement::HGCAL)
          numHGCALElements++;  // Element is a HGCAL Cluster

        if (element.clusterRef().isNonnull()) {
          auto const& cluster = *(element.clusterRef());
          numPFClusters++;
          h_PFClusterE_->Fill(cluster.energy());
          h_PFClusterEta_->Fill(cluster.eta());
          h_PFClusterPhi_->Fill(cluster.phi());
          h_PFClusterDepth_->Fill(cluster.depth());
          h_PFClusterNHits_->Fill(cluster.recHitFractions().size());
          h_PFClusterType_->Fill(element_type);
          for (const auto& hitFracPair : cluster.hitsAndFractions()) {
            DetId hitId = hitFracPair.first;
            float fraction = hitFracPair.second;
            h_PFClusterHitFraction_->Fill(fraction);
            h_PFClusterHitDetId_->Fill(hitId.rawId() % 10000);  // modulo for visualization
          }
        }
      }
    }

    // Fill the Respective Elements Sizes
    h_NumElements_->Fill(numElements);
    h_NumTrackElements_->Fill(numTrackElements);
    h_NumMuonElements_->Fill(numMuonElements);
    h_NumPS1Elements_->Fill(numPS1Elements);
    h_NumPS2Elements_->Fill(numPS2Elements);
    h_NumECALElements_->Fill(numECALElements);
    h_NumHCALElements_->Fill(numHCALElements);
    h_NumHGCALElements_->Fill(numHGCALElements);
  }
}

// convert a double to the string format
template <typename RecoClusterCollection>
std::string PFTesterT<RecoClusterCollection>::doubleToString(double x) const {
  std::ostringstream result;
  result << std::setprecision(2) << x;

  std::string xnew = result.str();
  std::size_t pos = xnew.find('.');
  if (pos != std::string::npos)
    xnew.replace(pos, 1, "p");
  else  //if the double was provided without decimal places
    xnew += "p0";

  return xnew;
}

using PFClusterTester = PFTesterT<reco::PFClusterCollection>;
using CaloClusterTester = PFTesterT<reco::CaloClusterCollection>;

DEFINE_FWK_MODULE(PFClusterTester);
DEFINE_FWK_MODULE(CaloClusterTester);
