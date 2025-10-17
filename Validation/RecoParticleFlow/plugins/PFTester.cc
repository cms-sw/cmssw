// author: Mike Schmitt, University of Florida
// first version 11/7/2007

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
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

class PFTester : public DQMEDAnalyzer {
public:
  explicit PFTester(const edm::ParameterSet&);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::string doubleToString(double x) const;

  edm::EDGetTokenT<reco::PFCandidateCollection> PFCandToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterToken_;
  edm::EDGetTokenT<SimClusterCollection> SimClusterToken_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionWithSimClustersT<reco::PFClusterCollection>> RecoToSimAssociatorToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionWithSimClustersT<reco::PFClusterCollection>> SimToRecoAssociatorToken_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionT<reco::PFClusterCollection>> RecoToCpAssociatorToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionT<reco::PFClusterCollection>> CpToRecoAssociatorToken_;

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

  MonitorElement* h_NumPFClusters_;
  MonitorElement* h_PFClusterE_;
  MonitorElement* h_PFClusterEta_;
  MonitorElement* h_PFClusterPhi_;
  MonitorElement* h_PFClusterDepth_;
  MonitorElement* h_PFClusterNHits_;
  MonitorElement* h_PFClusterType_;
  MonitorElement* h_PFClusterHitFraction_;
  MonitorElement* h_PFClusterHitDetId_;

  MonitorElement* h_simToRecoScore_;
  MonitorElement* h_recoToSimScore_;

  std::vector<double> assocScoreThresholds_;

  const std::unordered_map<std::string, std::tuple<unsigned, float, float>> histoVars = {
      {"En", std::make_tuple(100, 0., 50.)},
      {"Pt", std::make_tuple(200, 0., 100.)},
	  {"PtLow", std::make_tuple(100, 0., 10.)},
      {"Eta", std::make_tuple(50, -6.5, 6.5)},
      {"Phi", std::make_tuple(50, -3.5, 3.5)},
      {"Mult", std::make_tuple(200, 0., 200.)},
  };

  using UMap = std::unordered_map<std::string, MonitorElement*>;
  using VUMap = std::vector<UMap>;
  UMap h_simClusters_;
  UMap h_simClustersReconstructable_;
  VUMap h_simClustersMatchedRecoClusters_{histoVars.size()};
  VUMap h_simClustersMultiMatchedRecoClusters_{histoVars.size()};
  UMap h_recoClusters_;
  UMap h_recoClustersReconstructable_;
  VUMap h_recoClustersMatchedSimClusters_{histoVars.size()};
  VUMap h_recoClustersMultiMatchedSimClusters_{histoVars.size()};

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> histo2dVars = {
      {"En_Eta", std::make_tuple(100, 0., 50., 50, -6.5, 6.5)},
      {"En_Phi", std::make_tuple(100, 0., 50., 50, -3.5, 3.5)},
      {"En_Mult", std::make_tuple(100, 0., 50., 200, 0., 200.)},
      {"Pt_Eta", std::make_tuple(100, 0., 40., 50, -6.5, 6.5)},
      {"Pt_Phi", std::make_tuple(100, 0., 40., 50, -3.5, 3.5)},
      {"Pt_Mult", std::make_tuple(100, 0., 40., 200, 0., 200.)},
      {"Mult_Eta", std::make_tuple(200, 0., 200., 50, -6.5, 6.5)},
      {"Mult_Phi", std::make_tuple(200, 0., 200., 50, -3.5, 3.5)},
  };

  using U2Map = std::unordered_map<std::string, MonitorElement*>;
  using VU2Map = std::vector<std::unordered_map<std::string, MonitorElement*>>;
  U2Map h2d_simClusters_;
  U2Map h2d_simClustersReconstructable_;
  VU2Map h2d_simClustersMatchedRecoClusters_{histo2dVars.size()};
  U2Map h2d_recoClusters_;
  U2Map h2d_recoClustersReconstructable_;
  VU2Map h2d_recoClustersMatchedSimClusters_{histo2dVars.size()};

  VU2Map h2d_response_{histoVars.size()};
};

PFTester::PFTester(const edm::ParameterSet& iConfig)
    : PFCandToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCand"))),
      PFClusterToken_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("PFCluster"))),
      SimClusterToken_(consumes<SimClusterCollection>(iConfig.getParameter<edm::InputTag>("SimCluster"))),
      RecoToSimAssociatorToken_(consumes<ticl::RecoToSimCollectionWithSimClustersT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterSimClusterAssociator"))),
      SimToRecoAssociatorToken_(consumes<ticl::SimToRecoCollectionWithSimClustersT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterSimClusterAssociator"))),
      RecoToCpAssociatorToken_(consumes<ticl::RecoToSimCollectionT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterCaloParticleAssociator"))),
      CpToRecoAssociatorToken_(consumes<ticl::SimToRecoCollectionT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterCaloParticleAssociator"))),
      assocScoreThresholds_(iConfig.getParameter<std::vector<double>>("assocScoreThresholds")) {}

void PFTester::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const&) {
  ibook.setCurrentFolder("HLT/ParticleFlow/PFCandidates");
  h_PFCandEt_ = ibook.book1D("PFCandEt", "PFCandEt", 1000, 0, 1000);
  h_PFCandEta_ = ibook.book1D("PFCandEta", "PFCandEta", 200, -5, 5);
  h_PFCandPhi_ = ibook.book1D("PFCandPhi", "PFCandPhi", 200, -M_PI, M_PI);
  h_PFCandCharge_ = ibook.book1D("PFCandCharge", "PFCandCharge", 5, -2, 2);
  h_PFCandPdgId_ = ibook.book1D("PFCandPdgId", "PFCandPdgId", 44, -22, 22);
  h_PFCandType_ = ibook.book1D("PFCandidateType", "PFCandidateType", 10, 0, 10);

  ibook.setCurrentFolder("HLT/ParticleFlow/PFBlocks");
  h_NumElements_ = ibook.book1D("NumElements", "NumElements", 25, 0, 25);
  h_NumTrackElements_ = ibook.book1D("NumTrackElements", "NumTrackElements", 5, 0, 5);
  h_NumMuonElements_ = ibook.book1D("NumMuonElements", "NumMuonElements", 5, 0, 5);
  h_NumPS1Elements_ = ibook.book1D("NumPS1Elements", "NumPS1Elements", 5, 0, 5);
  h_NumPS2Elements_ = ibook.book1D("NumPS2Elements", "NumPS2Elements", 5, 0, 5);
  h_NumECALElements_ = ibook.book1D("NumECALElements", "NumECALElements", 5, 0, 5);
  h_NumHCALElements_ = ibook.book1D("NumHCALElements", "NumHCALElements", 5, 0, 5);
  h_NumHGCALElements_ = ibook.book1D("NumHGCALElements", "NumHGCALElements", 5, 0, 5);

  ibook.setCurrentFolder("HLT/ParticleFlow/PFTracks");
  h_TrackCharge_ = ibook.book1D("TrackCharge", "TrackCharge", 5, -2, 2);
  h_TrackNumPoints_ = ibook.book1D("TrackNumPoints", "TrackNumPoints", 100, 0, 100);
  h_TrackNumMeasurements_ = ibook.book1D("TrackNumMeasurements", "TrackNumMeasurements", 100, 0, 100);
  h_TrackImpactParameter_ = ibook.book1D("TrackImpactParameter", "TrackImpactParameter", 1000, 0, 1);

  ibook.setCurrentFolder("HLT/ParticleFlow/PFClusters");
  h_NumPFClusters_ = ibook.book1D("NumPFClusters", "Number of PFClusters per PFCandidate", 25, 0, 25);
  h_PFClusterE_ = ibook.book1D("PFClusterE", "PFCluster Energy;E [GeV]", 100, 0, 100);
  h_PFClusterEta_ = ibook.book1D("PFClusterEta", "PFCluster Eta;#eta", 120, -6, 6);
  h_PFClusterPhi_ = ibook.book1D("PFClusterPhi", "PFCluster Phi;#phi", 128, -3.2, 3.2);
  h_PFClusterDepth_ = ibook.book1D("PFClusterDepth", "PFCluster Depth;Depth", 10, 0, 10);
  h_PFClusterNHits_ = ibook.book1D("PFClusterNHits", "PFCluster Number of Hits", 100, 0, 100);
  h_PFClusterType_ = ibook.book1D("PFClusterEtaWidth", "PFCluster Eta Width;#sigma_{#eta}", 20, 0, 20);
  h_PFClusterHitFraction_ = ibook.book1D("PFClusterHitFraction", "PFCluster Hit Fraction;Fraction", 100, 0.0, 1.1);
  h_PFClusterHitDetId_ =
      ibook.book1D("PFClusterHitDetId", "PFCluster Hit DetId modulo 10000;DetId mod 10000", 100, 0, 10000);

  h_simToRecoScore_ = ibook.book1D("simToRecoScore", "simToRecoScore;Sim #rightarrow Reco score", 50, 0, 1);
  h_recoToSimScore_ = ibook.book1D("recoToSimScore", "recoToSimScore;Reco #rightarrow Sim score", 50, 0, 1);

  for (auto& hVar : histoVars) {
    auto [nBins, hMin, hMax] = hVar.second;

    ibook.setCurrentFolder("HLT/ParticleFlow/PFClusterValidation");
    h_simClusters_[hVar.first] =
        ibook.book1D("SimClusters" + hVar.first, "SimClusters;" + hVar.first, nBins, hMin, hMax);
    h_simClustersReconstructable_[hVar.first] = ibook.book1D(
        "SimClustersReconstructable" + hVar.first, "SimClustersReconstructable;" + hVar.first, nBins, hMin, hMax);
    h_recoClusters_[hVar.first] =
        ibook.book1D("RecoClusters" + hVar.first, "RecoClusters;" + hVar.first, nBins, hMin, hMax);
    h_recoClustersReconstructable_[hVar.first] = ibook.book1D(
        "RecoClustersReconstructable" + hVar.first, "RecoClustersReconstructable;" + hVar.first, nBins, hMin, hMax);

    for (unsigned ithr = 0; ithr < assocScoreThresholds_.size(); ++ithr) {
      std::string threshStr = "Score" + doubleToString(assocScoreThresholds_[ithr]);

      ibook.setCurrentFolder("HLT/ParticleFlow/PFClusterValidation/" + threshStr);
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
  }

  for (auto& h2dVar : histo2dVars) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = h2dVar.second;

    ibook.setCurrentFolder("HLT/ParticleFlow/PFClusterValidation");
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
    h2d_simClustersReconstructable_[h2dVar.first] =
        ibook.book2D("SimClustersReconstructable" + h2dVar.first,
                     "SimClustersReconstructable;" + x_title + ";" + y_title,
                     nBinsX,
                     hMinX,
                     hMaxX,
                     nBinsY,
                     hMinY,
                     hMaxY);
    h2d_recoClusters_[h2dVar.first] = ibook.book2D("RecoClusters" + h2dVar.first,
                                                   "RecoClusters;" + x_title + ";" + y_title,
                                                   nBinsX,
                                                   hMinX,
                                                   hMaxX,
                                                   nBinsY,
                                                   hMinY,
                                                   hMaxY);
    h2d_recoClustersReconstructable_[h2dVar.first] =
        ibook.book2D("RecoClustersReconstructable" + h2dVar.first,
                     "RecoClustersReconstructable;" + x_title + ";" + y_title,
                     nBinsX,
                     hMinX,
                     hMaxX,
                     nBinsY,
                     hMinY,
                     hMaxY);

    for (unsigned ithr = 0; ithr < assocScoreThresholds_.size(); ++ithr) {
      std::string threshStr = "Score" + doubleToString(assocScoreThresholds_[ithr]);

      ibook.setCurrentFolder("HLT/ParticleFlow/PFClusterValidation/" + threshStr);
      h2d_simClustersMatchedRecoClusters_[ithr][h2dVar.first] =
          ibook.book2D("SimClustersMatchedRecoClusters" + h2dVar.first,
                       "SimClusters matched to RecoClusters;" + x_title + ";" + y_title,
                       nBinsX,
                       hMinX,
                       hMaxX,
                       nBinsY,
                       hMinY,
                       hMaxY);
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

  for (auto& hVar : histoVars) {
    auto [nBins, hMin, hMax] = hVar.second;
    for (unsigned ithr = 0; ithr < assocScoreThresholds_.size(); ++ithr) {
      std::string threshStr = "Score" + doubleToString(assocScoreThresholds_[ithr]);
      ibook.setCurrentFolder("HLT/ParticleFlow/PFClusterValidation/" + threshStr);
      h2d_response_[ithr][hVar.first] =
          ibook.book2D("Response" + hVar.first, "Response;" + hVar.first, nBins, hMin, hMax, 50, 0., 2.);
    }
  }
}

void PFTester::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  // --------------------------------------------------------------------
  // ---------------- PF Candidates -------------------------------------
  // --------------------------------------------------------------------

  const reco::PFCandidateCollection* pf_candidates;
  edm::Handle<reco::PFCandidateCollection> PFCand;
  iEvent.getByToken(PFCandToken_, PFCand);
  if (!PFCand.isValid()) {
    edm::LogInfo("PFTester") << "Input PFCand collection not found.";
    return;
  }

  pf_candidates = PFCand.product();
  if (!pf_candidates) {
    edm::LogInfo("PFTester") << " Failed to retrieve data required by PFTester.cc";
    return;
  }

  // --------------------------------------------------------------------
  // ---------------- PF Clusters and associators -----------------------
  // --------------------------------------------------------------------

  edm::Handle<reco::PFClusterCollection> PFCluster;
  iEvent.getByToken(PFClusterToken_, PFCluster);
  if (!PFCluster.isValid()) {
    edm::LogInfo("PFTester") << "Input PFCluster collection not found.";
    return;
  }
  auto recoClusters = *PFCluster;

  edm::Handle<SimClusterCollection> SimCluster;
  iEvent.getByToken(SimClusterToken_, SimCluster);
  if (!SimCluster.isValid()) {
    edm::LogInfo("PFTester") << "Input SimCluster collection not found.";
    return;
  }
  auto simClusters = *SimCluster;

  edm::Handle<ticl::SimToRecoCollectionWithSimClustersT<reco::PFClusterCollection>> SimToRecoAssociatorCollection;
  iEvent.getByToken(SimToRecoAssociatorToken_, SimToRecoAssociatorCollection);
  if (!SimToRecoAssociatorCollection.isValid()) {
    edm::LogInfo("PFTester") << "Input PFClusterSimClusterAssociator SimToReco collection not found.";
    return;
  }
  auto simToRecoAssoc = *SimToRecoAssociatorCollection;
  // std::cout << "simRecColl size : " << simToRecoAssoc.size() << std::endl;

  edm::Handle<ticl::RecoToSimCollectionWithSimClustersT<reco::PFClusterCollection>> RecoToSimAssociatorCollection;
  iEvent.getByToken(RecoToSimAssociatorToken_, RecoToSimAssociatorCollection);
  if (!RecoToSimAssociatorCollection.isValid()) {
    edm::LogInfo("PFTester") << "Input PFClusterSimClusterAssociator RecoToSim collection not found.";
    return;
  }
  auto recoToSimAssoc = *RecoToSimAssociatorCollection;

  // --------------------------------------------------------------------
  // ----- Efficiency and merge computation at cluster level ------------
  // --------------------------------------------------------------------
  std::vector<unsigned> recoIdsMerged;

  for (unsigned int simId = 0; simId < simClusters.size(); ++simId) {
    // filter all sim clusters produced by a sim track which crossed the
    // tracker/calorimeter boundary outside the barrel
    auto const scTrack = simClusters[simId].g4Tracks()[0];
    const math::XYZTLorentzVectorF pos = scTrack.getPositionAtBoundary();
    if (abs(pos.Eta()) > 1.48)  // simTrack does not cross the barrel
      continue;

    h_simClusters_["En"]->Fill(simClusters[simId].energy());
    h_simClusters_["Pt"]->Fill(simClusters[simId].pt());
	h_simClusters_["PtLow"]->Fill(simClusters[simId].pt());
    h_simClusters_["Eta"]->Fill(simClusters[simId].eta());
    h_simClusters_["Phi"]->Fill(simClusters[simId].phi());
    h_simClusters_["Mult"]->Fill(simClusters[simId].numberOfRecHits());

    h2d_simClusters_["En_Eta"]->Fill(simClusters[simId].energy(), simClusters[simId].eta());
    h2d_simClusters_["En_Phi"]->Fill(simClusters[simId].energy(), simClusters[simId].phi());
    h2d_simClusters_["En_Mult"]->Fill(simClusters[simId].energy(), simClusters[simId].numberOfRecHits());
    h2d_simClusters_["Pt_Eta"]->Fill(simClusters[simId].pt(), simClusters[simId].eta());
    h2d_simClusters_["Pt_Phi"]->Fill(simClusters[simId].pt(), simClusters[simId].phi());
    h2d_simClusters_["Pt_Mult"]->Fill(simClusters[simId].pt(), simClusters[simId].numberOfRecHits());
    h2d_simClusters_["Mult_Eta"]->Fill(simClusters[simId].numberOfRecHits(), simClusters[simId].eta());
    h2d_simClusters_["Mult_Phi"]->Fill(simClusters[simId].numberOfRecHits(), simClusters[simId].phi());

    const edm::Ref<SimClusterCollection> simClusterRef(SimCluster, simId);
    const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
    if (simToRecoIt == simToRecoAssoc.end())
      continue;
    const auto& simToRecoMatched = simToRecoIt->val;
    if (simToRecoMatched.empty())
      continue;

    h_simClustersReconstructable_["En"]->Fill(simClusters[simId].energy());
    h_simClustersReconstructable_["Pt"]->Fill(simClusters[simId].pt());
	h_simClustersReconstructable_["PtLow"]->Fill(simClusters[simId].pt());
    h_simClustersReconstructable_["Eta"]->Fill(simClusters[simId].eta());
    h_simClustersReconstructable_["Phi"]->Fill(simClusters[simId].phi());
    h_simClustersReconstructable_["Mult"]->Fill(simClusters[simId].numberOfRecHits());

    h2d_simClustersReconstructable_["En_Eta"]->Fill(simClusters[simId].energy(), simClusters[simId].eta());
    h2d_simClustersReconstructable_["En_Phi"]->Fill(simClusters[simId].energy(), simClusters[simId].phi());
    h2d_simClustersReconstructable_["En_Mult"]->Fill(simClusters[simId].energy(), simClusters[simId].numberOfRecHits());
    h2d_simClustersReconstructable_["Pt_Eta"]->Fill(simClusters[simId].pt(), simClusters[simId].eta());
    h2d_simClustersReconstructable_["Pt_Phi"]->Fill(simClusters[simId].pt(), simClusters[simId].phi());
    h2d_simClustersReconstructable_["Pt_Mult"]->Fill(simClusters[simId].pt(), simClusters[simId].numberOfRecHits());
    h2d_simClustersReconstructable_["Mult_Eta"]->Fill(simClusters[simId].numberOfRecHits(), simClusters[simId].eta());
    h2d_simClustersReconstructable_["Mult_Phi"]->Fill(simClusters[simId].numberOfRecHits(), simClusters[simId].phi());

    std::vector<bool> wasNotFilled(assocScoreThresholds_.size(), true);
    for (const auto& recoPair : simToRecoMatched) {
      const auto recoPairIdx = recoPair.first.index();

      for (unsigned ithr = 0; ithr < assocScoreThresholds_.size(); ++ithr) {
        const double& thresh = assocScoreThresholds_[ithr];

        h_simToRecoScore_->Fill(recoPair.second.second);
        if (recoPair.second.second > thresh)
          continue;

        // numerator histograms must be filled only once per sim cluster
        // they are filled inside the recoPair loop to enable a different denominator per threshold
        if (wasNotFilled[ithr]) {
          wasNotFilled[ithr] = false;
          h_simClustersMatchedRecoClusters_[ithr]["En"]->Fill(simClusters[simId].energy());
          h_simClustersMatchedRecoClusters_[ithr]["Pt"]->Fill(simClusters[simId].pt());
		  h_simClustersMatchedRecoClusters_[ithr]["PtLow"]->Fill(simClusters[simId].pt());
          h_simClustersMatchedRecoClusters_[ithr]["Eta"]->Fill(simClusters[simId].eta());
          h_simClustersMatchedRecoClusters_[ithr]["Phi"]->Fill(simClusters[simId].phi());
          h_simClustersMatchedRecoClusters_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits());

          h2d_simClustersMatchedRecoClusters_[ithr]["En_Eta"]->Fill(simClusters[simId].energy(),
                                                                    simClusters[simId].eta());
          h2d_simClustersMatchedRecoClusters_[ithr]["En_Phi"]->Fill(simClusters[simId].energy(),
                                                                    simClusters[simId].phi());
          h2d_simClustersMatchedRecoClusters_[ithr]["En_Mult"]->Fill(simClusters[simId].energy(),
                                                                     simClusters[simId].numberOfRecHits());
          h2d_simClustersMatchedRecoClusters_[ithr]["Pt_Eta"]->Fill(simClusters[simId].pt(), simClusters[simId].eta());
          h2d_simClustersMatchedRecoClusters_[ithr]["Pt_Phi"]->Fill(simClusters[simId].pt(), simClusters[simId].phi());
          h2d_simClustersMatchedRecoClusters_[ithr]["Pt_Mult"]->Fill(simClusters[simId].pt(),
                                                                     simClusters[simId].numberOfRecHits());
          h2d_simClustersMatchedRecoClusters_[ithr]["Mult_Eta"]->Fill(simClusters[simId].numberOfRecHits(),
                                                                      simClusters[simId].eta());
          h2d_simClustersMatchedRecoClusters_[ithr]["Mult_Phi"]->Fill(simClusters[simId].numberOfRecHits(),
                                                                      simClusters[simId].phi());
        }

        // discard reco clusters from merge counting if already considered for a previous sim cluster
        const auto& mergeIt = std::find(recoIdsMerged.begin(), recoIdsMerged.end(), recoPairIdx);
        if (mergeIt != recoIdsMerged.end())
          continue;
        recoIdsMerged.push_back(recoPairIdx);

        const edm::Ref<reco::PFClusterCollection> recoClusterRef(PFCluster, recoPairIdx);
        const auto& recoToSimIt = recoToSimAssoc.find(recoClusterRef);
        assert(recoToSimIt != recoToSimAssoc.end());
        const auto& recoToSimMatched = recoToSimIt->val;
        assert(!recoToSimMatched.empty());

        // find how many sim clusters are associated to the matched reco cluster
        unsigned nSimMerged = 0;
        for (const auto& simPair : recoToSimMatched) {
          if (simPair.second > thresh)
            continue;
          ++nSimMerged;
        }

        if (nSimMerged > 1) {
          h_simClustersMultiMatchedRecoClusters_[ithr]["En"]->Fill(simClusters[simId].energy());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Pt"]->Fill(simClusters[simId].pt());
		  h_simClustersMultiMatchedRecoClusters_[ithr]["PtLow"]->Fill(simClusters[simId].pt());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Eta"]->Fill(simClusters[simId].eta());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Phi"]->Fill(simClusters[simId].phi());
          h_simClustersMultiMatchedRecoClusters_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits());
        }

        // for (const auto& recoPair : simToRecoMatched) {
        //   std::cout << " simToRecoAssoc simCluster id " << simId << " : matched recoCluster id = " << recoPair.first.index()
        // 			<< " shared energy = " << recoPair.second.first
        // 			<< " score = " << recoPair.second.second << std::endl;
        // }
      }
    }
  }

  std::vector<unsigned> simIdsDuplicates;

  // --------------------------------------------------------------------
  // ----- Fakes and duplicates computation at cluster level ------------
  // --------------------------------------------------------------------

  for (unsigned int recoId = 0; recoId < recoClusters.size(); ++recoId) {
    h_recoClusters_["En"]->Fill(recoClusters[recoId].energy());
    h_recoClusters_["Pt"]->Fill(recoClusters[recoId].pt());
	h_recoClusters_["PtLow"]->Fill(recoClusters[recoId].pt());
    h_recoClusters_["Eta"]->Fill(recoClusters[recoId].eta());
    h_recoClusters_["Phi"]->Fill(recoClusters[recoId].phi());
    h_recoClusters_["Mult"]->Fill(recoClusters[recoId].size());

    h2d_recoClusters_["En_Eta"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].eta());
    h2d_recoClusters_["En_Phi"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].phi());
    h2d_recoClusters_["En_Mult"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].size());
    h2d_recoClusters_["Pt_Eta"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].eta());
    h2d_recoClusters_["Pt_Phi"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].phi());
    h2d_recoClusters_["Pt_Mult"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].size());
    h2d_recoClusters_["Mult_Eta"]->Fill(recoClusters[recoId].size(), recoClusters[recoId].eta());
    h2d_recoClusters_["Mult_Phi"]->Fill(recoClusters[recoId].size(), recoClusters[recoId].phi());

    const edm::Ref<reco::PFClusterCollection> recoClusterRef(PFCluster, recoId);
    const auto& recoToSimIt = recoToSimAssoc.find(recoClusterRef);
    if (recoToSimIt == recoToSimAssoc.end())
      continue;
    const auto& recoToSimMatched = recoToSimIt->val;
    if (recoToSimMatched.empty())
      continue;

    h_recoClustersReconstructable_["En"]->Fill(recoClusters[recoId].energy());
    h_recoClustersReconstructable_["Pt"]->Fill(recoClusters[recoId].pt());
	h_recoClustersReconstructable_["PtLow"]->Fill(recoClusters[recoId].pt());
    h_recoClustersReconstructable_["Eta"]->Fill(recoClusters[recoId].eta());
    h_recoClustersReconstructable_["Phi"]->Fill(recoClusters[recoId].phi());
    h_recoClustersReconstructable_["Mult"]->Fill(recoClusters[recoId].size());

    h2d_recoClustersReconstructable_["En_Eta"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].eta());
    h2d_recoClustersReconstructable_["En_Phi"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].phi());
    h2d_recoClustersReconstructable_["En_Mult"]->Fill(recoClusters[recoId].energy(), recoClusters[recoId].size());
    h2d_recoClustersReconstructable_["Pt_Eta"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].eta());
    h2d_recoClustersReconstructable_["Pt_Phi"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].phi());
    h2d_recoClustersReconstructable_["Pt_Mult"]->Fill(recoClusters[recoId].pt(), recoClusters[recoId].size());
    h2d_recoClustersReconstructable_["Mult_Eta"]->Fill(recoClusters[recoId].size(), recoClusters[recoId].eta());
    h2d_recoClustersReconstructable_["Mult_Phi"]->Fill(recoClusters[recoId].size(), recoClusters[recoId].phi());

    std::vector<bool> wasNotFilled(assocScoreThresholds_.size(), true);
    for (const auto& simPair : recoToSimMatched) {
      const auto simPairIdx = simPair.first.index();
      // std::cout << " recoToSimAssoc recoCluster id " << recoId << " : matched simCluster id = " << simPairIdx
      // 			<< " score = " << simPair.second << std::endl;

      // filter all sim clusters produced by a sim track which crossed the
      // tracker/calorimeter boundary outside the barrel
      auto const scTrack = simClusters[simPairIdx].g4Tracks()[0];
      const math::XYZTLorentzVectorF pos = scTrack.getPositionAtBoundary();
      if (abs(pos.Eta()) > 1.48)  // simTrack does not cross the barrel
        continue;

      for (unsigned ithr = 0; ithr < assocScoreThresholds_.size(); ++ithr) {
        const double& thresh = assocScoreThresholds_[ithr];

        h_recoToSimScore_->Fill(simPair.second);
        if (simPair.second > thresh)
          continue;

        // numerator histograms must be filled only once per reco cluster
        if (wasNotFilled[ithr]) {
          wasNotFilled[ithr] = false;
          h_recoClustersMatchedSimClusters_[ithr]["En"]->Fill(recoClusters[recoId].energy());
          h_recoClustersMatchedSimClusters_[ithr]["Pt"]->Fill(recoClusters[recoId].pt());
		  h_recoClustersMatchedSimClusters_[ithr]["PtLow"]->Fill(recoClusters[recoId].pt());
          h_recoClustersMatchedSimClusters_[ithr]["Eta"]->Fill(recoClusters[recoId].eta());
          h_recoClustersMatchedSimClusters_[ithr]["Phi"]->Fill(recoClusters[recoId].phi());
          h_recoClustersMatchedSimClusters_[ithr]["Mult"]->Fill(recoClusters[recoId].size());

          h2d_recoClustersMatchedSimClusters_[ithr]["En_Eta"]->Fill(recoClusters[recoId].energy(),
                                                                    recoClusters[recoId].eta());
          h2d_recoClustersMatchedSimClusters_[ithr]["En_Phi"]->Fill(recoClusters[recoId].energy(),
                                                                    recoClusters[recoId].phi());
          h2d_recoClustersMatchedSimClusters_[ithr]["En_Mult"]->Fill(recoClusters[recoId].energy(),
                                                                     recoClusters[recoId].size());
          h2d_recoClustersMatchedSimClusters_[ithr]["Pt_Eta"]->Fill(recoClusters[recoId].pt(),
                                                                    recoClusters[recoId].eta());
          h2d_recoClustersMatchedSimClusters_[ithr]["Pt_Phi"]->Fill(recoClusters[recoId].pt(),
                                                                    recoClusters[recoId].phi());
          h2d_recoClustersMatchedSimClusters_[ithr]["Pt_Mult"]->Fill(recoClusters[recoId].pt(),
                                                                     recoClusters[recoId].size());
          h2d_recoClustersMatchedSimClusters_[ithr]["Mult_Eta"]->Fill(recoClusters[recoId].size(),
                                                                      recoClusters[recoId].eta());
          h2d_recoClustersMatchedSimClusters_[ithr]["Mult_Phi"]->Fill(recoClusters[recoId].size(),
                                                                      recoClusters[recoId].phi());
        }

        // discard sim clusters from duplicate counting if already considered for a previous reco cluster
        const auto& dupIt = std::find(simIdsDuplicates.begin(), simIdsDuplicates.end(), simPairIdx);
        if (dupIt != simIdsDuplicates.end())
          continue;
        simIdsDuplicates.push_back(simPairIdx);

        const edm::Ref<SimClusterCollection> simClusterRef(SimCluster, simPairIdx);
        const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
        assert(simToRecoIt != simToRecoAssoc.end());
        const auto& simToRecoMatched = simToRecoIt->val;
        assert(!simToRecoMatched.empty());

        // find how many reco clusters are associated to the matched sim cluster
        unsigned nRecoDuplicates = 0;
        for (const auto& recoPair : simToRecoMatched) {
          if (recoPair.second.second > thresh)
            continue;
          ++nRecoDuplicates;
        }

        if (nRecoDuplicates > 1) {
          h_recoClustersMultiMatchedSimClusters_[ithr]["En"]->Fill(recoClusters[recoId].energy());
		  h_recoClustersMultiMatchedSimClusters_[ithr]["Pt"]->Fill(recoClusters[recoId].pt());
          h_recoClustersMultiMatchedSimClusters_[ithr]["PtLow"]->Fill(recoClusters[recoId].pt());
          h_recoClustersMultiMatchedSimClusters_[ithr]["Eta"]->Fill(recoClusters[recoId].eta());
          h_recoClustersMultiMatchedSimClusters_[ithr]["Phi"]->Fill(recoClusters[recoId].phi());
          h_recoClustersMultiMatchedSimClusters_[ithr]["Mult"]->Fill(recoClusters[recoId].size());
        }
      }
    }
  }

  // --------------------------------------------------------------------
  // ----- Cluster response computation ---------------------------------
  // --------------------------------------------------------------------

  for (unsigned int simId = 0; simId < simClusters.size(); ++simId) {
    // filter all sim clusters produced by a sim track which crossed the
    // tracker/calorimeter boundary outside the barrel
    auto const scTrack = simClusters[simId].g4Tracks()[0];
    const math::XYZTLorentzVectorF pos = scTrack.getPositionAtBoundary();
    if (abs(pos.Eta()) > 1.48)  // simTrack does not cross the barrel
      continue;

    const edm::Ref<SimClusterCollection> simClusterRef(SimCluster, simId);
    const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
    if (simToRecoIt == simToRecoAssoc.end())
      continue;
    const auto& simToRecoMatched = simToRecoIt->val;
    if (simToRecoMatched.empty())
      continue;

    for (unsigned ithr = 0; ithr < assocScoreThresholds_.size(); ++ithr) {
      const double& thresh = assocScoreThresholds_[ithr];

      bool fill = true;

      // check for duplicate reco clusters
      unsigned recoId = 0;
      unsigned nRecoDuplicates = 0;
      for (const auto& recoPair : simToRecoMatched) {
        if (recoPair.second.second > thresh)
          continue;
        ++nRecoDuplicates;
        recoId = recoPair.first.index();  // used for the response computation
      }
      // only one reco cluster should be associated for the response
      if (nRecoDuplicates != 1) {
        fill = false;
      } else {  // check for merged sim cluster

        const edm::Ref<reco::PFClusterCollection> recoClusterRef(PFCluster, recoId);
        const auto& recoToSimIt = recoToSimAssoc.find(recoClusterRef);
        assert(recoToSimIt != recoToSimAssoc.end());
        const auto& recoToSimMatched = recoToSimIt->val;
        assert(!recoToSimMatched.empty());

        // find how many sim clusters are associated to the matched reco cluster
        unsigned nSimMerged = 0;
        for (const auto& simPair : recoToSimMatched) {
          if (simPair.second > thresh)
            continue;
          ++nSimMerged;
        }

        // only one sim cluster should be associated for the response
        if (nSimMerged != 1) {
          fill = false;
        }
      }

      if (fill) {
        h2d_response_[ithr]["En"]->Fill(simClusters[simId].energy(),
                                        recoClusters[recoId].pt() / simClusters[simId].pt());
        h2d_response_[ithr]["Pt"]->Fill(simClusters[simId].pt(),
                                        recoClusters[recoId].pt() / simClusters[simId].pt());
        h2d_response_[ithr]["Eta"]->Fill(simClusters[simId].eta(),
                                         recoClusters[recoId].pt() / simClusters[simId].pt());
        h2d_response_[ithr]["Phi"]->Fill(simClusters[simId].phi(),
                                         recoClusters[recoId].pt() / simClusters[simId].pt());
        h2d_response_[ithr]["Mult"]->Fill(simClusters[simId].numberOfRecHits(),
                                          recoClusters[recoId].pt() / simClusters[simId].pt());
      }
    }
  }

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  edm::Handle<ticl::RecoToSimCollectionT<reco::PFClusterCollection>> RecoToCpAssociatorCollection;
  iEvent.getByToken(RecoToCpAssociatorToken_, RecoToCpAssociatorCollection);
  if (!RecoToCpAssociatorCollection.isValid()) {
    std::cout << "Input PFClusterCpClusterAssociator RecoToSim collection not found." << std::endl;
    edm::LogInfo("PFTester") << "Input PFClusterCpClusterAssociator RecoToSim collection not found.";
  } else {
    auto recCpColl = *RecoToCpAssociatorCollection;
    // std::cout << "recCpColl size : " << recCpColl.size() << std::endl;

    for (unsigned int cId = 0; cId < recoClusters.size(); ++cId) {
      const edm::Ref<reco::PFClusterCollection> clusterRef(PFCluster, cId);
      const auto& scsIt = recCpColl.find(clusterRef);
      if (scsIt == recCpColl.end())
        continue;
      // const auto& scs = scsIt->val;
      // if (!scs.empty()) {
      //   for (const auto& scPair : scs) {
      //     // std::cout << " recCpColl Cluster id " << cId << " : first=" << scPair.first.index()
      //     //           << " second=" << scPair.second << std::endl;
      //   }
      // }
    }
  }

  edm::Handle<ticl::SimToRecoCollectionT<reco::PFClusterCollection>> CpToRecoAssociatorCollection;
  iEvent.getByToken(CpToRecoAssociatorToken_, CpToRecoAssociatorCollection);
  if (!CpToRecoAssociatorCollection.isValid()) {
    std::cout << "Input PFClusterCpClusterAssociator SimToReco collection not found." << std::endl;
    edm::LogInfo("PFTester") << "Input PFClusterCpClusterAssociator SimToReco collection not found.";
  } else {
    auto CpRecColl = *CpToRecoAssociatorCollection;
    // std::cout << "CpRecColl size : " << CpRecColl.size() << std::endl;
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
    h_NumPFClusters_->Fill(numPFClusters);
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
std::string PFTester::doubleToString(double x) const {
  std::ostringstream result;
  result << std::setprecision(2) << x;

  std::string xnew = result.str();
  std::size_t pos = xnew.find(".");
  if (pos != std::string::npos)
    xnew.replace(pos, 1, "p");
  else  //if the double was provided without decimal places
    xnew += "p0";

  return xnew;
}

DEFINE_FWK_MODULE(PFTester);
