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
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
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

  edm::EDGetTokenT<reco::PFCandidateCollection> PFCandToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> PFClusterHCALToken_;
  edm::EDGetTokenT<SimClusterCollection> SimClusterHCALToken_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionWithSimClustersT<reco::PFClusterCollection>> RecoToSimAssociatorHCALToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionWithSimClustersT<reco::PFClusterCollection>> SimToRecoAssociatorHCALToken_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionT<reco::PFClusterCollection>> RecoToCpAssociatorHCALToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionT<reco::PFClusterCollection>> CpToRecoAssociatorHCALToken_;

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

  double assocScoreThreshold_;

  std::unordered_map<std::string, std::tuple<unsigned, float, float>> histoVars = {
      {"Energy", std::make_tuple(100, 0., 50.)},
      {"Pt", std::make_tuple(100, 0., 40.)},
      {"Eta", std::make_tuple(50, -6.5, 6.5)},
      {"Phi", std::make_tuple(50, -3.5, 3.5)},
      {"Mult", std::make_tuple(20, 0., 20.)},
  };

  using UMap = std::unordered_map<std::string, MonitorElement*>;
  UMap h_simClusters_;
  UMap h_simClustersMatchedRecoClusters_;
  UMap h_simClustersMultiMatchedRecoClusters_;
  UMap h_recoClusters_;
  UMap h_recoClustersMatchedSimClusters_;
  UMap h_recoClustersMultiMatchedSimClusters_;
};

PFTester::PFTester(const edm::ParameterSet& iConfig)
    : PFCandToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCand"))),
      PFClusterHCALToken_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("PFClusterHCAL"))),
      SimClusterHCALToken_(consumes<SimClusterCollection>(iConfig.getParameter<edm::InputTag>("SimClusterHCAL"))),
      RecoToSimAssociatorHCALToken_(consumes<ticl::RecoToSimCollectionWithSimClustersT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterSimClusterAssociatorHCAL"))),
      SimToRecoAssociatorHCALToken_(consumes<ticl::SimToRecoCollectionWithSimClustersT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterSimClusterAssociatorHCAL"))),
      RecoToCpAssociatorHCALToken_(consumes<ticl::RecoToSimCollectionT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterCaloParticleAssociatorHCAL"))),
      CpToRecoAssociatorHCALToken_(consumes<ticl::SimToRecoCollectionT<reco::PFClusterCollection>>(
          iConfig.getParameter<edm::InputTag>("PFClusterCaloParticleAssociatorHCAL"))),
      assocScoreThreshold_(iConfig.getParameter<double>("assocScoreThreshold")) {}

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

  ibook.setCurrentFolder("HLT/ParticleFlow/PFClusterValidation");
  for (auto& hVar : histoVars) {
    auto [nBins, hMin, hMax] = hVar.second;
    h_simClusters_[hVar.first] =
        ibook.book1D("SimClusters" + hVar.first, "SimClusters;" + hVar.first, nBins, hMin, hMax);
    h_simClustersMatchedRecoClusters_[hVar.first] = ibook.book1D("SimClustersMatchedRecoClusters" + hVar.first,
                                                                 "SimClusters matched to RecoClusters;" + hVar.first,
                                                                 nBins,
                                                                 hMin,
                                                                 hMax);
    h_simClustersMultiMatchedRecoClusters_[hVar.first] =
        ibook.book1D("SimClustersMultiMatchedRecoClusters" + hVar.first,
                     "SimClusters multi-matched to RecoClusters;" + hVar.first,
                     nBins,
                     hMin,
                     hMax);

    h_recoClusters_[hVar.first] =
        ibook.book1D("RecoClusters" + hVar.first, "RecoClusters;" + hVar.first, nBins, hMin, hMax);

    h_recoClustersMatchedSimClusters_[hVar.first] = ibook.book1D("RecoClustersMatchedSimClusters" + hVar.first,
                                                                 "RecoClusters matched to SimClusters;" + hVar.first,
                                                                 nBins,
                                                                 hMin,
                                                                 hMax);
    h_recoClustersMultiMatchedSimClusters_[hVar.first] =
        ibook.book1D("RecoClustersMultiMatchedSimClusters" + hVar.first,
                     "RecoClusters multi-matched to SimClusters;" + hVar.first,
                     nBins,
                     hMin,
                     hMax);
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
  // ---------------- PF Cluster Efficiency -----------------------------
  // --------------------------------------------------------------------

  edm::Handle<reco::PFClusterCollection> PFClusterHCAL;
  iEvent.getByToken(PFClusterHCALToken_, PFClusterHCAL);
  if (!PFClusterHCAL.isValid()) {
    edm::LogInfo("PFTester") << "Input PFClusterHCAL collection not found.";
    return;
  }
  auto recoClusters = *PFClusterHCAL;

  edm::Handle<SimClusterCollection> SimClusterHCAL;
  iEvent.getByToken(SimClusterHCALToken_, SimClusterHCAL);
  if (!SimClusterHCAL.isValid()) {
    edm::LogInfo("PFTester") << "Input SimClusterHCAL collection not found.";
    return;
  }
  auto simClusters = *SimClusterHCAL;

  edm::Handle<ticl::SimToRecoCollectionWithSimClustersT<reco::PFClusterCollection>> SimToRecoAssociatorHCALCollection;
  iEvent.getByToken(SimToRecoAssociatorHCALToken_, SimToRecoAssociatorHCALCollection);
  if (!SimToRecoAssociatorHCALCollection.isValid()) {
    edm::LogInfo("PFTester") << "Input PFClusterSimClusterAssociatorHCAL SimToReco collection not found.";
    return;
  }
  auto simToRecoAssoc = *SimToRecoAssociatorHCALCollection;
  // std::cout << "simRecColl size : " << simToRecoAssoc.size() << std::endl;

  edm::Handle<ticl::RecoToSimCollectionWithSimClustersT<reco::PFClusterCollection>> RecoToSimAssociatorHCALCollection;
  iEvent.getByToken(RecoToSimAssociatorHCALToken_, RecoToSimAssociatorHCALCollection);
  if (!RecoToSimAssociatorHCALCollection.isValid()) {
    edm::LogInfo("PFTester") << "Input PFClusterSimClusterAssociatorHCAL RecoToSim collection not found.";
    return;
  }
  auto recoToSimAssoc = *RecoToSimAssociatorHCALCollection;

  std::vector<unsigned> recoIdsMerged;

  // efficiency and merge rate computation
  for (unsigned int simId = 0; simId < simClusters.size(); ++simId) {
    h_simClusters_["Energy"]->Fill(simClusters[simId].energy());
    h_simClusters_["Pt"]->Fill(simClusters[simId].pt());
    h_simClusters_["Eta"]->Fill(simClusters[simId].eta());
    h_simClusters_["Phi"]->Fill(simClusters[simId].phi());
    h_simClusters_["Mult"]->Fill(simClusters[simId].numberOfSimHits());

    const edm::Ref<SimClusterCollection> simClusterRef(SimClusterHCAL, simId);
    const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
    if (simToRecoIt == simToRecoAssoc.end())
      continue;
    const auto& simToRecoMatched = simToRecoIt->val;
    if (simToRecoMatched.empty())
      continue;

    bool wasNotFilled = true;
    for (const auto& recoPair : simToRecoMatched) {
      const auto recoPairIdx = recoPair.first.index();

      if (recoPair.second.second < assocScoreThreshold_)
        continue;

      // numerator histograms must be filled only once per sim cluster
      if (wasNotFilled) {
        wasNotFilled = false;
        h_simClustersMatchedRecoClusters_["Energy"]->Fill(simClusters[simId].energy());
        h_simClustersMatchedRecoClusters_["Pt"]->Fill(simClusters[simId].pt());
        h_simClustersMatchedRecoClusters_["Eta"]->Fill(simClusters[simId].eta());
        h_simClustersMatchedRecoClusters_["Phi"]->Fill(simClusters[simId].phi());
        h_simClustersMatchedRecoClusters_["Mult"]->Fill(simClusters[simId].numberOfSimHits());
      }

      // discard reco clusters from merge counting if already considered for a previous sim cluster
      const auto& mergeIt = std::find(recoIdsMerged.begin(), recoIdsMerged.end(), recoPairIdx);
      if (mergeIt != recoIdsMerged.end())
        continue;
      recoIdsMerged.push_back(recoPairIdx);

      const edm::Ref<reco::PFClusterCollection> recoClusterRef(PFClusterHCAL, recoPairIdx);
      const auto& recoToSimIt = recoToSimAssoc.find(recoClusterRef);
      assert(recoToSimIt != recoToSimAssoc.end());
      const auto& recoToSimMatched = recoToSimIt->val;
      assert(!recoToSimMatched.empty());

      // find how many reco clusters are associated to the matched sim cluster
      unsigned nSimMerged = 0;
      for (const auto& simPair : recoToSimMatched) {
        if (simPair.second < assocScoreThreshold_)
          continue;
        ++nSimMerged;
      }

      if (nSimMerged > 1) {
        h_simClustersMultiMatchedRecoClusters_["Energy"]->Fill(recoClusters[simId].energy());
        h_simClustersMultiMatchedRecoClusters_["Pt"]->Fill(recoClusters[simId].pt());
        h_simClustersMultiMatchedRecoClusters_["Eta"]->Fill(recoClusters[simId].eta());
        h_simClustersMultiMatchedRecoClusters_["Phi"]->Fill(recoClusters[simId].phi());
        h_simClustersMultiMatchedRecoClusters_["Mult"]->Fill(recoClusters[simId].size());
      }

      // for (const auto& recoPair : simToRecoMatched) {
      //   std::cout << " simToRecoAssoc simCluster id " << simId << " : matched recoCluster id = " << recoPair.first.index()
      // 			<< " shared energy = " << recoPair.second.first
      // 			<< " score = " << recoPair.second.second << std::endl;
      // }
    }
  }

  std::vector<unsigned> simIdsDuplicates;

  // fake rate and duplicate computation
  for (unsigned int recoId = 0; recoId < recoClusters.size(); ++recoId) {
    h_recoClusters_["Energy"]->Fill(recoClusters[recoId].energy());
    h_recoClusters_["Pt"]->Fill(recoClusters[recoId].pt());
    h_recoClusters_["Eta"]->Fill(recoClusters[recoId].eta());
    h_recoClusters_["Phi"]->Fill(recoClusters[recoId].phi());
    h_recoClusters_["Mult"]->Fill(recoClusters[recoId].size());

    const edm::Ref<reco::PFClusterCollection> recoClusterRef(PFClusterHCAL, recoId);
    const auto& recoToSimIt = recoToSimAssoc.find(recoClusterRef);
    if (recoToSimIt == recoToSimAssoc.end())
      continue;
    const auto& recoToSimMatched = recoToSimIt->val;
    if (recoToSimMatched.empty())
      continue;

    bool wasNotFilled = true;
    for (const auto& simPair : recoToSimMatched) {
      const auto simPairIdx = simPair.first.index();
      // std::cout << " recoToSimAssoc recoCluster id " << recoId << " : matched simCluster id = " << simPairIdx
      // 			<< " score = " << simPair.second << std::endl;

      if (simPair.second < assocScoreThreshold_)
        continue;

      // numerator histograms must be filled only once per reco cluster
      if (wasNotFilled) {
        wasNotFilled = false;
        h_recoClustersMatchedSimClusters_["Energy"]->Fill(recoClusters[recoId].energy());
        h_recoClustersMatchedSimClusters_["Pt"]->Fill(recoClusters[recoId].pt());
        h_recoClustersMatchedSimClusters_["Eta"]->Fill(recoClusters[recoId].eta());
        h_recoClustersMatchedSimClusters_["Phi"]->Fill(recoClusters[recoId].phi());
        h_recoClustersMatchedSimClusters_["Mult"]->Fill(recoClusters[recoId].size());
      }

      // discard sim clusters from duplicate counting if already considered for a previous reco cluster
      const auto& dupIt = std::find(simIdsDuplicates.begin(), simIdsDuplicates.end(), simPairIdx);
      if (dupIt != simIdsDuplicates.end())
        continue;
      simIdsDuplicates.push_back(simPairIdx);

      const edm::Ref<SimClusterCollection> simClusterRef(SimClusterHCAL, simPairIdx);
      const auto& simToRecoIt = simToRecoAssoc.find(simClusterRef);
      assert(simToRecoIt != simToRecoAssoc.end());
      const auto& simToRecoMatched = simToRecoIt->val;
      assert(!simToRecoMatched.empty());

      // find how many reco clusters are associated to the matched sim cluster
      unsigned nRecoDuplicates = 0;
      for (const auto& recoPair : simToRecoMatched) {
        if (recoPair.second.second < assocScoreThreshold_)
          continue;
        ++nRecoDuplicates;
      }

      if (nRecoDuplicates > 1) {
        h_recoClustersMultiMatchedSimClusters_["Energy"]->Fill(recoClusters[recoId].energy());
        h_recoClustersMultiMatchedSimClusters_["Pt"]->Fill(recoClusters[recoId].pt());
        h_recoClustersMultiMatchedSimClusters_["Eta"]->Fill(recoClusters[recoId].eta());
        h_recoClustersMultiMatchedSimClusters_["Phi"]->Fill(recoClusters[recoId].phi());
        h_recoClustersMultiMatchedSimClusters_["Mult"]->Fill(recoClusters[recoId].size());
      }
    }
  }

  // --------------------------------------------------------------------
  // --------------------------------------------------------------------

  edm::Handle<ticl::RecoToSimCollectionT<reco::PFClusterCollection>> RecoToCpAssociatorHCALCollection;
  iEvent.getByToken(RecoToCpAssociatorHCALToken_, RecoToCpAssociatorHCALCollection);
  if (!RecoToCpAssociatorHCALCollection.isValid()) {
    std::cout << "Input PFClusterCpClusterAssociatorHCAL RecoToSim collection not found." << std::endl;
    edm::LogInfo("PFTester") << "Input PFClusterCpClusterAssociatorHCAL RecoToSim collection not found.";
  } else {
    auto recCpColl = *RecoToCpAssociatorHCALCollection;
    // std::cout << "recCpColl size : " << recCpColl.size() << std::endl;

    for (unsigned int cId = 0; cId < recoClusters.size(); ++cId) {
      const edm::Ref<reco::PFClusterCollection> clusterRef(PFClusterHCAL, cId);
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

  edm::Handle<ticl::SimToRecoCollectionT<reco::PFClusterCollection>> CpToRecoAssociatorHCALCollection;
  iEvent.getByToken(CpToRecoAssociatorHCALToken_, CpToRecoAssociatorHCALCollection);
  if (!CpToRecoAssociatorHCALCollection.isValid()) {
    std::cout << "Input PFClusterCpClusterAssociatorHCAL SimToReco collection not found." << std::endl;
    edm::LogInfo("PFTester") << "Input PFClusterCpClusterAssociatorHCAL SimToReco collection not found.";
  } else {
    auto CpRecColl = *CpToRecoAssociatorHCALCollection;
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

DEFINE_FWK_MODULE(PFTester);
