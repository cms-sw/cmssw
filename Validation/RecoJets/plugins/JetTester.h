#ifndef ValidationRecoJetsJetTester_h
#define ValidationRecoJetsJetTester_h

// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Rewritten by Viola Sordini, Matthias Artur Weber, Robert Schoefbeck Nov./Dez.
// 2013

#include <cmath>
#include <string>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

class JetTester : public DQMEDAnalyzer {
public:
  JetTester(const edm::ParameterSet &);
  ~JetTester() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void fillMatchHists(const double GenEta,
                      const double GenPhi,
                      const double GenPt,
                      const double GenMass,
                      const double RecoEta,
                      const double RecoPhi,
                      const double RecoPt,
                      const double RecoMass);

  edm::InputTag mInputCollection;
  edm::InputTag mInputGenCollection;
  edm::InputTag mJetCorrector;
  std::string JetType;

  // Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
  edm::EDGetTokenT<GenEventInfoProduct> evtToken_;
  edm::EDGetTokenT<pat::JetCollection> patJetsToken_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;

  static const int ptSize = 10;
  static constexpr std::array<double, ptSize + 1> ptBins_ = {
      {20., 30., 40., 100., 200., 300., 600., 2000., 5000., 6500., 1e6}};
  double minJetPt;

  static constexpr std::array<std::tuple<std::string, std::string, double, double>, 3> etaInfo = {{
      std::make_tuple("B", "0<|#eta|<1.5", 0.0, 1.5),     // barrel
      std::make_tuple("E", "1.5#leq|#eta|<3", 1.5, 3.0),  // endcap
      std::make_tuple("F", "3#leq|#eta|<6", 3.0, 6.0),    // forward
  }};
  static constexpr size_t etaSize = etaInfo.size();

  // Event variables
  MonitorElement *mNvtx;

  // Jet parameters
  MonitorElement *mJetEta;
  MonitorElement *mJetPhi;
  MonitorElement *mJetPt;
  MonitorElement *mJetEnergy;
  MonitorElement *mJetMass;
  MonitorElement *mJetConstituents;
  MonitorElement *mJetArea;
  std::array<MonitorElement *, etaSize> mJetPt_EtaBins;

  // Corrected jet parameters
  MonitorElement *mCorrJetEta;
  MonitorElement *mCorrJetPhi;
  MonitorElement *mCorrJetPt;
  std::array<MonitorElement *, etaSize> mCorrJetPt_EtaBins;

  // Gen jet parameters
  MonitorElement *mGenEta;
  MonitorElement *mGenPhi;
  MonitorElement *mGenPt;
  std::array<MonitorElement *, etaSize> mGenPt_EtaBins;

  // Matched jet parameters
  MonitorElement *mMatchedJetEta;
  MonitorElement *mMatchedJetPhi;
  std::array<MonitorElement *, etaSize> mMatchedJetPt_EtaBins;
  std::array<MonitorElement *, etaSize> mMatchedCorrPt_EtaBins;

  // Matched gen jet parameters
  MonitorElement *mMatchedGenEta;
  MonitorElement *mMatchedGenPhi;
  std::array<MonitorElement *, etaSize> mMatchedGenPt_EtaBins;

  // Duplicates (gen and reco)
  static constexpr size_t nLevelsDuplicates = 3;
  std::unordered_map<std::string, std::array<MonitorElement *, nLevelsDuplicates>> mGenRepeat, mRecoRepeat;
  std::unordered_map<std::string, std::array<std::array<MonitorElement *, etaSize>, nLevelsDuplicates>> mGenRepeat_EtaBins, mRecoRepeat_EtaBins;
  
  // Jet response vs gen histograms
  std::array<MonitorElement *, etaSize> h_JetPtRecoOverGen;
  std::array<std::array<MonitorElement *, ptSize>, etaSize> hVector_JetPtRecoOverGen_ptBins;

  // Corrected jet response vs gen histograms
  std::array<MonitorElement *, etaSize> h_JetPtCorrOverGen;
  std::array<std::array<MonitorElement *, ptSize>, etaSize> hVector_JetPtCorrOverGen_ptBins;

  // Corrected jet response vs reco histograms
  std::array<MonitorElement *, etaSize> h_JetPtCorrOverReco;
  std::array<std::array<MonitorElement *, ptSize>, etaSize> hVector_JetPtCorrOverReco_ptBins;

  // Jet response vs gen profiled in gen variable
  std::array<MonitorElement *, ptSize> p_JetPtRecoOverGen_vs_GenEta;
  std::array<MonitorElement *, etaSize> p_JetPtRecoOverGen_vs_GenPhi;
  std::array<MonitorElement *, etaSize> p_JetPtRecoOverGen_vs_GenPt;

  std::array<MonitorElement *, ptSize> h2d_JetPtRecoOverGen_vs_GenEta;
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_GenPhi;
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_GenPt;

  // Corrected jet response vs gen profiled in gen variable
  std::array<MonitorElement *, ptSize> p_JetPtCorrOverGen_vs_GenEta;
  std::array<MonitorElement *, etaSize> p_JetPtCorrOverGen_vs_GenPhi;
  std::array<MonitorElement *, etaSize> p_JetPtCorrOverGen_vs_GenPt;

  std::array<MonitorElement *, ptSize> h2d_JetPtCorrOverGen_vs_GenEta;
  std::array<MonitorElement *, etaSize> h2d_JetPtCorrOverGen_vs_GenPhi;
  std::array<MonitorElement *, etaSize> h2d_JetPtCorrOverGen_vs_GenPt;

  // Corrected jet response vs reco profiled in reco variable
  std::array<MonitorElement *, ptSize> p_JetPtCorrOverReco_vs_Eta;
  std::array<MonitorElement *, etaSize> p_JetPtCorrOverReco_vs_Phi;
  std::array<MonitorElement *, etaSize> p_JetPtCorrOverReco_vs_Pt;

  std::array<MonitorElement *, ptSize> h2d_JetPtCorrOverReco_vs_Eta;
  std::array<MonitorElement *, etaSize> h2d_JetPtCorrOverReco_vs_Phi;
  std::array<MonitorElement *, etaSize> h2d_JetPtCorrOverReco_vs_Pt;

  // Jet em/had fractions profiled in pt
  std::array<MonitorElement *, etaSize> p_chHad_vs_pt;
  std::array<MonitorElement *, etaSize> p_neHad_vs_pt;
  std::array<MonitorElement *, etaSize> p_chEm_vs_pt;
  std::array<MonitorElement *, etaSize> p_neEm_vs_pt;

  // Jet response vs gen profiled in em/had fractions
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_chHad;
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_neHad;
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_chEm;
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_neEm;
  std::array<MonitorElement *, etaSize> h2d_JetPtRecoOverGen_vs_nCost;

  // Generation
  MonitorElement *mJetEtaFirst;
  MonitorElement *mJetPhiFirst;
  MonitorElement *mJetPtFirst;
  MonitorElement *mGenEtaFirst;
  MonitorElement *mGenPhiFirst;
  MonitorElement *mGenPtFirst;

  MonitorElement *mMjj;
  MonitorElement *mNJets;
  MonitorElement *mNJetsPt1;
  MonitorElement *mNJetsPt2;
  MonitorElement *mDeltaEta;
  MonitorElement *mDeltaPhi;
  MonitorElement *mDeltaPt;
  std::array<MonitorElement *, etaSize> mNJets_EtaBins;

  // ---- Calo Jet specific information ----
  MonitorElement *maxEInEmTowers;
  MonitorElement *maxEInHadTowers;
  MonitorElement *energyFractionHadronic;
  MonitorElement *emEnergyFraction;
  MonitorElement *hadEnergyInHB;
  MonitorElement *hadEnergyInHO;
  MonitorElement *hadEnergyInHE;
  MonitorElement *hadEnergyInHF;
  MonitorElement *emEnergyInEB;
  MonitorElement *emEnergyInEE;
  MonitorElement *emEnergyInHF;
  MonitorElement *towersArea;
  MonitorElement *n90;
  MonitorElement *n60;

  // ---- JPT or PF Jet specific information ----
  MonitorElement *muonMultiplicity;
  MonitorElement *chargedMultiplicity;
  MonitorElement *chargedEmEnergy;
  MonitorElement *neutralEmEnergy;
  MonitorElement *chargedHadronEnergy;
  MonitorElement *neutralHadronEnergy;
  MonitorElement *chargedHadronEnergyFraction;
  MonitorElement *neutralHadronEnergyFraction;
  MonitorElement *chargedEmEnergyFraction;
  MonitorElement *neutralEmEnergyFraction;

  // ---- PF Jet specific information ----
  MonitorElement *photonEnergy;
  MonitorElement *photonEnergyFraction;
  MonitorElement *electronEnergy;
  MonitorElement *electronEnergyFraction;
  MonitorElement *muonEnergy;
  MonitorElement *muonEnergyFraction;
  MonitorElement *HFHadronEnergy;
  MonitorElement *HFHadronEnergyFraction;
  MonitorElement *HFEMEnergy;
  MonitorElement *HFEMEnergyFraction;
  MonitorElement *chargedHadronMultiplicity;
  MonitorElement *neutralHadronMultiplicity;
  MonitorElement *photonMultiplicity;
  MonitorElement *electronMultiplicity;
  MonitorElement *HFHadronMultiplicity;
  MonitorElement *HFEMMultiplicity;
  MonitorElement *chargedMuEnergy;
  MonitorElement *chargedMuEnergyFraction;
  MonitorElement *neutralMultiplicity;
  MonitorElement *HOEnergy;
  MonitorElement *HOEnergyFraction;

  // contained in MiniAOD
  MonitorElement *hadronFlavor;
  MonitorElement *partonFlavor;
  MonitorElement *genPartonPDGID;

  // Parameters
  double mRecoJetPtThreshold;
  double mMatchGenPtThreshold;
  double mRThreshold;
  bool isCaloJet;
  bool isPFJet;
  bool isMiniAODJet;
  bool isHLT_;
};

#endif
