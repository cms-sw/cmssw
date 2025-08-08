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
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

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

  std::vector<double> ptBins_;
  int n_bins_pt;
  double minJetPt;
  double medJetPt;

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
  std::vector<MonitorElement*> mJetPt_EtaBins;

  // Corrected jet parameters
  MonitorElement *mCorrJetEta;
  MonitorElement *mCorrJetPhi;
  MonitorElement *mCorrJetPt;
  std::vector<MonitorElement*> mCorrJetPt_EtaBins;

  // Gen jet parameters
  MonitorElement *mGenEta;
  MonitorElement *mGenPhi;
  MonitorElement *mGenPt;
  std::vector<MonitorElement*> mGenPt_EtaBins;

  // Matched jet parameters
  MonitorElement *mMatchedJetEta;
  MonitorElement *mMatchedJetPhi;
  std::vector<MonitorElement*> mMatchedJetPt_EtaBins;
  std::vector<MonitorElement*> mMatchedCorrPt_EtaBins;

  // Matched gen jet parameters
  MonitorElement *mMatchedGenEta;
  MonitorElement *mMatchedGenPhi;
  std::vector<MonitorElement*> mMatchedGenPt_EtaBins;

  // Jet response vs gen histograms
  std::vector<MonitorElement*> h_JetPtRecoOverGen;
  std::vector<std::vector<MonitorElement*>> hVector_JetPtRecoOverGen_ptBins;

  // Corrected jet response vs gen histograms
  std::vector<MonitorElement*> h_JetPtCorrOverGen;
  std::vector<std::vector<MonitorElement*>> hVector_JetPtCorrOverGen_ptBins;

  // Corrected jet response vs reco histograms
  std::vector<MonitorElement*> h_JetPtCorrOverReco;
  std::vector<std::vector<MonitorElement*>> hVector_JetPtCorrOverReco_ptBins;

  // Jet response vs gen profiled in gen variable
  std::vector<MonitorElement*> p_JetPtRecoOverGen_vs_GenEta;
  std::vector<MonitorElement*> p_JetPtRecoOverGen_vs_GenPhi;
  std::vector<MonitorElement*> p_JetPtRecoOverGen_vs_GenPt;

  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_GenEta;
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_GenPhi;
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_GenPt;

  // Corrected jet response vs gen profiled in gen variable
  std::vector<MonitorElement*> p_JetPtCorrOverGen_vs_GenEta;
  std::vector<MonitorElement*> p_JetPtCorrOverGen_vs_GenPhi;
  std::vector<MonitorElement*> p_JetPtCorrOverGen_vs_GenPt;

  std::vector<MonitorElement*> h2d_JetPtCorrOverGen_vs_GenEta;
  std::vector<MonitorElement*> h2d_JetPtCorrOverGen_vs_GenPhi;
  std::vector<MonitorElement*> h2d_JetPtCorrOverGen_vs_GenPt;

  // Corrected jet response vs reco profiled in reco variable
  std::vector<MonitorElement*> p_JetPtCorrOverReco_vs_Eta;
  std::vector<MonitorElement*> p_JetPtCorrOverReco_vs_Phi;
  std::vector<MonitorElement*> p_JetPtCorrOverReco_vs_Pt;

  std::vector<MonitorElement*> h2d_JetPtCorrOverReco_vs_Eta;
  std::vector<MonitorElement*> h2d_JetPtCorrOverReco_vs_Phi;
  std::vector<MonitorElement*> h2d_JetPtCorrOverReco_vs_Pt;

  // Jet em/had fractions profiled in pt
  std::vector<MonitorElement*> p_chHad_vs_pt;
  std::vector<MonitorElement*> p_neHad_vs_pt;
  std::vector<MonitorElement*> p_chEm_vs_pt;
  std::vector<MonitorElement*> p_neEm_vs_pt;

  // Jet response vs gen profiled in em/had fractions
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_chHad;
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_neHad;
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_chEm;
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_neEm;
  std::vector<MonitorElement*> h2d_JetPtRecoOverGen_vs_nCost;

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
  std::vector<MonitorElement*> mNJets_EtaBins;

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

  std::vector<std::tuple<std::string, std::string, double, double>> etaInfo;
};

#endif
