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

  // Corrected jet parameters
  MonitorElement *mCorrJetEta;
  MonitorElement *mCorrJetPhi;
  MonitorElement *mCorrJetPt;

  // Gen jet parameters
  MonitorElement *mGenEta;
  MonitorElement *mGenPhi;
  MonitorElement *mGenPt;

  // Jet response vs gen histograms
  MonitorElement *h_JetPtRecoOverGen_B;
  MonitorElement *h_JetPtRecoOverGen_E;
  MonitorElement *h_JetPtRecoOverGen_F;
  std::vector<MonitorElement*> hVector_JetPtRecoOverGen_B_ptBins;
  std::vector<MonitorElement*> hVector_JetPtRecoOverGen_E_ptBins;
  std::vector<MonitorElement*> hVector_JetPtRecoOverGen_F_ptBins;

  // Corrected jet response vs gen histograms
  MonitorElement *h_JetPtCorrOverGen_B;
  MonitorElement *h_JetPtCorrOverGen_E;
  MonitorElement *h_JetPtCorrOverGen_F;
  std::vector<MonitorElement*> hVector_JetPtCorrOverGen_B_ptBins;
  std::vector<MonitorElement*> hVector_JetPtCorrOverGen_E_ptBins;
  std::vector<MonitorElement*> hVector_JetPtCorrOverGen_F_ptBins;

  // Corrected jet response vs reco histograms
  MonitorElement *h_JetPtCorrOverReco_B;
  MonitorElement *h_JetPtCorrOverReco_E;
  MonitorElement *h_JetPtCorrOverReco_F;
  std::vector<MonitorElement*> hVector_JetPtCorrOverReco_B_ptBins;
  std::vector<MonitorElement*> hVector_JetPtCorrOverReco_E_ptBins;
  std::vector<MonitorElement*> hVector_JetPtCorrOverReco_F_ptBins;

  // Jet response vs gen profiled in gen variable
  MonitorElement *p_JetPtRecoOverGen_vs_GenEta;
  MonitorElement *p_JetPtRecoOverGen_vs_GenPhi_B;
  MonitorElement *p_JetPtRecoOverGen_vs_GenPhi_E;
  MonitorElement *p_JetPtRecoOverGen_vs_GenPhi_F;
  MonitorElement *p_JetPtRecoOverGen_vs_GenPt_B;
  MonitorElement *p_JetPtRecoOverGen_vs_GenPt_E;
  MonitorElement *p_JetPtRecoOverGen_vs_GenPt_F;

  MonitorElement *h2d_JetPtRecoOverGen_vs_GenEta;
  MonitorElement *h2d_JetPtRecoOverGen_vs_GenPhi_B;
  MonitorElement *h2d_JetPtRecoOverGen_vs_GenPhi_E;
  MonitorElement *h2d_JetPtRecoOverGen_vs_GenPhi_F;
  MonitorElement *h2d_JetPtRecoOverGen_vs_GenPt_B;
  MonitorElement *h2d_JetPtRecoOverGen_vs_GenPt_E;
  MonitorElement *h2d_JetPtRecoOverGen_vs_GenPt_F;

  // Corrected jet response vs gen profiled in gen variable
  MonitorElement *p_JetPtCorrOverGen_vs_GenEta;
  MonitorElement *p_JetPtCorrOverGen_vs_GenPhi_B;
  MonitorElement *p_JetPtCorrOverGen_vs_GenPhi_E;
  MonitorElement *p_JetPtCorrOverGen_vs_GenPhi_F;
  MonitorElement *p_JetPtCorrOverGen_vs_GenPt_B;
  MonitorElement *p_JetPtCorrOverGen_vs_GenPt_E;
  MonitorElement *p_JetPtCorrOverGen_vs_GenPt_F;

  MonitorElement *h2d_JetPtCorrOverGen_vs_GenEta;
  MonitorElement *h2d_JetPtCorrOverGen_vs_GenPhi_B;
  MonitorElement *h2d_JetPtCorrOverGen_vs_GenPhi_E;
  MonitorElement *h2d_JetPtCorrOverGen_vs_GenPhi_F;
  MonitorElement *h2d_JetPtCorrOverGen_vs_GenPt_B;
  MonitorElement *h2d_JetPtCorrOverGen_vs_GenPt_E;
  MonitorElement *h2d_JetPtCorrOverGen_vs_GenPt_F;

  // Corrected jet response vs reco profiled in reco variable
  MonitorElement *p_JetPtCorrOverReco_vs_Eta;
  MonitorElement *p_JetPtCorrOverReco_vs_Phi_B;
  MonitorElement *p_JetPtCorrOverReco_vs_Phi_E;
  MonitorElement *p_JetPtCorrOverReco_vs_Phi_F;
  MonitorElement *p_JetPtCorrOverReco_vs_Pt_B;
  MonitorElement *p_JetPtCorrOverReco_vs_Pt_E;
  MonitorElement *p_JetPtCorrOverReco_vs_Pt_F;

  MonitorElement *h2d_JetPtCorrOverReco_vs_Eta;
  MonitorElement *h2d_JetPtCorrOverReco_vs_Phi_B;
  MonitorElement *h2d_JetPtCorrOverReco_vs_Phi_E;
  MonitorElement *h2d_JetPtCorrOverReco_vs_Phi_F;
  MonitorElement *h2d_JetPtCorrOverReco_vs_Pt_B;
  MonitorElement *h2d_JetPtCorrOverReco_vs_Pt_E;
  MonitorElement *h2d_JetPtCorrOverReco_vs_Pt_F;

  // Generation
  MonitorElement *mJetEtaFirst;
  MonitorElement *mJetPhiFirst;
  MonitorElement *mJetPtFirst;
  MonitorElement *mGenEtaFirst;
  MonitorElement *mGenPhiFirst;
  MonitorElement *mGenPtFirst;

  MonitorElement *mMjj;
  MonitorElement *mNJets1;
  MonitorElement *mNJets2;
  MonitorElement *mDeltaEta;
  MonitorElement *mDeltaPhi;
  MonitorElement *mDeltaPt;

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
};

#endif
