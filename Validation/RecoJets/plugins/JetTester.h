#ifndef ValidationRecoJetsJetTester_h
#define ValidationRecoJetsJetTester_h

// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Rewritten by Viola Sordini, Matthias Artur Weber, Robert Schoefbeck Nov./Dez. 2013

#include <cmath>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
class MonitorElement;

class JetTester : public DQMEDAnalyzer {
 public:

  JetTester (const edm::ParameterSet&);
  ~JetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  
  void fillMatchHists(const double GenEta,  const double GenPhi,  const double GenPt,
		      const double RecoEta, const double RecoPhi, const double RecoPt);
  
  edm::InputTag   mInputCollection;
  edm::InputTag   mInputGenCollection;
  edm::InputTag   mJetCorrector;
  std::string     JetType;

  //Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex> > pvToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
  edm::EDGetTokenT<GenEventInfoProduct> evtToken_;
  edm::EDGetTokenT<pat::JetCollection> patJetsToken_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;

  // Event variables
  MonitorElement* mNvtx;

  // Jet parameters
  MonitorElement* mEta;
  MonitorElement* mPhi;
  MonitorElement* mPt;
  MonitorElement* mP;
  MonitorElement* mEnergy;
  MonitorElement* mMass;
  MonitorElement* mConstituents;
  MonitorElement* mJetArea;
//  MonitorElement* mRho;

  // Corrected jets
  MonitorElement* mCorrJetPt;
  MonitorElement* mCorrJetEta;
  MonitorElement* mCorrJetPhi;
  MonitorElement* mCorrJetEta_Pt40;
  MonitorElement* mCorrJetPhi_Pt40;
  // Corrected jets profiles
  MonitorElement* mPtCorrOverReco_Pt_B;
  MonitorElement* mPtCorrOverReco_Pt_E;
  MonitorElement* mPtCorrOverReco_Pt_F;
  MonitorElement* mPtCorrOverReco_Eta_20_40;
  MonitorElement* mPtCorrOverReco_Eta_40_200;
  MonitorElement* mPtCorrOverReco_Eta_200_600;
  MonitorElement* mPtCorrOverReco_Eta_600_1500;
  MonitorElement* mPtCorrOverReco_Eta_1500_3500;
  MonitorElement* mPtCorrOverReco_Eta_3500_5000;
  MonitorElement* mPtCorrOverReco_Eta_5000_6500;
  MonitorElement* mPtCorrOverReco_Eta_3500;
  MonitorElement* mPtCorrOverGen_GenPt_B;
  MonitorElement* mPtCorrOverGen_GenPt_E;
  MonitorElement* mPtCorrOverGen_GenPt_F;
  MonitorElement* mPtCorrOverGen_GenEta_20_40;
  MonitorElement* mPtCorrOverGen_GenEta_40_200;
  MonitorElement* mPtCorrOverGen_GenEta_200_600;
  MonitorElement* mPtCorrOverGen_GenEta_600_1500;
  MonitorElement* mPtCorrOverGen_GenEta_1500_3500;
  MonitorElement* mPtCorrOverGen_GenEta_3500_5000;
  MonitorElement* mPtCorrOverGen_GenEta_5000_6500;
  MonitorElement* mPtCorrOverGen_GenEta_3500;

  // Generation
  MonitorElement* mGenEta;
  MonitorElement* mGenPhi;
  MonitorElement* mGenPt;
  MonitorElement* mGenEtaFirst;
  MonitorElement* mGenPhiFirst;
  MonitorElement* mPtHat;
  MonitorElement* mDeltaEta;
  MonitorElement* mDeltaPhi;
  MonitorElement* mDeltaPt;

  MonitorElement* mPtRecoOverGen_B_20_40;
  MonitorElement* mPtRecoOverGen_E_20_40;
  MonitorElement* mPtRecoOverGen_F_20_40;
  MonitorElement* mPtRecoOverGen_B_40_200;
  MonitorElement* mPtRecoOverGen_E_40_200;
  MonitorElement* mPtRecoOverGen_F_40_200;
  MonitorElement* mPtRecoOverGen_B_200_600;
  MonitorElement* mPtRecoOverGen_E_200_600;
  MonitorElement* mPtRecoOverGen_F_200_600;
  MonitorElement* mPtRecoOverGen_B_600_1500;
  MonitorElement* mPtRecoOverGen_E_600_1500;
  MonitorElement* mPtRecoOverGen_F_600_1500;
  MonitorElement* mPtRecoOverGen_B_1500_3500;
  MonitorElement* mPtRecoOverGen_E_1500_3500;
  MonitorElement* mPtRecoOverGen_F_1500_3500;

  MonitorElement* mPtRecoOverGen_B_3500_5000;
  MonitorElement* mPtRecoOverGen_E_3500_5000;
  MonitorElement* mPtRecoOverGen_B_5000_6500;
  MonitorElement* mPtRecoOverGen_E_5000_6500;
  MonitorElement* mPtRecoOverGen_B_3500;
  MonitorElement* mPtRecoOverGen_E_3500;
  MonitorElement* mPtRecoOverGen_F_3500;

  // Generation profiles
  MonitorElement* mPtRecoOverGen_GenPt_B;
  MonitorElement* mPtRecoOverGen_GenPt_E;
  MonitorElement* mPtRecoOverGen_GenPt_F;
  MonitorElement* mPtRecoOverGen_GenPhi_B;
  MonitorElement* mPtRecoOverGen_GenPhi_E;
  MonitorElement* mPtRecoOverGen_GenPhi_F;
  MonitorElement* mPtRecoOverGen_GenEta_20_40;
  MonitorElement* mPtRecoOverGen_GenEta_40_200;
  MonitorElement* mPtRecoOverGen_GenEta_200_600;
  MonitorElement* mPtRecoOverGen_GenEta_600_1500;
  MonitorElement* mPtRecoOverGen_GenEta_1500_3500;
  MonitorElement* mPtRecoOverGen_GenEta_3500_5000;
  MonitorElement* mPtRecoOverGen_GenEta_5000_6500;
  MonitorElement* mPtRecoOverGen_GenEta_3500;


  // Some jet algebra
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  MonitorElement* mPtFirst;
  MonitorElement* mMjj;
  MonitorElement* mNJetsEta_B_20_40;
  MonitorElement* mNJetsEta_E_20_40;
  MonitorElement* mNJetsEta_B_40;
  MonitorElement* mNJetsEta_E_40;
  MonitorElement* mNJets_40;
  MonitorElement* mNJets1;
  MonitorElement* mNJets2;

  // ---- Calo Jet specific information ----
  MonitorElement* maxEInEmTowers;
  MonitorElement* maxEInHadTowers;
  MonitorElement* energyFractionHadronic;
  MonitorElement* emEnergyFraction;
  MonitorElement* hadEnergyInHB;
  MonitorElement* hadEnergyInHO;
  MonitorElement* hadEnergyInHE;
  MonitorElement* hadEnergyInHF;
  MonitorElement* emEnergyInEB;
  MonitorElement* emEnergyInEE;
  MonitorElement* emEnergyInHF;
  MonitorElement* towersArea;
  MonitorElement* n90;
  MonitorElement* n60;

  // ---- JPT or PF Jet specific information ----
  MonitorElement* muonMultiplicity;
  MonitorElement* chargedMultiplicity;
  MonitorElement* chargedEmEnergy;
  MonitorElement* neutralEmEnergy;
  MonitorElement* chargedHadronEnergy;
  MonitorElement* neutralHadronEnergy;
  MonitorElement* chargedHadronEnergyFraction;
  MonitorElement* neutralHadronEnergyFraction;
  MonitorElement* chargedEmEnergyFraction;
  MonitorElement* neutralEmEnergyFraction;

  // ---- PF Jet specific information ----
  MonitorElement* photonEnergy;
  MonitorElement* photonEnergyFraction;
  MonitorElement* electronEnergy;
  MonitorElement* electronEnergyFraction;
  MonitorElement* muonEnergy;
  MonitorElement* muonEnergyFraction;
  MonitorElement* HFHadronEnergy;
  MonitorElement* HFHadronEnergyFraction;
  MonitorElement* HFEMEnergy;
  MonitorElement* HFEMEnergyFraction;
  MonitorElement* chargedHadronMultiplicity;
  MonitorElement* neutralHadronMultiplicity;
  MonitorElement* photonMultiplicity;
  MonitorElement* electronMultiplicity;
  MonitorElement* HFHadronMultiplicity;
  MonitorElement* HFEMMultiplicity;
  MonitorElement* chargedMuEnergy;
  MonitorElement* chargedMuEnergyFraction;
  MonitorElement* neutralMultiplicity;
  MonitorElement* HOEnergy;
  MonitorElement* HOEnergyFraction;

  // Parameters
  double          mRecoJetPtThreshold;
  double          mMatchGenPtThreshold;
  double          mRThreshold;
  bool            isCaloJet;
  bool            isPFJet;
  bool            isMiniAODJet;


};

#endif
