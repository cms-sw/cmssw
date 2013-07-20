#ifndef ValidationRecoJetsPFJetTester_h
#define ValidationRecoJetsPFJetTester_h

// Producer for validation histograms for PFJet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung.Jeong Feb 2, 2010
// $Id: PFJetTester.h,v 1.15 2012/02/15 21:41:52 kovitang Exp $

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco {
  class PFJet;
  class GenJet;
}

class MonitorElement;

class PFJetTester : public edm::EDAnalyzer {
public:

  PFJetTester (const edm::ParameterSet&);
  ~PFJetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;

private:
  
  void fillMatchHists (const reco::GenJet& fGenJet, const reco::PFJet& fPFJet,std::vector<reco::Vertex> goodVertices);

  edm::InputTag mInputCollection;
  edm::InputTag mInputGenCollection;
  std::string mOutputFile;
  edm::InputTag inputMETLabel_;
  std::string METType_;
  std::string inputGenMETLabel_;
  std::string inputCaloMETLabel_;

  // count number of events
  MonitorElement* numberofevents;

  // Generic Jet Parameters
  MonitorElement* mEta;
  MonitorElement* mEtaFineBin;
  MonitorElement* mEtaFineBin1p;
  MonitorElement* mEtaFineBin2p;
  MonitorElement* mEtaFineBin3p;
  MonitorElement* mEtaFineBin1m;
  MonitorElement* mEtaFineBin2m;
  MonitorElement* mEtaFineBin3m;
  MonitorElement* mPhi;
  MonitorElement* mPhiFineBin;
  MonitorElement* mE;
  MonitorElement* mE_80;
  MonitorElement* mE_3000;
  MonitorElement* mP;
  MonitorElement* mP_80;
  MonitorElement* mP_3000;
  MonitorElement* mPt;
  MonitorElement* mPt_80;
  MonitorElement* mPt_3000;
  MonitorElement* mMass;
  MonitorElement* mMass_80;
  MonitorElement* mMass_3000;
  MonitorElement* mConstituents;
  MonitorElement* mConstituents_80;
  MonitorElement* mConstituents_3000;
  MonitorElement* mHadTiming;
  MonitorElement* mEmTiming;

  //Corr jets
  MonitorElement* mCorrJetPt;
  MonitorElement* mCorrJetPt_80;
  MonitorElement* mCorrJetPt_3000;
  MonitorElement* mCorrJetEta;
  MonitorElement* mCorrJetPhi;
  MonitorElement* mpTRatio;
  MonitorElement* mpTRatioB_d;
  MonitorElement* mpTRatioE_d;
  MonitorElement* mpTRatioF_d;
  MonitorElement* mpTRatio_30_200_d;
  MonitorElement* mpTRatio_200_600_d;
  MonitorElement* mpTRatio_600_1500_d;
  MonitorElement* mpTRatio_1500_3500_d;
  MonitorElement* mpTResponse;
  MonitorElement* mpTResponseB_d;
  MonitorElement* mpTResponseE_d;
  MonitorElement* mpTResponseF_d;
  MonitorElement* mpTResponse_30_200_d;
  MonitorElement* mpTResponse_200_600_d;
  MonitorElement* mpTResponse_600_1500_d;
  MonitorElement* mpTResponse_1500_3500_d;
  MonitorElement* mpTResponse_30_d;
  MonitorElement* mjetArea;

  // nvtx
  MonitorElement* nvtx_0_30;
  MonitorElement* nvtx_0_60;
  MonitorElement* mpTResponse_nvtx_0_5;
  MonitorElement* mpTResponse_nvtx_5_10; 
  MonitorElement* mpTResponse_nvtx_10_15;
  MonitorElement* mpTResponse_nvtx_15_20;
  MonitorElement* mpTResponse_nvtx_20_30; 
  MonitorElement* mpTResponse_nvtx_30_inf;
  MonitorElement* mpTScale_a_nvtx_0_5;
  MonitorElement* mpTScale_b_nvtx_0_5;
  MonitorElement* mpTScale_c_nvtx_0_5;
  MonitorElement* mpTScale_a_nvtx_5_10;
  MonitorElement* mpTScale_b_nvtx_5_10;
  MonitorElement* mpTScale_c_nvtx_5_10;
  MonitorElement* mpTScale_a_nvtx_10_15;
  MonitorElement* mpTScale_b_nvtx_10_15;
  MonitorElement* mpTScale_c_nvtx_10_15;
  MonitorElement* mpTScale_a_nvtx_15_20;
  MonitorElement* mpTScale_b_nvtx_15_20;
  MonitorElement* mpTScale_c_nvtx_15_20;
  MonitorElement* mpTScale_a_nvtx_20_30;
  MonitorElement* mpTScale_b_nvtx_20_30;
  MonitorElement* mpTScale_c_nvtx_20_30;
  MonitorElement* mpTScale_a_nvtx_30_inf;
  MonitorElement* mpTScale_b_nvtx_30_inf;
  MonitorElement* mpTScale_c_nvtx_30_inf;
  MonitorElement* mpTScale_nvtx_0_5;
  MonitorElement* mpTScale_nvtx_5_10;
  MonitorElement* mpTScale_nvtx_10_15;
  MonitorElement* mpTScale_nvtx_15_20;
  MonitorElement* mpTScale_nvtx_20_30;
  MonitorElement* mpTScale_nvtx_30_inf;
  MonitorElement* mNJetsEtaF_30;
  MonitorElement* mpTScale_a;
  MonitorElement* mpTScale_b;
  MonitorElement* mpTScale_c;
  MonitorElement* mpTScale_pT;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  MonitorElement* mEFirst;
  MonitorElement* mEFirst_80;
  MonitorElement* mEFirst_3000;
  MonitorElement* mPtFirst;
  MonitorElement* mPtFirst_80;
  MonitorElement* mPtFirst_3000;

  MonitorElement* mNJetsEtaC;
  MonitorElement* mNJetsEtaF;

  MonitorElement* mNJets1;
  MonitorElement* mNJets2;

  // DiJet Parameters
  MonitorElement* mMjj;
  MonitorElement* mMjj_3000;

  // PFJet specific
  MonitorElement* mChargedEmEnergy_80;
  MonitorElement* mChargedHadronEnergy_80;
  MonitorElement* mNeutralEmEnergy_80;
  MonitorElement* mNeutralHadronEnergy_80;

  MonitorElement* mChargedEmEnergy_3000;
  MonitorElement* mChargedHadronEnergy_3000;
  MonitorElement* mNeutralEmEnergy_3000;
  MonitorElement* mNeutralHadronEnergy_3000;

  MonitorElement* mChargedEmEnergyFraction_B;
  MonitorElement* mChargedEmEnergyFraction_E;
  MonitorElement* mChargedEmEnergyFraction_F;
  MonitorElement* mChargedHadronEnergyFraction_B;
  MonitorElement* mChargedHadronEnergyFraction_E;
  MonitorElement* mChargedHadronEnergyFraction_F;
  MonitorElement* mNeutralEmEnergyFraction_B;
  MonitorElement* mNeutralEmEnergyFraction_E;
  MonitorElement* mNeutralEmEnergyFraction_F;
  MonitorElement* mNeutralHadronEnergyFraction_B;
  MonitorElement* mNeutralHadronEnergyFraction_E;
  MonitorElement* mNeutralHadronEnergyFraction_F;

  //  MonitorElement* mMaxEInEmTowers;
  //  MonitorElement* mMaxEInHadTowers;
  //  MonitorElement* mHadEnergyInHO;
  //  MonitorElement* mHadEnergyInHB;
  MonitorElement* mHadEnergyInHF;
  MonitorElement* mHadEnergyInHF_80;
  MonitorElement* mHadEnergyInHF_3000;
  //  MonitorElement* mHadEnergyInHE;
  //  MonitorElement* mHadEnergyInHO_80;
  //  MonitorElement* mHadEnergyInHB_80;
  //  MonitorElement* mHadEnergyInHE_80;
  //  MonitorElement* mHadEnergyInHO_3000;
  //  MonitorElement* mHadEnergyInHB_3000;
  //  MonitorElement* mHadEnergyInHE_3000;
  //  MonitorElement* mEmEnergyInEB;
  //  MonitorElement* mEmEnergyInEE;
  MonitorElement* mEmEnergyInHF;
  MonitorElement* mEmEnergyInHF_80;
  MonitorElement* mEmEnergyInHF_3000;
  //  MonitorElement* mEmEnergyInEB_80;
  //  MonitorElement* mEmEnergyInEE_80;
  //  MonitorElement* mEmEnergyInEB_3000;
  //  MonitorElement* mEmEnergyInEE_3000;
  //  MonitorElement* mEnergyFractionHadronic;
  //  MonitorElement* mEnergyFractionEm;
  //  MonitorElement* mHFTotal;
  //  MonitorElement* mHFTotal_80;
  //  MonitorElement* mHFTotal_3000;
  //  MonitorElement* mHFLong;
  //  MonitorElement* mHFLong_80;
  //  MonitorElement* mHFLong_3000;
  //  MonitorElement* mHFShort;
  //  MonitorElement* mHFShort_80;
  //  MonitorElement* mHFShort_3000;
  //  MonitorElement* mN90;

  MonitorElement* mElectronEnergy;
  MonitorElement* mElectronEnergy_80;
  MonitorElement* mElectronEnergy_3000;
  MonitorElement* mMuonEnergy;
  MonitorElement* mMuonEnergy_80;
  MonitorElement* mMuonEnergy_3000;
  MonitorElement* mPhotonEnergy;
  MonitorElement* mPhotonEnergy_80;
  MonitorElement* mPhotonEnergy_3000;

  // pthat
  MonitorElement* mPthat_80;
  MonitorElement* mPthat_3000;

  // GenJet Generic Jet Parameters
  MonitorElement* mGenEta;
  MonitorElement* mGenPhi;
  MonitorElement* mGenPt;
  MonitorElement* mGenPt_80;
  MonitorElement* mGenPt_3000;

  // GenJet Leading Jet Parameters
  MonitorElement* mGenEtaFirst;
  MonitorElement* mGenPhiFirst;

  // PFJet<->GenJet matching
  MonitorElement* mAllGenJetsPt;
  MonitorElement* mMatchedGenJetsPt;
  MonitorElement* mAllGenJetsEta;
  MonitorElement* mMatchedGenJetsEta;
  MonitorElement* mGenJetMatchEnergyFraction;
  MonitorElement* mReverseMatchEnergyFraction;
  MonitorElement* mRMatch;
  MonitorElement* mDeltaEta;
  MonitorElement* mDeltaPhi;
  MonitorElement* mEScale;
  MonitorElement* mlinEScale;  //new
  MonitorElement* mDeltaE;

  MonitorElement* mEScale_pt10;  //new
  MonitorElement* mEScaleFineBin;  //new

  MonitorElement* mpTScaleB_s;
  MonitorElement* mpTScaleE_s;
  MonitorElement* mpTScaleF_s;
  MonitorElement* mpTScaleB_d;
  MonitorElement* mpTScaleE_d;
  MonitorElement* mpTScaleF_d;
  MonitorElement* mpTScalePhiB_d;
  MonitorElement* mpTScalePhiE_d;
  MonitorElement* mpTScalePhiF_d;

  MonitorElement* mpTScale_30_200_s;
  MonitorElement* mpTScale_200_600_s;
  MonitorElement* mpTScale_600_1500_s;
  MonitorElement* mpTScale_1500_3500_s;

  MonitorElement* mpTScale_30_200_d;
  MonitorElement* mpTScale_200_600_d;
  MonitorElement* mpTScale_600_1500_d;
  MonitorElement* mpTScale_1500_3500_d;

  MonitorElement* mpTScale1DB_30_200;
  MonitorElement* mpTScale1DE_30_200;
  MonitorElement* mpTScale1DF_30_200;
  MonitorElement* mpTScale1DB_200_600;
  MonitorElement* mpTScale1DE_200_600;
  MonitorElement* mpTScale1DF_200_600;
  MonitorElement* mpTScale1DB_600_1500;
  MonitorElement* mpTScale1DE_600_1500;
  MonitorElement* mpTScale1DF_600_1500;
  MonitorElement* mpTScale1DB_1500_3500;
  MonitorElement* mpTScale1DE_1500_3500;
  MonitorElement* mpTScale1DF_1500_3500;
  MonitorElement* mpTScale1D_30_200;
  MonitorElement* mpTScale1D_200_600;
  MonitorElement* mpTScale1D_600_1500;
  MonitorElement* mpTScale1D_1500_3500;

  MonitorElement* mDelEta;
  MonitorElement* mDelPhi;
  MonitorElement* mDelPt;

  // Matching parameters
  double mMatchGenPtThreshold;
  double mGenEnergyFractionThreshold;
  double mReverseEnergyFractionThreshold;
  double mRThreshold;

  std::string JetCorrectionService;

  // Switch on/off unimportant histogram
  std::string  mTurnOnEverything;

  // Energy Profiles
  MonitorElement* mHadEnergyProfile;
  MonitorElement* mEmEnergyProfile;
  MonitorElement* mJetEnergyProfile;
  //  MonitorElement* mHadJetEnergyProfile;
  //  MonitorElement* mEMJetEnergyProfile;

  // CaloMET
  MonitorElement* mCaloMEx;
  MonitorElement* mCaloMEx_3000;
  MonitorElement* mCaloMEy;
  MonitorElement* mCaloMEy_3000;
  MonitorElement* mCaloMET;
  MonitorElement* mCaloMET_3000;
  MonitorElement* mCaloMETPhi;
  MonitorElement* mCaloSumET;
  MonitorElement* mCaloSumET_3000;
  MonitorElement* mCaloMETSig;
  MonitorElement* mCaloMETSig_3000;

  // RecHits
  MonitorElement* mHBEne;
  MonitorElement* mHBTime;
  MonitorElement* mHEEne;
  MonitorElement* mHETime;
  MonitorElement* mHOEne;
  MonitorElement* mHOTime;
  MonitorElement* mHFEne;
  MonitorElement* mHFTime;
  MonitorElement* mEBEne;
  MonitorElement* mEBTime;
  MonitorElement* mEEEne;
  MonitorElement* mEETime;


};
#endif
