#ifndef ValidationRecoJetsCaloJetTester_h
#define ValidationRecoJetsCaloJetTester_h

// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by J F Novak July 10, 2008
// $Id: CaloJetTester.h,v 1.16 2009/07/13 19:03:02 chjeong Exp $

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

namespace reco {
  class CaloJet;
  class GenJet;
}

class MonitorElement;

class CaloJetTester : public edm::EDAnalyzer {
public:

  CaloJetTester (const edm::ParameterSet&);
  ~CaloJetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;
 
private:
  
  void fillMatchHists (const reco::GenJet& fGenJet, const reco::CaloJet& fCaloJet);

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

  // CaloJet specific
  MonitorElement* mMaxEInEmTowers;
  MonitorElement* mMaxEInHadTowers;
  MonitorElement* mHadEnergyInHO;
  MonitorElement* mHadEnergyInHB;
  MonitorElement* mHadEnergyInHF;
  MonitorElement* mHadEnergyInHE;
  MonitorElement* mHadEnergyInHO_80;
  MonitorElement* mHadEnergyInHB_80;
  MonitorElement* mHadEnergyInHE_80;
  MonitorElement* mHadEnergyInHO_3000;
  MonitorElement* mHadEnergyInHB_3000;
  MonitorElement* mHadEnergyInHE_3000;
  MonitorElement* mEmEnergyInEB;
  MonitorElement* mEmEnergyInEE;
  MonitorElement* mEmEnergyInHF;
  MonitorElement* mEmEnergyInEB_80;
  MonitorElement* mEmEnergyInEE_80;
  MonitorElement* mEmEnergyInEB_3000;
  MonitorElement* mEmEnergyInEE_3000;
  MonitorElement* mEnergyFractionHadronic;
  MonitorElement* mEnergyFractionEm;
  MonitorElement* mHFTotal;
  MonitorElement* mHFTotal_80;
  MonitorElement* mHFTotal_3000;
  MonitorElement* mHFLong;
  MonitorElement* mHFLong_80;
  MonitorElement* mHFLong_3000;
  MonitorElement* mHFShort;
  MonitorElement* mHFShort_80;
  MonitorElement* mHFShort_3000;
  MonitorElement* mN90;

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

  // CaloJet<->GenJet matching
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

  MonitorElement* mpTScale_60_120_s;
  MonitorElement* mpTScale_200_300_s;
  MonitorElement* mpTScale_600_900_s;
  MonitorElement* mpTScale_2700_3500_s;

  MonitorElement* mpTScale_60_120_d;
  MonitorElement* mpTScale_200_300_d;
  MonitorElement* mpTScale_600_900_d;
  MonitorElement* mpTScale_2700_3500_d;

  MonitorElement* mpTScale1DB_60_120;
  MonitorElement* mpTScale1DE_60_120;
  MonitorElement* mpTScale1DF_60_120;
  MonitorElement* mpTScale1DB_200_300;
  MonitorElement* mpTScale1DE_200_300;
  MonitorElement* mpTScale1DF_200_300;
  MonitorElement* mpTScale1DB_600_900;
  MonitorElement* mpTScale1DE_600_900;
  MonitorElement* mpTScale1DF_600_900;
  MonitorElement* mpTScale1DB_2700_3500;
  MonitorElement* mpTScale1DE_2700_3500;
  MonitorElement* mpTScale1DF_2700_3500;
  MonitorElement* mpTScale1D_60_120;
  MonitorElement* mpTScale1D_200_300;
  MonitorElement* mpTScale1D_600_900;
  MonitorElement* mpTScale1D_2700_3500;

  MonitorElement* mDelEta;
  MonitorElement* mDelPhi;
  MonitorElement* mDelPt;

  // Matching parameters
  double mMatchGenPtThreshold;
  double mGenEnergyFractionThreshold;
  double mReverseEnergyFractionThreshold;
  double mRThreshold;

  // Switch on/off unimportant histogram
  std::string  mTurnOnEverything;

  // Energy Profiles
  MonitorElement* mHadEnergyProfile;
  MonitorElement* mEmEnergyProfile;
  MonitorElement* mJetEnergyProfile;
  MonitorElement* mHadJetEnergyProfile;
  MonitorElement* mEMJetEnergyProfile;

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
