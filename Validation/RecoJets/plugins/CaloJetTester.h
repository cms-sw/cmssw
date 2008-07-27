#ifndef ValidationRecoJetsCaloJetTester_h
#define ValidationRecoJetsCaloJetTester_h

// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by J F Novak July 10, 2008
// $Id: CaloJetTester.h,v 1.3 2008/07/21 21:57:26 chlebana Exp $

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
  virtual void beginJob(const edm::EventSetup&) ;
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

  // Generic Jet Parameters
  MonitorElement* mEta;
  MonitorElement* mEtaFineBin;
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

  // DiJet Parameters
  MonitorElement* mMjj;

  // CaloJet specific
  MonitorElement* mMaxEInEmTowers;
  MonitorElement* mMaxEInHadTowers;
  MonitorElement* mHadEnergyInHO;
  MonitorElement* mHadEnergyInHB;
  MonitorElement* mHadEnergyInHF;
  MonitorElement* mHadEnergyInHE;
  MonitorElement* mEmEnergyInEB;
  MonitorElement* mEmEnergyInEE;
  MonitorElement* mEmEnergyInHF;
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
  MonitorElement* mDeltaE;

  // Matching parameters
  double mMatchGenPtThreshold;
  double mGenEnergyFractionThreshold;
  double mReverseEnergyFractionThreshold;
  double mRThreshold;

  // Energy Profiles
  MonitorElement* mHadEnergyProfile;
  MonitorElement* mEmEnergyProfile;
  MonitorElement* mJetEnergyProfile;
  MonitorElement* mHadJetEnergyProfile;
  MonitorElement* mEMJetEnergyProfile;

  // CaloMET
  MonitorElement* mCaloMEx;
  MonitorElement* mCaloMEy;
  MonitorElement* mCaloMET;
  MonitorElement* mCaloMETPhi;
  MonitorElement* mCaloSumET;
  MonitorElement* mCaloMETSig;


};
#endif
