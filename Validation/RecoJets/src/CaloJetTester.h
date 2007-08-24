#ifndef ValidationRecoJetsCaloJetTester_h
#define ValidationRecoJetsCaloJetTester_h

// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: CaloJetTester.h,v 1.1 2007/08/20 21:51:36 fedor Exp $

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"


class MonitorElement;

class CaloJetTester : public edm::EDAnalyzer {
public:

  CaloJetTester (const edm::ParameterSet&);
  ~CaloJetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 
 private:

  edm::InputTag mInputCollection;
  edm::InputTag mInputGenCollection;
  std::string mOutputFile;

  // Generic Jet Parameters
  MonitorElement* mEta;
  MonitorElement* mPhi;
  MonitorElement* mE;
  MonitorElement* mP;
  MonitorElement* mPt;
  MonitorElement* mMass;
  MonitorElement* mConstituents;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  MonitorElement* mEFirst;
  MonitorElement* mPtFirst;

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
  MonitorElement* mN90;

  // CaloJet<->GenJet matching
  MonitorElement* mAllGenJetsPt;
  MonitorElement* mMatchedGenJetsPt;
  MonitorElement* mAllGenJetsEta;
  MonitorElement* mMatchedGenJetsEta;
  MonitorElement* mGenJetMatchEnergyFraction;
  MonitorElement* mReverseMatchEnergyFraction;
  MonitorElement* mDeltaEta_B;
  MonitorElement* mDeltaEta_E;
  MonitorElement* mDeltaEta_F;
  MonitorElement* mDeltaPhi_B;
  MonitorElement* mDeltaPhi_E;
  MonitorElement* mDeltaPhi_F;
  MonitorElement* mEScale_B;
  MonitorElement* mEScale_E;
  MonitorElement* mEScale_F;
  MonitorElement* mDeltaE_B;
  MonitorElement* mDeltaE_E;
  MonitorElement* mDeltaE_F;

  // Matching parameters
  double mMatchGenPtThreshold;
  double mGenEnergyFractionThreshold;
  double mReverseEnergyFractionThreshold;
};

#endif
