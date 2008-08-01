#ifndef ValidationRecoJetsCaloJetTester_h
#define ValidationRecoJetsCaloJetTester_h

// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: CaloJetTester.h,v 1.3 2007/08/28 16:51:54 fedor Exp $

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
};

#endif
