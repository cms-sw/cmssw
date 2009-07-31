#ifndef ValidationRecoJetsPFJetTester_h
#define ValidationRecoJetsPFJetTester_h

// Producer for validation histograms for PFlowJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: PFJetTester.h,v 1.1 2008/05/27 21:52:15 ksmith Exp $

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
namespace reco {
  class PFJet;
  class GenJet;
}

class MonitorElement;

class PFJetTester : public edm::EDAnalyzer {
public:

  explicit PFJetTester (const edm::ParameterSet&);
  virtual ~PFJetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 private:
  
  void fillMatchHists (const reco::GenJet& fGenJet, const reco::PFJet& fFPJet) ;
  
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

  // PFlowJet specific

  MonitorElement* mChargedHadronEnergy;
  MonitorElement* mNeutralHadronEnergy;
  MonitorElement* mChargedEmEnergy;
  MonitorElement* mChargedMuEnergy;
  MonitorElement* mNeutralEmEnergy;
  MonitorElement* mChargedMultiplicity;
  MonitorElement* mNeutralMultiplicity;
  MonitorElement* mMuonMultiplicity;

  //new Plots with Res./ Eff. as function of neutral, charged &  em fraction

  MonitorElement* mNeutralFraction;
  MonitorElement* mNeutralFraction2;

  MonitorElement* mEEffNeutralFraction;
  MonitorElement* mEEffChargedFraction;
  MonitorElement* mEResNeutralFraction;
  MonitorElement* mEResChargedFraction;
  MonitorElement* nEEff;

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
