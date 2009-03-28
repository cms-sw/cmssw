#ifndef ValidationRecoJetsPFJetTester_h
#define ValidationRecoJetsPFJetTester_h

// Producer for validation histograms for PFlowJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: PFJetTester.h,v 1.3 2008/10/31 11:43:14 jueugste Exp $

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

  PFJetTester (const edm::ParameterSet&);
  ~PFJetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 private:
  
  void fillMatchHists (const reco::GenJet& fGenJet, const reco::PFJet& fFPJet) ;
  
    edm::InputTag mInputCollection; 
  edm::InputTag mInputGenCollection;
  std::string mOutputFile;

  // count number of events
  MonitorElement* numberofevents;

  // Generic Jet Parameters
  MonitorElement* mEta;
  MonitorElement* mEtaFineBin_Pt10;  //new
  MonitorElement* mPhi;
  MonitorElement* mPhiFineBin_Pt10;  //new
  MonitorElement* mE;
  MonitorElement* mE_80;  //new
  MonitorElement* mE_3000;  //new
  MonitorElement* mP;
  MonitorElement* mP_80;  //new
  MonitorElement* mP_3000;  //new
  MonitorElement* mPt;
  MonitorElement* mPt_80;  //new
  MonitorElement* mPt_3000;  //new
  MonitorElement* mMass;
  MonitorElement* mMass_80;
  MonitorElement* mMass_3000;  //new
  MonitorElement* mConstituents;
  MonitorElement* mConstituents_80;  //new
  MonitorElement* mConstituents_3000;  //new

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  MonitorElement* mEFirst;
  MonitorElement* mEFirst_80;  //new
  MonitorElement* mEFirst_3000;  //new
  MonitorElement* mPtFirst;
  MonitorElement* mPtFirst_80;  //new
  MonitorElement* mPtFirst_3000;  //new

  // PFlowJet specific

  MonitorElement* mChargedHadronEnergy;
  MonitorElement* mNeutralHadronEnergy;
  MonitorElement* mChargedEmEnergy;
  MonitorElement* mChargedMuEnergy;
  MonitorElement* mNeutralEmEnergy;
  MonitorElement* mChargedMultiplicity;
  MonitorElement* mNeutralMultiplicity;
  MonitorElement* mMuonMultiplicity;
  // new
  MonitorElement* mNeutralEmEnergy_80;
  MonitorElement* mNeutralEmEnergy_3000;   
  MonitorElement* mNeutralHadronEnergy_80;
  MonitorElement* mNeutralHadronEnergy_3000;
  MonitorElement* mChargedEmEnergy_80;       
  MonitorElement* mChargedEmEnergy_3000;
  MonitorElement* mChargedHadronEnergy_80;       
  MonitorElement* mChargedHadronEnergy_3000;       
    

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

  MonitorElement* mEScale_pt10;   ///new
  MonitorElement* mEScaleFineBin;  //new
  MonitorElement* mlinEScale;  //new
   


  // Matching parameters
  double mMatchGenPtThreshold;
  double mGenEnergyFractionThreshold;
  double mReverseEnergyFractionThreshold;
  double mRThreshold;

  // Switch on/off unimportant histogram
  std::string  mTurnOnEverything;
};

#endif
