#ifndef ValidationRecoJetsPFJetTester_h
#define ValidationRecoJetsPFJetTester_h

// Producer for validation histograms for PFlowJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: PFlowJetTester.h,v 1.2 2007/02/21 01:53:40 fedor Exp $

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"


class MonitorElement;

class PFJetTester : public edm::EDAnalyzer {
public:

  PFJetTester (const edm::ParameterSet&);
  ~PFJetTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 
 private:

  edm::InputTag mInputCollection;
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
};

#endif
