#ifndef TtSemiLepSignalSelMVATrainer_h
#define TtSemiLepSignalSelMVATrainer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#ifndef TtSemiLepSignalSelMVARcd_defined  // to avoid conflicts with the TtSemiSignalSelMVAComputer
#define TtSemiLepSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepSignalSelMVA);  // defines TtSemiLepSignalSelMVARcd
#endif

class TtSemiLepSignalSelMVATrainer : public edm::EDAnalyzer {
  
 public:
  
  explicit TtSemiLepSignalSelMVATrainer(const edm::ParameterSet&);
  ~TtSemiLepSignalSelMVATrainer();
  
 private:
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& setup);

  edm::InputTag leptons_;
  edm::InputTag jets_;
  edm::InputTag METs_;

  unsigned int maxNJets_;
  
  int lepChannel_;

  PhysicsTools::MVAComputerCache mvaComputer;

  // compare two jets in ET
  struct CompareJetET {
    bool operator()( pat::Jet j1, pat::Jet j2 ) const
    {
      return j1.et() > j2.et();
    }
  };
  
  CompareJetET JetETComparison;

  // compare two muons in ET
  struct CompareLeptonET {
    bool operator()( reco::RecoCandidate* lep1, reco::RecoCandidate* lep2 ) const
    {
      return lep1->et() > lep2->et();
    }
  };
  
  CompareLeptonET LeptonETComparison;
};

#endif
