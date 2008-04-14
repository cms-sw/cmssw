#ifndef TtSemiJetCombMVATrainer_h
#define TtSemiJetCombMVATrainer_h

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
#include "DataFormats/PatCandidates/interface/Muon.h"

#ifndef TtSemiJetCombMVARcd_defined  // to avoid conflicts with the TopSemiLepMuonJetCombMVAComputer
#define TtSemiJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiJetCombMVA);  // defines TopSemiLepMuonJetCombMVARcd
#endif

class TtSemiJetCombMVATrainer : public edm::EDAnalyzer {
  
 public:
  
  explicit TtSemiJetCombMVATrainer(const edm::ParameterSet&);
  ~TtSemiJetCombMVATrainer();
  
 private:
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& setup);
  
  typedef std::vector<pat::Muon> TopMuonCollection;
  typedef std::vector<pat::Jet> TopJetCollection;

  edm::InputTag muons_;
  edm::InputTag jets_;
  edm::InputTag matching_;

  unsigned int nJetsMax_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
