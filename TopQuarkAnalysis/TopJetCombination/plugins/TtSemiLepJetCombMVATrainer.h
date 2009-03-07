#ifndef TtSemiLepJetCombMVATrainer_h
#define TtSemiLepJetCombMVATrainer_h

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

#ifndef TtSemiLepJetCombMVARcd_defined  // to avoid conflicts with the TtSemiLepJetCombMVAComputer
#define TtSemiLepJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepJetCombMVA);  // defines TtSemiLepJetCombMVARcd
#endif

class TtSemiLepJetCombMVATrainer : public edm::EDAnalyzer {
  
 public:
  
  explicit TtSemiLepJetCombMVATrainer(const edm::ParameterSet&);
  ~TtSemiLepJetCombMVATrainer();
  
 private:
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& setup);

  edm::InputTag leptons_;
  edm::InputTag jets_;
  edm::InputTag mets_;
  edm::InputTag matching_;


  unsigned int maxNJets_;
  
  int lepChannel_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
