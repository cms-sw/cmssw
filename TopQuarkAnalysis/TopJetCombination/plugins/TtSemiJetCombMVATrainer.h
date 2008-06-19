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

#ifndef TtSemiJetCombMVARcd_defined  // to avoid conflicts with the TtSemiJetCombMVAComputer
#define TtSemiJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiJetCombMVA);  // defines TtSemiJetCombMVARcd
#endif

class TtSemiJetCombMVATrainer : public edm::EDAnalyzer {
  
 public:
  
  explicit TtSemiJetCombMVATrainer(const edm::ParameterSet&);
  ~TtSemiJetCombMVATrainer();
  
 private:
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& setup);

  edm::InputTag leptons_;
  edm::InputTag jets_;
  edm::InputTag matching_;

  unsigned int nJetsMax_;
  
  int lepChannel_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
