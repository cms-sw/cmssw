#ifndef TtSemiJetCombMVAComputer_h
#define TtSemiJetCombMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#ifndef TtSemiJetCombMVARcd_defined  // to avoid conflicts with the TopSemiLepJetCombMVATrainer
#define TtSemiJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiJetCombMVA);  // defines TopSemiLepJetCombMVARcd
#endif

class TtSemiJetCombMVAComputer : public edm::EDProducer {

 public:
  
  explicit TtSemiJetCombMVAComputer(const edm::ParameterSet&);
  ~TtSemiJetCombMVAComputer();
  
 private:

  virtual void beginJob(const edm::EventSetup&);
  virtual void produce(edm::Event& evt, const edm::EventSetup& setup);
  virtual void endJob();

  edm::InputTag leptons_;
  edm::InputTag jets_;

  unsigned int nJetsMax_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
