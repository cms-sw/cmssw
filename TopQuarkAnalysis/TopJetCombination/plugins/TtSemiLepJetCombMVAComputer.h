#ifndef TtSemiLepJetCombMVAComputer_h
#define TtSemiLepJetCombMVAComputer_h

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

#ifndef TtSemiLepJetCombMVARcd_defined  // to avoid conflicts with the TopSemiLepJetCombMVATrainer
#define TtSemiLepJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepJetCombMVA);  // defines TopSemiLepJetCombMVARcd
#endif

class TtSemiLepJetCombMVAComputer : public edm::EDProducer {

 public:
  
  explicit TtSemiLepJetCombMVAComputer(const edm::ParameterSet&);
  ~TtSemiLepJetCombMVAComputer();
  
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
