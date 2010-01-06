#ifndef TtSemiLepJetCombMVAComputer_h
#define TtSemiLepJetCombMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#ifndef TtSemiLepJetCombMVARcd_defined  // to avoid conflicts with the TtSemiLepJetCombMVATrainer
#define TtSemiLepJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepJetCombMVA);  // defines TtSemiLepJetCombMVARcd
#endif

class TtSemiLepJetCombMVAComputer : public edm::EDProducer {

 public:
  
  explicit TtSemiLepJetCombMVAComputer(const edm::ParameterSet&);
  ~TtSemiLepJetCombMVAComputer();
  
 private:

  virtual void beginJob();
  virtual void produce(edm::Event& evt, const edm::EventSetup& setup);
  virtual void endJob();

  edm::InputTag leps_;  
  edm::InputTag jets_;
  edm::InputTag mets_;

  int maxNJets_;
  int maxNComb_;

  PhysicsTools::MVAComputerCache mvaComputer;

};

#endif
