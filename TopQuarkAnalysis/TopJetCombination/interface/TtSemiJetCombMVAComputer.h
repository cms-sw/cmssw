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

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#ifndef TtSemiJetCombMVARcd_defined  // to avoid conflicts with the TopSemiLepMuonJetCombMVATrainer
#define TtSemiJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiJetCombMVA);  // defines TopSemiLepMuonJetCombMVARcd
#endif

class TtSemiJetCombMVAComputer : public edm::EDProducer {

 public:
  
  explicit TtSemiJetCombMVAComputer(const edm::ParameterSet&);
  ~TtSemiJetCombMVAComputer();
  
 private:

  virtual void beginJob(const edm::EventSetup&);
  virtual void produce(edm::Event& evt, const edm::EventSetup& setup);
  virtual void endJob();

  typedef std::vector<pat::Muon> TopMuonCollection;
  typedef std::vector<pat::Jet> TopJetCollection;

  edm::InputTag muons_;
  edm::InputTag jets_;

  unsigned int nJetsMax_;
  double discrimCut_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
