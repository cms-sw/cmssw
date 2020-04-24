#ifndef TtSemiLepJetCombMVAComputer_h
#define TtSemiLepJetCombMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetCombEval.h"

#ifndef TtSemiLepJetCombMVARcd_defined  // to avoid conflicts with the TtSemiLepJetCombMVATrainer
#define TtSemiLepJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepJetCombMVA);  // defines TtSemiLepJetCombMVARcd
#endif

class TtSemiLepJetCombMVAComputer : public edm::EDProducer {

 public:

  explicit TtSemiLepJetCombMVAComputer(const edm::ParameterSet&);
  ~TtSemiLepJetCombMVAComputer() override;

 private:

  void beginJob() override;
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;
  void endJob() override;

  edm::EDGetTokenT< edm::View<reco::RecoCandidate>> lepsToken_;
  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT< std::vector<pat::MET> > metsToken_;

  int maxNJets_;
  int maxNComb_;

  PhysicsTools::MVAComputerCache mvaComputer;

};

#endif
