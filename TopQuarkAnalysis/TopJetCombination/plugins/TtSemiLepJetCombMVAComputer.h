#ifndef TtSemiLepJetCombMVAComputer_h
#define TtSemiLepJetCombMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

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

class TtSemiLepJetCombMVAComputer : public edm::stream::EDProducer<> {
public:
  explicit TtSemiLepJetCombMVAComputer(const edm::ParameterSet&);

private:
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;

  edm::ESGetToken<PhysicsTools::Calibration::MVAComputerContainer, TtSemiLepJetCombMVARcd> mvaToken_;
  edm::EDGetTokenT<edm::View<reco::RecoCandidate>> lepsToken_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetsToken_;
  edm::EDGetTokenT<std::vector<pat::MET>> metsToken_;

  const int maxNJets_;
  const int maxNComb_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
