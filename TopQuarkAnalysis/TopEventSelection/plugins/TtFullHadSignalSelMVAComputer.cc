#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtFullHadSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSelEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/PatCandidates/interface/Flags.h"

TtFullHadSignalSelMVAComputer::TtFullHadSignalSelMVAComputer(const edm::ParameterSet& cfg)
    : mvaToken_(esConsumes()),
      jetsToken_(consumes<std::vector<pat::Jet> >(cfg.getParameter<edm::InputTag>("jets"))),
      putToken_(produces("DiscSel")) {}

void TtFullHadSignalSelMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup) {
  mvaComputer.update(&setup.getData(mvaToken_), "ttFullHadSignalSelMVA");

  const auto& jets = evt.get(jetsToken_);

  //calculation of InputVariables
  //see TopQuarkAnalysis/TopTools/interface/TtFullHadSignalSel.h
  //                             /src/TtFullHadSignalSel.cc
  //all objects, jets, which are needed for the calculation
  //of the input-variables have to be passed to this class
  TtFullHadSignalSel selection(jets);

  double discrim = evaluateTtFullHadSignalSel(mvaComputer, selection);

  evt.emplace(putToken_, discrim);
}

// implement the plugins for the computer container
// -> register TtFullHadSignalSelMVARcd
// -> define TtFullHadSignalSelMVAFileSource
MVA_COMPUTER_CONTAINER_IMPLEMENT(TtFullHadSignalSelMVA);
