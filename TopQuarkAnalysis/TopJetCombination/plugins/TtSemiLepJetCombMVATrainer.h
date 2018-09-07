#ifndef TtSemiLepJetCombMVATrainer_h
#define TtSemiLepJetCombMVATrainer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#ifndef TtSemiLepJetCombMVARcd_defined  // to avoid conflicts with the TtSemiLepJetCombMVAComputer
#define TtSemiLepJetCombMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepJetCombMVA);  // defines TtSemiLepJetCombMVARcd
#endif

class TtSemiLepJetCombMVATrainer : public edm::EDAnalyzer {

 public:

  explicit TtSemiLepJetCombMVATrainer(const edm::ParameterSet&);
  ~TtSemiLepJetCombMVATrainer() override;

 private:

  void beginJob() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& setup) override;
  void endJob() override;

  WDecay::LeptonType readLeptonType(const std::string& str);

  edm::EDGetTokenT<TtGenEvent> genEvtToken_;
  edm::EDGetTokenT< edm::View<reco::RecoCandidate> > lepsToken_;
  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT< std::vector<pat::MET> > metsToken_;
  edm::EDGetTokenT< std::vector< std::vector<int> > > matchingToken_;

  int maxNJets_;

  WDecay::LeptonType leptonType_;

  PhysicsTools::MVAComputerCache mvaComputer;

  unsigned int nEvents[5];
};

#endif
