#ifndef TtFullHadSignalSelMVATrainer_h
#define TtFullHadSignalSelMVATrainer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#ifndef TtFullHadSignalSelMVARcd_defined  // to avoid conflicts with the TtFullHadSignalSelMVAComputer
#define TtFullHadSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtFullHadSignalSelMVA);  // defines TtFullHadSignalSelMVA
#endif

class TtFullHadSignalSelMVATrainer : public edm::EDAnalyzer {

 public:

  explicit TtFullHadSignalSelMVATrainer(const edm::ParameterSet&);
  ~TtFullHadSignalSelMVATrainer() override;

 private:

  void analyze(const edm::Event& evt, const edm::EventSetup& setup) override;
  void beginJob() override;

  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;

  int whatData_;
  int maxEv_;
  int selEv;
  double weight_;

  PhysicsTools::MVAComputerCache mvaComputer;

};

#endif
