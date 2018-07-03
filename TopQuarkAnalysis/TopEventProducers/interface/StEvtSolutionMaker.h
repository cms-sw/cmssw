#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitter.h"
//#include "TopQuarkAnalysis/TopJetCombination/interface/TtJetCombinationProbability.h"

class StEvtSolutionMaker : public edm::EDProducer {
 public:

  explicit StEvtSolutionMaker(const edm::ParameterSet&);
  ~StEvtSolutionMaker() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  StKinFitter * myKinFitter;
  //std::vector<TtJetCombinationProbability> jetCombProbs;
  edm::EDGetTokenT<std::vector<pat::Electron> > electronSrcToken_;
  edm::EDGetTokenT<std::vector<pat::Muon> > muonSrcToken_;
  edm::EDGetTokenT<std::vector<pat::MET> > metSrcToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jetSrcToken_;
  edm::EDGetTokenT<StGenEvent> genEvtSrcToken_;
  std::string leptonFlavour_;
  int jetCorrScheme_;
  // std::string jetInput_;
  // bool addJetCombProb_,
  bool addLRJetComb_, doKinFit_, matchToGenEvt_;
  int maxNrIter_;
  double maxDeltaS_, maxF_;
  int jetParam_, lepParam_, metParam_;
  std::vector<int> constraints_;
};
