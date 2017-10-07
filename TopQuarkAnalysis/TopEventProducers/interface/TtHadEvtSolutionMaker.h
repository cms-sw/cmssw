#ifndef TtHadEvtSolutionMaker_h
#define TtHadEvtSolutionMaker_h
//
// adapted TtSemiEvtSolutionMaker.h, v1.13 2007/07/06 02:49:42 lowette Exp $
// for fully hadronic channel.

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

#include <vector>
#include <string>


class TtFullHadKinFitter;
class TtHadSimpleBestJetComb;
class TtHadLRJetCombObservables;
class TtHadLRJetCombCalc;
class TtHadLRSignalSelObservables;
class TtHadLRSignalSelCalc;

class TtHadEvtSolutionMaker : public edm::EDProducer {

 public:

  explicit TtHadEvtSolutionMaker(const edm::ParameterSet & iConfig);
  ~TtHadEvtSolutionMaker() override;

  void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

 private:
  // configurables

  edm::EDGetTokenT<std::vector<pat::Jet> > jetSrcToken_;
  int jetCorrScheme_;
  std::string lrSignalSelFile_, lrJetCombFile_;
  bool addLRSignalSel_, addLRJetComb_, doKinFit_, matchToGenEvt_;
  int matchingAlgo_;
  bool useMaxDist_, useDeltaR_;
  double maxDist_;
  int maxNrIter_;
  double maxDeltaS_, maxF_;
  int jetParam_;
  std::vector<int> lrSignalSelObs_, lrJetCombObs_;
  std::vector<unsigned int> constraints_;
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;
  // tools
  TtFullHadKinFitter          * myKinFitter;
  TtHadSimpleBestJetComb      * mySimpleBestJetComb;
  TtHadLRJetCombObservables   * myLRJetCombObservables;
  TtHadLRJetCombCalc          * myLRJetCombCalc;
  TtHadLRSignalSelObservables * myLRSignalSelObservables;
  TtHadLRSignalSelCalc        * myLRSignalSelCalc;
};


#endif
