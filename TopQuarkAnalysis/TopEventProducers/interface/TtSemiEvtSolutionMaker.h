//
//

#ifndef TopEventProducers_TtSemiEvtSolutionMaker_h
#define TopEventProducers_TtSemiEvtSolutionMaker_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"

#include <vector>
#include <string>

class TtSemiSimpleBestJetComb;
class TtSemiLRJetCombObservables;
class TtSemiLRJetCombCalc;
class TtSemiLRSignalSelObservables;
class TtSemiLRSignalSelCalc;


class TtSemiEvtSolutionMaker : public edm::EDProducer {

 public:

  explicit TtSemiEvtSolutionMaker(const edm::ParameterSet & iConfig);
  ~TtSemiEvtSolutionMaker() override;

  void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

  // convert unsigned to Param
  TtSemiLepKinFitter::Param param(unsigned);
  // convert unsigned to Param
  TtSemiLepKinFitter::Constraint constraint(unsigned);
  // convert unsigned to Param
  std::vector<TtSemiLepKinFitter::Constraint> constraints(std::vector<unsigned>&);

 private:

  // configurables
  edm::EDGetTokenT<std::vector<pat::Electron> > electronSrcToken_;
  edm::EDGetTokenT<std::vector<pat::Muon> > muonSrcToken_;
  edm::EDGetTokenT<std::vector<pat::MET> > metSrcToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jetSrcToken_;
  std::string leptonFlavour_;
  int jetCorrScheme_;
  unsigned int nrCombJets_;
  std::string lrSignalSelFile_, lrJetCombFile_;
  bool addLRSignalSel_, addLRJetComb_, doKinFit_, matchToGenEvt_;
  int matchingAlgo_;
  bool useMaxDist_, useDeltaR_;
  double maxDist_;
  int maxNrIter_;
  double maxDeltaS_, maxF_;
  int jetParam_, lepParam_, metParam_;
  std::vector<int> lrSignalSelObs_, lrJetCombObs_;
  std::vector<unsigned> constraints_;
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;
  // tools
  TtSemiLepKinFitter           * myKinFitter;
  TtSemiSimpleBestJetComb      * mySimpleBestJetComb;
  TtSemiLRJetCombObservables   * myLRJetCombObservables;
  TtSemiLRJetCombCalc          * myLRJetCombCalc;
  TtSemiLRSignalSelObservables * myLRSignalSelObservables;
  TtSemiLRSignalSelCalc        * myLRSignalSelCalc;
};


#endif
