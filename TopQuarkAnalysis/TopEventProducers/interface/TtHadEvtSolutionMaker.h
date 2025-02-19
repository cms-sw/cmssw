#ifndef TtHadEvtSolutionMaker_h
#define TtHadEvtSolutionMaker_h
//
// $Id: TtHadEvtSolutionMaker.h,v 1.8 2010/02/15 13:41:06 snaumann Exp $
// adapted TtSemiEvtSolutionMaker.h, v1.13 2007/07/06 02:49:42 lowette Exp $
// for fully hadronic channel.

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
  ~TtHadEvtSolutionMaker();
  
  virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
  
 private:
  // configurables
  
  edm::InputTag jetSrc_;
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
  // tools
  TtFullHadKinFitter          * myKinFitter;
  TtHadSimpleBestJetComb      * mySimpleBestJetComb;
  TtHadLRJetCombObservables   * myLRJetCombObservables;
  TtHadLRJetCombCalc          * myLRJetCombCalc;
  TtHadLRSignalSelObservables * myLRSignalSelObservables;
  TtHadLRSignalSelCalc        * myLRSignalSelCalc;
};


#endif
