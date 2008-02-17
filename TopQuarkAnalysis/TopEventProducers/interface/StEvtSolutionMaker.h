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
  ~StEvtSolutionMaker();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  StKinFitter * myKinFitter;
  //std::vector<TtJetCombinationProbability> jetCombProbs;
  edm::InputTag electronSrc_;
  edm::InputTag muonSrc_;
  edm::InputTag metSrc_;
  edm::InputTag jetSrc_;
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
