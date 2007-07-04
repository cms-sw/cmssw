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
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEtEtaPhi.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEtThetaPhi.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEMom.h"
//#include "TopQuarkAnalysis/TopJetCombination/interface/TtJetCombinationProbability.h"

class StEvtSolutionMaker : public edm::EDProducer {
   public:
      explicit StEvtSolutionMaker(const edm::ParameterSet&);
      ~StEvtSolutionMaker();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      StKinFitterEtThetaPhi * myKinFitterEtThetaPhi;
      StKinFitterEtEtaPhi   * myKinFitterEtEtaPhi;
      StKinFitterEMom       * myKinFitterEMom;
      //std::vector<TtJetCombinationProbability> jetCombProbs;
      edm::InputTag electronSrc_;
      edm::InputTag muonSrc_;
      edm::InputTag metSrc_;
      edm::InputTag lJetSrc_;
      edm::InputTag bJetSrc_;
      std::string leptonFlavour_;
  //      std::string jetInput_;
  //      bool addJetCombProb_, 
      bool addLRJetComb_, doKinFit_, matchToGenEvt_;
      int maxNrIter_;
      double maxDeltaS_, maxF_;
      int param_;
      std::vector<int> constraints_;
};
