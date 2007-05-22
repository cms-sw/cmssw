// -*- C++ -*-
//
// Package:    StEvtSolutionMaker
// Class:      StEvtSolutionMaker
// 
/**\class TtSemiEvtSolutionMaker TtSemiEvtSolutionMaker.cc AnalysisDataFormats/TopObjectsProducers/src/TtSemiEvtSolutionMaker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Thu May 18 18:11:01 CEST 2006
// $Id: StEvtSolutionMaker.h,v 1.1 2007/05/11 15:33:34 giamman Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEtEtaPhi.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEtThetaPhi.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/StKinFitterEMom.h"
#include "AnalysisDataFormats/TopObjects/interface/BestMatching.h"
//#include "TopQuarkAnalysis/TopJetCombination/interface/TtJetCombinationProbability.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;


//
// class decleration
//

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
      string leptonFlavour_;
      string jetInput_;
      bool addJetCombProb_, doKinFit_, matchToGenEvt_;
      int maxNrIter_;
      double maxDeltaS_, maxF_;
      int param_;
      vector<int> constraints_;
};
