// -*- C++ -*-
//
// Package:    TtSemiEvtSolutionMaker
// Class:      TtSemiEvtSolutionMaker
// 
/**\class TtSemiEvtSolutionMaker TtSemiEvtSolutionMaker.cc AnalysisDataFormats/TopObjectsProducers/src/TtSemiEvtSolutionMaker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Thu May 18 18:11:01 CEST 2006
// $Id: TtSemiEvtSolutionMaker.h,v 1.9 2007/06/10 08:55:01 lowette Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiKinFitterEtEtaPhi.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiKinFitterEtThetaPhi.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiKinFitterEMom.h"
#include "AnalysisDataFormats/TopObjects/interface/BestMatching.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiSimpleBestJetComb.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLRSignalSelObservables.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLRSignalSelCalc.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombObservables.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombCalc.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>


//
// class decleration
//

class TtSemiEvtSolutionMaker : public edm::EDProducer {
   public:
      explicit TtSemiEvtSolutionMaker(const edm::ParameterSet&);
      ~TtSemiEvtSolutionMaker();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:

      edm::InputTag electronSrc_;
      edm::InputTag muonSrc_;
      edm::InputTag metSrc_;
      edm::InputTag lJetSrc_;
      edm::InputTag bJetSrc_;
      std::string leptonFlavour_;
      std::string lrSignalSelFile_, lrJetCombFile_;
      bool addLRSignalSel_, addLRJetComb_, doKinFit_, matchToGenEvt_;
      int maxNrIter_;
      double maxDeltaS_, maxF_;
      int param_;
      std::vector<int> lrSignalSelObs_, lrJetCombObs_, constraints_;
      TtSemiKinFitterEtThetaPhi     *myKinFitterEtThetaPhi;
      TtSemiKinFitterEtEtaPhi       *myKinFitterEtEtaPhi;
      TtSemiKinFitterEMom           *myKinFitterEMom;
      TtSemiSimpleBestJetComb       *mySimpleBestJetComb;
      TtSemiLRSignalSelObservables  *myLRSignalSelObservables;
      TtSemiLRJetCombObservables    *myLRJetCombObservables;
      TtSemiLRSignalSelCalc	    *myLRSignalSelCalc;
      TtSemiLRJetCombCalc	    *myLRJetCombCalc;

};
