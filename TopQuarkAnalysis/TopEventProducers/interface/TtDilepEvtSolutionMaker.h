// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtDilepKinSolver.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>


//
// class decleration
//

class TtDilepEvtSolutionMaker : public edm::EDProducer {
   public:
      explicit TtDilepEvtSolutionMaker(const edm::ParameterSet&);
      ~TtDilepEvtSolutionMaker();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
   
      //next methods are avoidable but they make the code legible
      bool PTComp(TopElectron e, TopMuon m);
      bool LepDiffCharge(TopElectron e, TopMuon m);
      bool LepDiffCharge(TopElectron e1, TopElectron e2);
      bool LepDiffCharge(TopMuon m1, TopMuon m2);
      bool HasPositiveCharge(TopMuon m);
      bool HasPositiveCharge(TopElectron e);
         
      std::string jetInput_;
      bool matchToGenEvt_, calcTopMass_;
      bool eeChannel_, emuChannel_, mumuChannel_;
      double tmassbegin_, tmassend_, tmassstep_;
   
};
