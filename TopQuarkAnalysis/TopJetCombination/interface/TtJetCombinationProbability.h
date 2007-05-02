#ifndef TtJetCombinationProbability_h
#define TtJetCombinationProbability_h


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Utilities/General/interface/envUtil.h"
// root stuff
#include "TFile.h"
#include "TKey.h"
#include "TH1.h"
#include "TF1.h"
#include "TString.h"

// General C++ stuff
#include <iostream>
#include <string>

//own code"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtJetCombLRsetup.h"


class TtJetCombinationProbability {
  
  public:

    TtJetCombinationProbability();
    TtJetCombinationProbability(std::string);
    ~TtJetCombinationProbability();	

    double  getPTrueCombExist(TtSemiEvtSolution * sol);
    double  getPTrueBJetSel(TtSemiEvtSolution * sol);
    double  getPTrueBhadrSel(TtSemiEvtSolution * sol);
    
    
  private:
    // debug output switch
    TF1 combBTagFit, fBSelPurity, fBhadrPurity;
    vector<TF1> fBSelObs, fBhadrObs;
    double PTrueCombExist;
};

#endif
