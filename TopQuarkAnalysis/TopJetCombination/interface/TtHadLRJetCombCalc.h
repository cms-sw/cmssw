#ifndef TtHadLRJetCombCalc_h
#define TtHadLRJetCombCalc_h
// $Id: TtHadLRJetCombCalc.h,v 1.0 2007/10/07 12:07:00 mfhansen Exp $
// copied TtSemiLRJetCombCalc.h,v 1.3 2007/06/15 08:53:52 by heyninck


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"
#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"


class TtHadLRJetCombCalc {

  public:
    TtHadLRJetCombCalc();
    TtHadLRJetCombCalc(TString,std::vector<int>);
    ~TtHadLRJetCombCalc();	

    void  operator()(TtHadEvtSolution&);

  private:
    LRHelpFunctions * myLR;
    bool addPurity;
};

#endif
