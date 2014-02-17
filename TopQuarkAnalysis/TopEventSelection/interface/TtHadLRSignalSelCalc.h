//
// $Id: TtHadLRSignalSelCalc.h,v 1.2 2008/02/17 11:18:53 rwolf Exp $
// Adapted TtHadLRSignalSelCalc.h,v 1.1 2007/06/15 08:49:19 heyninck Exp 
// for fully hadronic channel

#ifndef TtHadLRSignalSelCalc_h
#define TtHadLRSignalSelCalc_h

/**
  \class    TtHadLRSignalSelCalc TtHadLRSignalSelCalc.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtHadLRSignalSelCalc.h"
  \brief    Class to calculate the jet combination LR value and purity from a root-file with fit functions

  \author   Jan Heyninck - adapted hadronic version mfhansen
*/

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

class TtHadLRSignalSelCalc {

 public:
  
  TtHadLRSignalSelCalc();
  TtHadLRSignalSelCalc(TString,std::vector<int>);
  ~TtHadLRSignalSelCalc();	
  void  operator()(TtHadEvtSolution&);
  
 private:

  LRHelpFunctions  * myLR;
  bool addPurity;
};

#endif
