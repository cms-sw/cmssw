//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRSignalSelCalc.h,v 1.2 2008/02/17 11:18:54 rwolf Exp $
//

#ifndef TtSemiLRSignalSelCalc_h
#define TtSemiLRSignalSelCalc_h

/**
  \class    TtSemiLRSignalSelCalc TtSemiLRSignalSelCalc.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtSemiLRSignalSelCalc.h"
  \brief    Class to calculate the jet combination LR value and purity from a root-file with fit functions

  \author   Jan Heyninck
  \version  $Id: TtSemiLRSignalSelCalc.h,v 1.2 2008/02/17 11:18:54 rwolf Exp $
*/

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"

class TtSemiLRSignalSelCalc {

 public:
  
  TtSemiLRSignalSelCalc();
  TtSemiLRSignalSelCalc(TString,std::vector<int>);
  ~TtSemiLRSignalSelCalc();	
  
  void  operator()(TtSemiEvtSolution&);
  
 private:

  LRHelpFunctions  * myLR;
  bool addPurity;
};

#endif
