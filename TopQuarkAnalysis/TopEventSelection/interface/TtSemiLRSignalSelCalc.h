//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRSignalSelCalc.h,v 1.3 2013/05/28 17:54:36 gartung Exp $
//

#ifndef TtSemiLRSignalSelCalc_h
#define TtSemiLRSignalSelCalc_h

/**
  \class    TtSemiLRSignalSelCalc TtSemiLRSignalSelCalc.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtSemiLRSignalSelCalc.h"
  \brief    Class to calculate the jet combination LR value and purity from a root-file with fit functions

  \author   Jan Heyninck
  \version  $Id: TtSemiLRSignalSelCalc.h,v 1.3 2013/05/28 17:54:36 gartung Exp $
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
  TtSemiLRSignalSelCalc(const TString&,const std::vector<int>&);
  ~TtSemiLRSignalSelCalc();	
  
  void  operator()(TtSemiEvtSolution&);
  
 private:

  LRHelpFunctions  * myLR;
  bool addPurity;
};

#endif
