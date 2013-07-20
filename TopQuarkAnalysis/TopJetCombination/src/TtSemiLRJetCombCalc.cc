//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombCalc.cc,v 1.5 2013/05/28 17:57:05 gartung Exp $
//
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombCalc.h"

// constructor with path; default should not be used
TtSemiLRJetCombCalc::TtSemiLRJetCombCalc(const TString& fitInputPath, const std::vector<int>& observables) {
  std::cout << "=== Constructing a TtSemiLRJetCombCalc... " << std::endl; 
  myLR = new LRHelpFunctions();
  addPurity = false;
  if(observables[0] == -1) addPurity = true;
  myLR -> readObsHistsAndFits(fitInputPath, observables, addPurity);
  std::cout << "=== done." << std::endl;
}


// destructor
TtSemiLRJetCombCalc::~TtSemiLRJetCombCalc() {
  delete myLR;
}


void  TtSemiLRJetCombCalc::operator()(TtSemiEvtSolution & sol){
  
  // find the used observables
  std::vector<double> obsVals;
  for(unsigned int o = 0; o<100; o++){
    if( myLR->obsFitIncluded(o) ) {obsVals.push_back(sol.getLRJetCombObsVal(o)); };
  }
  
  // calculate the logLR and the purity
  double logLR = myLR->calcLRval(obsVals);
  double prob  = -999.;
  if(addPurity) prob = myLR->calcProb(logLR);
  
  // fill these values to the members in the TtSemiEvtSolution
  sol.setLRJetCombLRval(logLR);
  sol.setLRJetCombProb(prob);
}
