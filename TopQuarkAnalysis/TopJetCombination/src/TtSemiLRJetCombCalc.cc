//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombCalc.cc,v 1.2 2007/06/09 01:17:40 lowette Exp $
//
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombCalc.h"

// constructor with path; default should not be used
TtSemiLRJetCombCalc::TtSemiLRJetCombCalc(TString fitInputPath, std::vector<int> observables) {
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
    if( myLR->obsFitIncluded(o) ) { std::cout<<"uses obs value of obs"<<o<<std::endl; obsVals.push_back(sol.getLRSignalEvtObsVal(o)); };
  }
  
  // calculate the logLR and the purity
  double logLR = myLR->calcLRval(obsVals);
  double prob  = -999.;
  if(addPurity) myLR->calcProb(logLR);
  
  // fill these values to the members in the TtSemiEvtSolution
  sol.setLRJetCombLRval(logLR);
  sol.setLRJetCombProb(prob);
  std::cout<<"  Found logLR = "<<logLR<<" and Prob = "<<prob<<std::endl;
}
