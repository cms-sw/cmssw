// $Id: TtHadLRJetCombCalc.cc,v 1.2 2008/02/17 11:27:55 rwolf Exp $
// copied TtSemiLRJetCombCalc.cc,v 1.4 2007/06/18 14:08:16 by heyninck 
//
#include "TopQuarkAnalysis/TopJetCombination/interface/TtHadLRJetCombCalc.h"

// constructor with path; default should not be used
TtHadLRJetCombCalc::TtHadLRJetCombCalc(const TString& fitInputPath, const std::vector<int>& observables) 
{
  std::cout << "=== Constructing a TtHadLRJetCombCalc... " << std::endl; 
  myLR = new LRHelpFunctions();
  addPurity = false;
  if(observables[0] == -1) addPurity = true;
  myLR -> readObsHistsAndFits(fitInputPath, observables, addPurity);
  std::cout << "=== done." << std::endl;
}

TtHadLRJetCombCalc::~TtHadLRJetCombCalc() 
{
  delete myLR;
}

void  TtHadLRJetCombCalc::operator()(TtHadEvtSolution & sol)
{  
  // find the used observables
  std::vector<double> obsVals;
  for(unsigned int o = 0; o<100; o++){
    if( myLR->obsFitIncluded(o) ) {obsVals.push_back(sol.getLRJetCombObsVal(o)); };
  }
  
  // calculate the logLR and the purity
  double logLR = myLR->calcLRval(obsVals);
  double prob  = -999.;
  if(addPurity) prob = myLR->calcProb(logLR);
  
  // fill these values to the members in the TtHadEvtSolution
  sol.setLRJetCombLRval(logLR);
  sol.setLRJetCombProb(prob);
}
