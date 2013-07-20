// $Id: TtHadLRSignalSelCalc.cc,v 1.3 2013/05/28 17:54:37 gartung Exp $
// copied TtSemiLRSignalSelCalc.cc,v 1.2 2007/06/18 14:12:18 heyninck Exp 
// for fully hadronic channel

#include "TopQuarkAnalysis/TopEventSelection/interface/TtHadLRSignalSelCalc.h"

// constructor with path; default should not be used
TtHadLRSignalSelCalc::TtHadLRSignalSelCalc(const TString& fitInputPath, const std::vector<int>& observables) 
{
  std::cout << "=== Constructing a TtHadLRSignalSelCalc... " << std::endl; 
  myLR = new LRHelpFunctions();
  addPurity = false;
  if(observables[0] == -1) addPurity = true;
  myLR -> readObsHistsAndFits(fitInputPath, observables, addPurity);
  std::cout << "=== done." << std::endl;
}

TtHadLRSignalSelCalc::~TtHadLRSignalSelCalc() 
{
  delete myLR;
}

void  TtHadLRSignalSelCalc::operator()(TtHadEvtSolution & sol)
{
  // find the used observables
  std::vector<double> obsVals;
  for(unsigned int o = 0; o<100; o++){
    if( myLR->obsFitIncluded(o) ) obsVals.push_back(sol.getLRSignalEvtObsVal(o)); 
  }
  
  // calculate the logLR and the purity
  double logLR = myLR->calcLRval(obsVals);
  double prob  = -999.;
  if(addPurity) prob = myLR->calcProb(logLR);
  
  // fill these values to the members in the TtHadEvtSolution
  sol.setLRSignalEvtLRval(logLR);
  sol.setLRSignalEvtProb(prob);
}
