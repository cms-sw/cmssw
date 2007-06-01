//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombCalc.cc,v 1.1 2007/05/08 14:03:05 heyninck Exp $
//
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombCalc.h"

// constructor with path; default should not be used
TtSemiLRJetCombCalc::TtSemiLRJetCombCalc(TString fitInputPath) {

  cout << "=== Constructing a TtSemiLRJetCombCalc... " << endl; 
  myLR = new LRHelpFunctions();
  myLR -> readObsHistsAndFits(fitInputPath, true);
  cout << "=== done." << endl;

}


// destructor
TtSemiLRJetCombCalc::~TtSemiLRJetCombCalc() {
  delete myLR;
}


void  TtSemiLRJetCombCalc::operator()(TtSemiEvtSolution & sol){
  
  // find the used observables
  vector<double> selObsVals;
  unsigned int o=0;
  while(!(sol.getLRCorrJetCombVar(o)< 0.5)){
    if(myLR->isIncluded((int) sol.getLRCorrJetCombVar(o))) {cout<<"  obs "<<sol.getLRCorrJetCombVar(o)<<" was selected..."<<endl; selObsVals.push_back(sol.getLRCorrJetCombVal(o));};
    ++o;
  }
  
  // calculate the logLR and the purity
  double logLR = myLR->calcLRval(selObsVals);
  double prob  = myLR->calcProb(logLR);
  
  // fill these values to the members in the TtSemiEvtSolution
  sol.setLRCorrJetCombLRval(logLR);
  sol.setLRCorrJetCombProb(prob);
  cout<<"  Found logLR = "<<logLR<<" and Prob = "<<prob<<endl;
}
