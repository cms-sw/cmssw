//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombObservables.cc,v 1.1 2007/05/08 14:03:05 heyninck Exp $
//
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombObservables.h"

// constructor with path; default should not be used
TtSemiLRJetCombObservables::TtSemiLRJetCombObservables() {}


// destructor
TtSemiLRJetCombObservables::~TtSemiLRJetCombObservables() {}


// member function to add observables to the event
void  TtSemiLRJetCombObservables::operator()(TtSemiEvtSolution& sol){
  jetCombVarVal.clear();

  //obs1 : pt(had top)
  double obs1 = sol.getHadb().pt()+sol.getHadq().pt()+sol.getHadp().pt();
  jetCombVarVal.push_back(pair<double,double>(1,obs1));
  
  //obs2 : (pt_b1 + pt_b2)/(sum jetpt)
  double obs2 = (sol.getHadb().pt()+sol.getLepb().pt())/(sol.getHadp().pt()+sol.getHadq().pt()+sol.getHadb().pt()+sol.getLepb().pt());
  jetCombVarVal.push_back(pair<double,double>(2,obs2));

  //obs3: delta R between had top and lep b  
  double etadiff = (sol.getHadq().p4()+sol.getHadp().p4()+sol.getHadb().p4()).eta() - sol.getLepb().eta();
  double phidiff = fabs((sol.getHadq().p4()+sol.getHadp().p4()+sol.getHadb().p4()).phi() - sol.getLepb().phi());
  if(phidiff>3.14159) phidiff = 2.*3.14159 - phidiff;
  double obs3 = sqrt(pow(etadiff,2)+pow(phidiff,2));
  jetCombVarVal.push_back(pair<double,double>(3,obs3));

  sol.setLRCorrJetCombVarVal(jetCombVarVal);
}
