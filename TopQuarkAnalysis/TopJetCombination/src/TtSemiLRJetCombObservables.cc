//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombObservables.cc,v 1.6 2007/06/16 13:42:10 heyninck Exp $
//
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombObservables.h"

TtSemiLRJetCombObservables::TtSemiLRJetCombObservables() 
{
}

TtSemiLRJetCombObservables::~TtSemiLRJetCombObservables() 
{
}

void  TtSemiLRJetCombObservables::operator()(TtSemiEvtSolution& sol)
{
  jetCombVarVal.clear();

  //obs1 : pt(had top) 
  double AverageTop =((sol.getHadb().p4()+sol.getHadq().p4()+sol.getHadp().p4()).pt()+(sol.getLepb().p4()+sol.getHadq().p4()+sol.getHadp().p4()).pt()+(sol.getHadb().p4()+sol.getLepb().p4()+sol.getHadp().p4()).pt()+(sol.getHadb().p4()+sol.getHadq().p4()+sol.getLepb().p4()).pt())/4.;
  double Obs1 = ((sol.getHadb().p4()+sol.getHadq().p4()+sol.getHadp().p4()).pt())/AverageTop;
  jetCombVarVal.push_back(std::pair<unsigned int,double>(1,Obs1)); 

  //obs2 : (pt_b1 + pt_b2)/(sum jetpt)
  double obs2 = (sol.getHadb().pt()+sol.getLepb().pt())/(sol.getHadp().pt()+sol.getHadq().pt());
  jetCombVarVal.push_back(std::pair<unsigned int,double>(2,obs2));
  
  //obs3: delta R between lep b and lepton 
  double Obs3 = -10;
  if (sol.getDecay() == "muon")     Obs3 = ROOT::Math::VectorUtil::DeltaR( sol.getLepb().p4(),sol.getRecLepm().p4() );
  if (sol.getDecay() == "electron") Obs3 = ROOT::Math::VectorUtil::DeltaR( sol.getLepb().p4(),sol.getRecLepe().p4() );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(3,Obs3));
  
   //obs4 : del R ( had b, had W)
  double Obs4 = ROOT::Math::VectorUtil::DeltaR( sol.getHadb().p4(), sol.getHadq().p4()+sol.getHadp().p4() );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(4,Obs4));  
   
  //obs5 : del R between light quarkssol.getHadp().phi(
  double Obs5 = ROOT::Math::VectorUtil::DeltaR( sol.getHadq().p4(),sol.getHadp().p4() );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(5,Obs5)); 
  
  //obs6 : b-tagging information
  double Obs6 = 0;
  if ( fabs(sol.getHadb().bDiscriminator("trackCountingJetTags") +10) < 0.0001 || fabs(sol.getLepb().bDiscriminator("trackCountingJetTags") +10)< 0.0001 ){
    Obs6 = -10.;
  } else {
    Obs6 = (sol.getHadb().bDiscriminator("trackCountingJetTags")+sol.getLepb().bDiscriminator("trackCountingJetTags"));
  }
  jetCombVarVal.push_back(std::pair<unsigned int,double>(6,Obs6)); 
   
  //obs7 : chi2 value of kinematical fit with W-mass constraint
  double Obs7 =0;
  if(sol.getProbChi2() <0){Obs7 = -0;} else { Obs7 = log10(sol.getProbChi2()+.00001);}
  jetCombVarVal.push_back(std::pair<unsigned int,double>(7,Obs7)); 
 
  sol.setLRJetCombObservables(jetCombVarVal);
}
