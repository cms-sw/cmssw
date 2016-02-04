// $Id: TtHadLRJetCombObservables.cc,v 1.2 2008/02/17 11:27:55 rwolf Exp $

#include "TopQuarkAnalysis/TopJetCombination/interface/TtHadLRJetCombObservables.h"

// constructor with path; default should not be used
TtHadLRJetCombObservables::TtHadLRJetCombObservables() 
{
}

TtHadLRJetCombObservables::~TtHadLRJetCombObservables() 
{
}

void  TtHadLRJetCombObservables::operator()(TtHadEvtSolution& sol)
{
  jetCombVarVal.clear();

  //observable 1 : pt(had top)
  //Calculate the average pt for all possible combinations of light jets with the two b-jets
  double AverageTop =((sol.getHadb().p4()+sol.getHadq().p4()+sol.getHadp().p4()).pt()+
		      (sol.getHadbbar().p4()+sol.getHadq().p4()+sol.getHadp().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadbbar().p4()+sol.getHadp().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadbbar().p4()+sol.getHadq().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadk().p4()+sol.getHadj().p4()).pt()+
		      (sol.getHadbbar().p4()+sol.getHadk().p4()+sol.getHadj().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadbbar().p4()+sol.getHadj().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadbbar().p4()+sol.getHadk().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadq().p4()+sol.getHadj().p4()).pt()+
		      (sol.getHadb().p4()+sol.getHadq().p4()+sol.getHadk().p4()).pt()+
		      (sol.getHadbbar().p4()+sol.getHadq().p4()+sol.getHadj().p4()).pt()+
		      (sol.getHadbbar().p4()+sol.getHadq().p4()+sol.getHadk().p4()).pt())/12.;

  double Obs1 = ((sol.getHadb().p4()+sol.getHadq().p4()+sol.getHadp().p4()+sol.getHadbbar().p4()+sol.getHadk().p4()+sol.getHadj().p4()).pt())/AverageTop;
  jetCombVarVal.push_back(std::pair<unsigned int,double>(1,Obs1)); 

  //observable 2 : (pt_b1 + pt_b2)/(sum jetpt)
  double obs2 = (sol.getHadb().pt()+sol.getHadbbar().pt())/(sol.getHadp().pt()+sol.getHadq().pt()+sol.getHadj().pt()+sol.getHadk().pt());
  jetCombVarVal.push_back(std::pair<unsigned int,double>(2,obs2));
  
  //observable 3 and 4: delta R between had b and had W and delta R between had bbar and had W
  double Obs3 = ROOT::Math::VectorUtil::DeltaR( sol.getHadb().p4(),(sol.getHadq().p4()+sol.getHadp().p4()) );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(3,Obs3));
 
  double Obs4 = ROOT::Math::VectorUtil::DeltaR( sol.getHadbbar().p4(),(sol.getHadk().p4()+sol.getHadj().p4()) );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(4,Obs4));  
   
  //observalbe 5 and 6: delta R between light quarks pq and jk
  double Obs5 = ROOT::Math::VectorUtil::DeltaR( sol.getHadq().p4(),sol.getHadp().p4() );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(5,Obs5)); 
  
  double Obs6 = ROOT::Math::VectorUtil::DeltaR( sol.getHadk().p4(),sol.getHadj().p4() );
  jetCombVarVal.push_back(std::pair<unsigned int,double>(6,Obs6)); 

  //observable 7: b-tagging information
  double Obs7 = 0;
  if ( fabs(sol.getHadb().bDiscriminator("trackCountingJetTags") +10) < 0.0001 || fabs(sol.getHadbbar().bDiscriminator("trackCountingJetTags") +10)< 0.0001 ){
    Obs7 = -10.;
  } else {
    Obs7 = (sol.getHadb().bDiscriminator("trackCountingJetTags")+sol.getHadbbar().bDiscriminator("trackCountingJetTags"));
  }
  jetCombVarVal.push_back(std::pair<unsigned int,double>(7,Obs7)); 
   
  //observable 8 : chi2 value of kinematical fit with W-mass constraint
  double Obs8 =0;
  if(sol.getProbChi2() <0){
    Obs8 = -0;
  } else { 
    Obs8 = log10(sol.getProbChi2()+.00001);
  }
  jetCombVarVal.push_back(std::pair<unsigned int,double>(8,Obs8)); 
 
  sol.setLRJetCombObservables(jetCombVarVal);
}
