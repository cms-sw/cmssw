//
// $Id: TtSemiSimpleBestJetComb.cc,v 1.3 2007/09/20 18:03:20 lowette Exp $
//

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiSimpleBestJetComb.h"

TtSemiSimpleBestJetComb::TtSemiSimpleBestJetComb() 
{
}

TtSemiSimpleBestJetComb::~TtSemiSimpleBestJetComb() 
{
}

int  TtSemiSimpleBestJetComb::operator()(std::vector<TtSemiEvtSolution> & sols)
{ 
  // search the highest probChi^2 value in the among the different jet combination solutions   
  double maxProbChi2 = 0;
  for(unsigned int s=0; s<sols.size(); s++)  maxProbChi2 = std::max(maxProbChi2,sols[s].getProbChi2());
  
  //search indices of original solutions with highest probChi2 value
  std::vector<unsigned int> indices;
  indices.clear();
  for(unsigned int s=0; s<sols.size(); s++){
    if(fabs(sols[s].getProbChi2()-maxProbChi2) < 0.0001) indices.push_back(s);
  }
     
  int bestSol = -999;
  if(maxProbChi2 > 0.){
    if(indices.size() == 1) bestSol = indices[0];
    if(indices.size() == 2) {
      //typically only light jets constraints applied, so still b-jet ambiguity to resolve 
      // -> look at DPhi(Whadr,bhadr) and choose smallest value
      double DPhi_Wb0 = fabs(sols[indices[0]].getFitHadW().phi()-sols[indices[0]].getFitHadb().phi());
      double DPhi_Wb1 = fabs(sols[indices[1]].getFitHadW().phi()-sols[indices[1]].getFitHadb().phi());
      if(DPhi_Wb0>3.1415) DPhi_Wb0 = 2.*3.1415-DPhi_Wb0;
      if(DPhi_Wb1>3.1415) DPhi_Wb1 = 2.*3.1415-DPhi_Wb1;
      if(DPhi_Wb0 < DPhi_Wb1){
         bestSol = indices[0];
      }
      else{
        bestSol = indices[1];
      }
    }
  }
  return bestSol;
}
