#include "TopQuarkAnalysis/TopJetCombination/interface/TtHadSimpleBestJetComb.h"
//
// $Id: TtHadSimpleBestJetComb.cc,v 1.3 2008/02/17 11:27:55 rwolf Exp $
// adapted Id: TtSemiSimpleBestJetComb.cc,v 1.2 2007/06/09 01:17:40 lowette Exp 
// for fully hadronic channel

/**
  \class    TtHadSimpleBestJetComb
  \brief    Based on the TtSemiSimpleBestJetComb.by: Jan Heyninck
   version: TtSemiSimpleBestJetComb.cc,v 1.2 2007/06/09 01:17:40 lowette Exp 
  
   Simple method to get the correct jet combination in hadronic ttbar events.   
   Returns the solution with the highest probChi^2 value, starting from TtHadEvtSolutions.
   If more than one solution has a low enough Chi^2, the chosen best solution is that which
   either have both b-jets closer to a particular W-jet or the one in which the two angles added 
   together. WARNING WARNING WARNING, last option for selection best solution by taking the sum
   of angles needs to be checked/approved by experts!!!!
*/

TtHadSimpleBestJetComb::TtHadSimpleBestJetComb() 
{
}

TtHadSimpleBestJetComb::~TtHadSimpleBestJetComb() 
{
}

int TtHadSimpleBestJetComb::operator()(std::vector<TtHadEvtSolution> & sols)
{ 
  // search the highest probChi^2 value in the among the different jet combination solutions   
  double maxProbChi2 = 0.;
  for(unsigned int s=0; s<sols.size(); s++){
    maxProbChi2 = std::max(maxProbChi2,sols[s].getProbChi2());
  }

  //search indices of original solutions with highest probChi2 value and select those solutionsthat are close to the highest Chi^2 
  std::vector<unsigned int> indices;
  indices.clear();  
  for(unsigned int s=0; s<sols.size(); s++){  
    if(fabs(sols[s].getProbChi2()-maxProbChi2) < 0.0001) indices.push_back(s); 
  } 

  // TtHadSolutionMaker takes light jet combinations into account, but not b-jet ambiguity...
  int bestSol = -999;
  double prev_W1b = 999.;
  double prev_W2b = 999.;
  if(maxProbChi2 > 0.){
    if(indices.size() == 1) bestSol = indices[0];
    if(indices.size() > 1){ //for more than one solution...
      for(unsigned int i=0;i!=indices.size();i++){
	double DPhi_W1b0 = fabs(sols[indices[i]].getFitHadW_plus().phi()-sols[indices[i]].getFitHadb().phi());
	double DPhi_W1b1 = fabs(sols[indices[i]].getFitHadW_plus().phi()-sols[indices[i]].getFitHadbbar().phi());
	double DPhi_W2b0 = fabs(sols[indices[i]].getFitHadW_minus().phi()-sols[indices[i]].getFitHadb().phi());
	double DPhi_W2b1 = fabs(sols[indices[i]].getFitHadW_minus().phi()-sols[indices[i]].getFitHadbbar().phi());
	
	if(DPhi_W1b0>3.1415) DPhi_W1b0 = 2.*3.1415-DPhi_W1b0;
	if(DPhi_W1b1>3.1415) DPhi_W1b1 = 2.*3.1415-DPhi_W1b1;
	if(DPhi_W2b0>3.1415) DPhi_W2b0 = 2.*3.1415-DPhi_W2b0;
	if(DPhi_W2b1>3.1415) DPhi_W2b1 = 2.*3.1415-DPhi_W2b1;
	// Select as best solution the one which either has both b-jets closer to a particular W-jet
	// or the one in which the two angles added together are lower than the other.....FIXME!!!
	// W1b0 and W1b1 is a pair, W2b0 and W2b1
	if(DPhi_W1b0<DPhi_W2b0 && DPhi_W1b1<DPhi_W2b1){
	  if(DPhi_W1b0<prev_W1b && DPhi_W1b1<prev_W2b){
	    bestSol = indices[i];
	  }
	}
	if(DPhi_W1b0>DPhi_W2b0 && DPhi_W1b1>DPhi_W2b1){
	  if(DPhi_W2b0<prev_W1b && DPhi_W2b1<prev_W2b){
	    bestSol = indices[i];
	  }
	}
	if((DPhi_W1b0<DPhi_W2b0 && DPhi_W1b1>DPhi_W2b1)||(DPhi_W1b0>DPhi_W2b0 && DPhi_W1b1<DPhi_W2b1)){
	  if((DPhi_W1b0+DPhi_W1b1)<(DPhi_W2b0+DPhi_W2b1)){
	    if(DPhi_W1b0<prev_W1b && DPhi_W1b1<prev_W2b){
	      bestSol = indices[i];
	    }  
	  }else{
	    if(DPhi_W2b0<prev_W1b && DPhi_W2b1<prev_W2b){
	      bestSol = indices[i];
	    }
	  }
	}
      }
    }
  }
  return bestSol;
}
