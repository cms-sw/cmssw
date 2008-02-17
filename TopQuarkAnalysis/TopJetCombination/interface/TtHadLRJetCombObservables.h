#ifndef TtHadLRJetCombObservables_h
#define TtHadLRJetCombObservables_h

// $Id: TtHadLRJetCombObservables.h,v 1.1 2007/10/07 15:33:37 mfhansen Exp $
// copied TtSemiLRJetCombObservables.h,v 1.4 2007/06/15 08:53:52 by heyninck 
/**
  \class    TtHadLRJetCombObservables is based on TtSemiLRJetCombObservables.h 
  \brief    Steering class for the overall hadronic top likelihood

   In this TtHadLRJetCombObservables class a list of observables is calculated that might be used in the evaluation of the combined Likelihood ratio to distinguish between correct and wrong jet combinations
  // obs1 : pt(hadronic tops)
  // obs2 : (pt_b1 + pt_b2)/(sum jetpt)
  // obs3 : delta R between had b and had W_plus  
  // obs4 : delta R between had bbar and had W_minus
  // obs5 : delta R between light quark-jets from W_plus
  // obs6 : delta R between light quark-jets from W_minus 
  // obs7 : b-tagging information
  // obs8 : chi2 value of kinematical fit with W-mass constraint
  \author   Jan Heyninck
  \version  $Id: TtHadLRJetCombObservables.h,v 1.1 2007/10/07 15:33:37 mfhansen Exp $
*/

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"
#include <Math/VectorUtil.h>


class TtHadLRJetCombObservables {

 public:
  
  TtHadLRJetCombObservables();
  ~TtHadLRJetCombObservables();	
  
  void  operator()(TtHadEvtSolution&);
  
 private:
  std::vector<std::pair<unsigned int,double> > jetCombVarVal;  
};

#endif
