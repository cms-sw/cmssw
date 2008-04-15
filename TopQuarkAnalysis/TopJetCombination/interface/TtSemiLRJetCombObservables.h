//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombObservables.h,v 1.4.8.1 2008/03/26 08:01:09 jmmaes Exp $
//

#ifndef TtSemiLRJetCombObservables_h
#define TtSemiLRJetCombObservables_h

/**
  \class    TtSemiLRJetCombObservables TtSemiLRJetCombObservables.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtSemiLRJetCombObservables.h"
  \brief    Steering class for the overall top-lepton likelihood

   In this TtSemiLRJetCombObservables class a list of observables is calculated that might be used in the evaluation of the
   combined Likelihood ratio to distinguish between correct and wrong jet combinations
  // obs1 : 
  // obs2 : 
  // obs3 : 
  // ...
  
  \author   Jan Heyninck
  \version  $Id: TtSemiLRJetCombObservables.h,v 1.4.8.1 2008/03/26 08:01:09 jmmaes Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// General C++ stuff
#include <iostream>
#include <string>
#include <vector>
#include <Math/VectorUtil.h>

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

using namespace std;

class TtSemiLRJetCombObservables {

  public:

  typedef pair<unsigned int,bool>   IntBoolPair;
  
  TtSemiLRJetCombObservables();
  ~TtSemiLRJetCombObservables();	
   
  vector< IntBoolPair > operator()(TtSemiEvtSolution&, const edm::Event & iEvent,bool matchOnly = false);
  //void  operator()(TtSemiEvtSolution&);
  void jetSource(const edm::InputTag & jetSource) {jetSource_ = jetSource;}
 
private:

  typedef pair<unsigned int,double> IntDblPair;
  //std::vector<std::pair<unsigned int,double> > jetCombVarVal;
  
  edm::InputTag jetSource_;
  
  vector< IntDblPair > evtselectVarVal;
  vector< IntBoolPair > evtselectVarMatch;
};

#endif
