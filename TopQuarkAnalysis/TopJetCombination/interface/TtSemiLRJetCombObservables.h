//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiLRJetCombObservables.h,v 1.2 2007/05/22 16:43:36 heyninck Exp $
//

#ifndef TtSemiLRJetCombObservables_h
#define TtSemiLRJetCombObservables_h

/**
  \class    TtSemiLRJetCombObservables TtSemiLRJetCombObservables.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtSemiLRJetCombObservables.h"
  \brief    Steering class for the overall top-lepton likelihood

   In this TtSemiLRJetCombObservables class a list of observables is calculated that might be used in the evaluation of the
   combined Likelihood ratio to distinguish between correct and wrong jet combinations
  // obs1 : pt(had top)
  // obs2 : (pt_b1 + pt_b2)/(sum jetpt)
  // obs3 : delta R between had top and lep b  

  \author   Jan Heyninck
  \version  $Id: TtSemiLRJetCombObservables.h,v 1.2 2007/05/22 16:43:36 heyninck Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"


using namespace std;


class TtSemiLRJetCombObservables {

  public:
    TtSemiLRJetCombObservables();
    ~TtSemiLRJetCombObservables();	

    void  operator()(TtSemiEvtSolution&);

  private:
    vector<pair<double,double> > jetCombVarVal;


};

#endif
