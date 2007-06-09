//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TtSemiSimpleBestJetComb.h,v 1.1 2007/05/19 09:54:38 heyninck Exp $
//

#ifndef TtSemiSimpleBestJetComb_h
#define TtSemiSimpleBestJetComb_h

/**
  \class    TtSemiSimpleBestJetComb TtSemiSimpleBestJetComb.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtSemiSimpleBestJetComb.h"
  \brief    Steering class for the overall top-lepton likelihood

   TtSemiSimpleBestJetComb allows to calculate and retrieve the overall top-lepton
   likelihood as defined in CMS Note 2006/024

  \author   Jan Heyninck
  \version  $Id: TtSemiSimpleBestJetComb.h,v 1.1 2007/05/19 09:54:38 heyninck Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"
#include <Math/VectorUtil.h>


class TtSemiSimpleBestJetComb {

  public:
    TtSemiSimpleBestJetComb();
    ~TtSemiSimpleBestJetComb();	

    int  operator()(std::vector<TtSemiEvtSolution> &);

  private:

};

#endif
