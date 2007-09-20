//
// $Id: TtSemiSimpleBestJetComb.h,v 1.2 2007/06/09 01:17:41 lowette Exp $
//

#ifndef TtSemiSimpleBestJetComb_h
#define TtSemiSimpleBestJetComb_h

/**
  \class    TtSemiSimpleBestJetComb TtSemiSimpleBestJetComb.h "TopQuarkAnalysis/TopLeptonSelection/interface/TtSemiSimpleBestJetComb.h"
  \brief    Simple method to get the correct jet combination in semileptonic ttbar events

   This method starts from a vector of fitted TtSemiEvtSolutions. This class returns the solution with the highest probChi^2 value. In case
   that there are more possibilities (eg when only a hadrW constraint was applied), the correct hadronic b is assumed to be the one with the
   smallest DR angle wrt the Whadr direction. 

  \author   Jan Heyninck
  \version  $Id: TtSemiSimpleBestJetComb.h,v 1.2 2007/06/09 01:17:41 lowette Exp $
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
