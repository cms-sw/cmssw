//
// $Id: TtHadSimpleBestJetComb.h,v 1.0 2007/09/20 13:13:13  mfhansen Exp $
// adapted from TtSemiSimpleBestJetComb.h,v 1.2 2007/06/09 01:17:41 lowette Exp 
// for fully hadronic channel

#ifndef TtHadSimpleBestJetComb_h
#define TtHadSimpleBestJetComb_h


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"
#include <Math/VectorUtil.h>


class TtHadSimpleBestJetComb {

  public:
    TtHadSimpleBestJetComb();
    ~TtHadSimpleBestJetComb();	

    int  operator()(std::vector<TtHadEvtSolution> &);

  private:

};

#endif
