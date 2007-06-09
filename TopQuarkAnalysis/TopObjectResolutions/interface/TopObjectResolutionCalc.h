//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.h,v 1.3 2007/06/08 19:43:38 lowette Exp $
//

#ifndef TopObjectResolutionCalc_h
#define TopObjectResolutionCalc_h

/**
  \class    TopObjectResolutionCalc TopObjectResolutionCalc.h "TopQuarkAnalysis/TopLeptonSelection/interface/TopObjectResolutionCalc.h"
  \author   Jan Heyninck
  \version  $Id: TopObjectResolutionCalc.h,v 1.3 2007/06/08 19:43:38 lowette Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/General/interface/envUtil.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"


class TopObjectResolutionCalc {

  public:
    TopObjectResolutionCalc();
    TopObjectResolutionCalc(TString);
    ~TopObjectResolutionCalc();	

    double getObsRes(int, int, double);
    void  operator()(TopJet&);
    void  operator()(TopMET&);
    void  operator()(TopMuon&);
    void  operator()(TopElectron&);

  private:
    TFile * resoFile;
    TF1 fResVsET[10][10];


};

#endif
