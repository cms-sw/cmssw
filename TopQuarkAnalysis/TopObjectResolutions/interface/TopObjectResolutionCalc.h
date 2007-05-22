//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.h,v 1.1 2007/05/08 14:03:05 heyninck Exp $
//

#ifndef TopObjectResolutionCalc_h
#define TopObjectResolutionCalc_h

/**
  \class    TopObjectResolutionCalc TopObjectResolutionCalc.h "TopQuarkAnalysis/TopLeptonSelection/interface/TopObjectResolutionCalc.h"
  \brief    Steering class for the overall top-lepton likelihood

   TopObjectResolutionCalc allows to calculate and retrieve the overall top-lepton
   likelihood as defined in CMS Note 2006/024

  \author   Jan Heyninck
  \version  $Id: TopObjectResolutionCalc.h,v 1.1 2007/05/08 14:03:05 heyninck Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"


using namespace std;


class TopObjectResolutionCalc {

  public:
    TopObjectResolutionCalc();
    TopObjectResolutionCalc(TString);
    ~TopObjectResolutionCalc();	

    double getObsRes(int, double);
    double getObsRes(int, double, double);
    void  operator()(TopJet&);
    void  operator()(TopMET&);
    void  operator()(TopMuon&);
    void  operator()(TopElectron&);

  private:
    TFile * resoFile;
    TF1 fResPar[10][2][3];
    TF1 fResVsET[10][2];


};

#endif
