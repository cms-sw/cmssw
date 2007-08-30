//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.h,v 1.5.2.2 2007/08/30 17:08:24 heyninck Exp $
//

#ifndef TopObjectResolutionCalc_h
#define TopObjectResolutionCalc_h

/**
  \class    TopObjectResolutionCalc TopObjectResolutionCalc.h "TopQuarkAnalysis/TopLeptonSelection/interface/TopObjectResolutionCalc.h"
  \author   Jan Heyninck
  \version  $Id: TopObjectResolutionCalc.h,v 1.5.2.2 2007/08/30 17:08:24 heyninck Exp $
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
#include "TMultiLayerPerceptron.h"
using namespace std;

class TopObjectResolutionCalc {

  public:
    TopObjectResolutionCalc();
    TopObjectResolutionCalc(TString,bool);
    ~TopObjectResolutionCalc();	

    double getObsRes(int, int, double);
    int    getEtaBin(double);
    void   operator()(TopJet&);
    void   operator()(TopMET&);
    void   operator()(TopTau&);
    void   operator()(TopMuon&);
    void   operator()(TopElectron&);

  private:
    TFile * resoFile;
    vector<double> etabinVals;
    TF1 fResVsET[10][10];
    TMultiLayerPerceptron* network[10];
    bool useNN;

};

#endif
