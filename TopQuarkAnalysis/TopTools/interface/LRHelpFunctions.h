
//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: LRHelpFunctions.h,v 1.2 2007/06/01 09:11:46 heyninck Exp $
//

#ifndef LRHelpFunctions_h
#define LRHelpFunctions_h

/**
  \class    LRHelpFunctions LRHelpFunctions.h "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"
  \brief    Help functionalities to implement and evaluate LR ratio method

  \author   Jan Heyninck
  \version  $Id: LRHelpFunctions.h,v 1.2 2007/06/01 09:11:46 heyninck Exp $
*/

#include "TString.h"
#include "TFile.h"
#include "TKey.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TGraph.h"
#include "TList.h"
#include "TPaveText.h"
#include "TText.h"
#include "TCanvas.h"
#include <iostream>
using namespace std;


class LRHelpFunctions {

  public:
    LRHelpFunctions();
    LRHelpFunctions(vector<int>, int, vector<double>, vector<double>,vector<const char*>, int, double, double, const char*);
    ~LRHelpFunctions();	

    void 	fillToSignalHists(vector<double>);
    void 	fillToBackgroundHists(vector<double>);
    void 	normalizeSandBhists();
    void        makeAndFitSoverSplusBHists();
    void        readObsHistsAndFits(TString,bool);
    void        storeToROOTfile(TString);
    void        storeControlPlots(TString); 
    void        fillLRSignalHist(double);
    void        fillLRBackgroundHist(double);
    void        makeAndFitPurityHists(); 
    double 	calcLRval(vector<double>);
    bool 	isIncluded(int);
    double 	calcProb(double);
   
  private:
    vector<TH1F*> hObsS, hObsB, hObsSoverSplusB;
    vector<TH2F*> hObsCorr;
    vector<TF1*>  fObsSoverSplusB;
    TH1F 	  *hLRtotS, *hLRtotB, *hLRtotSoverSplusB;
    TF1		  *fLRtotSoverSplusB;
    vector<int>   selObs;
};

#endif
