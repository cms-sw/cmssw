
//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: LRHelpFunctions.h,v 1.7 2007/07/05 14:18:52 heyninck Exp $
//

#ifndef LRHelpFunctions_h
#define LRHelpFunctions_h

/**
  \class    LRHelpFunctions LRHelpFunctions.h "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"
  \brief    Help functionalities to implement and evaluate LR ratio method

  \author   Jan Heyninck
  \version  $Id: LRHelpFunctions.h,v 1.7 2007/07/05 14:18:52 heyninck Exp $
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


class LRHelpFunctions {

  public:
    LRHelpFunctions();
    LRHelpFunctions(std::vector<int>, int, std::vector<double>, std::vector<double>,std::vector<const char*>);
    LRHelpFunctions(int, double, double, const char*);
    ~LRHelpFunctions();	
    void 	initLRHistsAndFits(int, double, double, const char*);
    void 	setObsFitParameters(int obs,std::vector<double>);
    void 	fillToSignalHists(std::vector<double>);
    void 	fillToBackgroundHists(std::vector<double>);
    void 	normalizeSandBhists();
    void        makeAndFitSoverSplusBHists();
    void        readObsHistsAndFits(TString,std::vector<int>,bool);
    void        storeToROOTfile(TString);
    void        storeControlPlots(TString); 
    void        fillLRSignalHist(double);
    void        fillLRBackgroundHist(double);
    void        makeAndFitPurityHists(); 
    double 	calcLRval(std::vector<double>);
    double 	calcProb(double);
    bool 	obsFitIncluded(int);
   
  private:
    std::vector<TH1F*> hObsS, hObsB, hObsSoverSplusB;
    std::vector<TH2F*> hObsCorr;
    std::vector<TF1*>  fObsSoverSplusB;
    TH1F 	       *hLRtotS, *hLRtotB, *hLRtotSoverSplusB;
    TF1		       *fLRtotSoverSplusB;
    TGraph             *hEffvsPur, *hLRValvsPur, *hLRValvsEff;
    bool 	       constructPurity;
};

#endif
