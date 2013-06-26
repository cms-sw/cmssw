
//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: LRHelpFunctions.h,v 1.11 2013/05/30 20:47:31 gartung Exp $
//

#ifndef LRHelpFunctions_h
#define LRHelpFunctions_h

/**
  \class    LRHelpFunctions LRHelpFunctions.h "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"
  \brief    Help functionalities to implement and evaluate LR ratio method

  \author   Jan Heyninck
  \version  $Id: LRHelpFunctions.h,v 1.11 2013/05/30 20:47:31 gartung Exp $
*/

#include <cmath>
#include <iostream>

#include "TROOT.h"
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


class LRHelpFunctions {

  public:
    LRHelpFunctions();
    LRHelpFunctions(const std::vector<int>&, int, const std::vector<double>&, const std::vector<double>&,const std::vector<const char*>&);
    LRHelpFunctions(int, double, double, const char*);
    ~LRHelpFunctions();	
    void	recreateFitFct(const std::vector<int>& obsNr, const std::vector<const char*>& functions);
    void 	initLRHistsAndFits(int, double, double, const char*);
    void 	setObsFitParameters(int obs,const std::vector<double>&);
    void 	fillToSignalHists(const std::vector<double>&, double weight = 1.0);
    void 	fillToBackgroundHists(const std::vector<double>&, double weight = 1.0);
    void 	fillToSignalHists(int, double, double weight = 1.0);
    void 	fillToBackgroundHists(int, double, double weight = 1.0);
    void 	fillToSignalCorrelation(int obsNbr1, double obsVal1, int obsNbr2,
	double obsVal2, double weight);
    void 	normalizeSandBhists();
    void        makeAndFitSoverSplusBHists();
    void        readObsHistsAndFits(const TString&,const std::vector<int>&,bool);
    void        storeToROOTfile(const TString&);
    void        storeControlPlots(const TString&); 
    void        fillLRSignalHist(double, double weight = 1.0);
    void        fillLRBackgroundHist(double, double weight = 1.0);
    void        makeAndFitPurityHists(); 
    double 	calcLRval(const std::vector<double>&);
    double 	calcPtdrLRval(const std::vector<double>& vals, bool useCorrelation = false);
    double 	calcProb(double);
    bool 	obsFitIncluded(int);
    void setXlabels(const std::vector<std::string> & xLabels);
    void setYlabels(const std::vector<std::string> & yLabels);
    void singlePlot(const TString& fname, int obsNbr, const TString& extension= TString("eps"));
    void purityPlot(const TString& fname, const TString& extension= TString("eps"));


  private:
    std::vector<TH1F*> hObsS, hObsB, hObsSoverSplusB;
    std::vector<TH2F*> hObsCorr;
    std::vector<TF1*>  fObsSoverSplusB;
    TH1F 	       *hLRtotS, *hLRtotB, *hLRtotSoverSplusB;
    TF1		       *fLRtotSoverSplusB;
    TGraph             *hEffvsPur, *hLRValvsPur, *hLRValvsEff;
    bool 	       constructPurity;
    std::vector<int> obsNumbers;
};

#endif
