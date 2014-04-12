// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "TStyle.h"

void setTDRStyle() {

  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

// For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600); //Height of canvas
  tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
  tdrStyle->SetCanvasDefY(0);

// For the Pad:
  tdrStyle->SetPadBorderMode(0);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

// For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

// For the histo:
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(1);
  tdrStyle->SetEndErrorSize(2);
  tdrStyle->SetErrorX(0.);
  tdrStyle->SetMarkerStyle(20);

//For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

//For the date:
  tdrStyle->SetOptDate(0);

// For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");
  tdrStyle->SetStatColor(kWhite);
  tdrStyle->SetStatFont(42);
  tdrStyle->SetStatFontSize(0.025);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.1);
  tdrStyle->SetStatW(0.15);

// Margins:
  tdrStyle->SetPadTopMargin(0.05);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.16);
  tdrStyle->SetPadRightMargin(0.02);

// For the Global title:
  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);

// For the axis titles:
  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.25);

// For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.05, "XYZ");

// For the axis:
  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

// Change for log plots:
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
  tdrStyle->SetOptLogz(0);

// Postscript options:
  tdrStyle->SetPaperSize(20.,20.);

  tdrStyle->cd();
}

// data dirs
TString theDirName = "Figures";

// data files
// All the rootfiles must be present:
//  TkStrct PixBar PixFwdPlus PixFwdMinus TIB TIDF TIDB TOB TEC BeamPipe InnerServices
//

// histograms
TProfile* prof_x0_BeamPipe;
TProfile* prof_x0_PixBar;
TProfile* prof_x0_PixFwdPlus;
TProfile* prof_x0_PixFwdMinus;
TProfile* prof_x0_TIB;
TProfile* prof_x0_TIDF;
TProfile* prof_x0_TIDB;
TProfile* prof_x0_InnerServices;
TProfile* prof_x0_TOB;
TProfile* prof_x0_TEC;
TProfile* prof_x0_Outside;
//
TProfile* prof_x0_SEN;
TProfile* prof_x0_SUP;
TProfile* prof_x0_ELE;
TProfile* prof_x0_CAB;
TProfile* prof_x0_COL;
TProfile* prof_x0_OTH;
TProfile* prof_x0_AIR;
//
TH1D* hist_x0_BeamPipe;
TH1D* hist_x0_Pixel;
TH1D* hist_x0_IB;
TH1D* hist_x0_TOB;
TH1D* hist_x0_TEC;
TH1D* hist_x0_Outside;
//
TH1D* hist_x0_SEN;
TH1D* hist_x0_SUP;
TH1D* hist_x0_ELE;
TH1D* hist_x0_CAB;
TH1D* hist_x0_COL;
TH1D* hist_x0_OTH;
//
float xmin;
float xmax; 

float ymin;
float ymax;
//

using namespace std;

// Main
MaterialBudget_TDR() {

  //TDR style
  setTDRStyle();  

  // plots
  createPlots("x_vs_eta");
  createPlots("x_vs_phi");
  createPlots("x_vs_R");
  createPlots("l_vs_eta");
  createPlots("l_vs_phi");
  createPlots("l_vs_R");
}

void createPlots(TString plot){
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  if(plot.CompareTo("x_vs_eta") == 0) {
    plotNumber = 10;
    abscissaName = TString("#eta");
    ordinateName = TString("t/X_{0}");
    ymin =  0.0;
    ymax =  2.575;
    xmin = -4.0;
    xmax =  4.0;
  }
  else if(plot.CompareTo("x_vs_phi") == 0) {
    plotNumber = 20;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("t/X_{0}");
    ymin =  0.0;
    ymax =  6.2;
    xmin = -4.0;
    xmax =  4.0;
  }
  else if(plot.CompareTo("x_vs_R") == 0) {
    plotNumber = 40;
    abscissaName = TString("R [cm]");
    ordinateName = TString("t/X_{0}");
    ymin =  0.0;
    ymax =  70.0;
    xmin =  0.0;
    xmax =  1200.0;
  }

  else if(plot.CompareTo("l_vs_eta") == 0) {
    plotNumber = 1010;
    abscissaName = TString("#eta");
    ordinateName = TString("t/#lambda_{I}");
    ymin =  0.0;
    ymax =  0.73;
    xmin = -4.0;
    xmax =  4.0;
  }
  else if(plot.CompareTo("l_vs_phi") == 0) {
    plotNumber = 1020;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("t/#lambda_{I}");
    ymin =  0.0;
    ymax =  1.2;
    xmin = -4.0;
    xmax =  4.0;
  }
  else if(plot.CompareTo("l_vs_R") == 0) {
    plotNumber = 1040;
    abscissaName = TString("R [cm]");
    ordinateName = TString("t/#lambda_{I}");
    ymin =  0.0;
    ymax =  7.5;
    xmin =  0.0;
    xmax =  1200.0;
  }
  else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }
  
  TString subDetector("empty");
  for(unsigned int i_detector=0; i_detector<=10; i_detector++) {
    switch(i_detector) {
    case 0: {
      subDetector = "TIB";
      break;
    }
    case 1: {
      subDetector = "TIDF";
      break;
    }
    case 2: {
      subDetector = "TIDB";
      break;
    }
    case 3: {
      subDetector = "InnerServices";
      break;
    }
    case 4: {
      subDetector = "TOB";
      break;
    }
    case 5: {
      subDetector = "TEC";
      break;
    }
    case 6: {
      subDetector = "TkStrct";
      break;
    }
    case 7: {
      subDetector = "PixBar";
      break;
    }
    case 8: {
      subDetector = "PixFwdPlus";
      break;
    }
    case 9: {
      subDetector = "PixFwdMinus";
      break;
    }
    case 10: {
      subDetector = "BeamPipe";
      break;
    }
    default: cout << " something wrong" << endl;
    }

    // file name
    TString subDetectorFileName = "matbdg_" + subDetector + ".root";

    // open file
    TFile* subDetectorFile = new TFile(subDetectorFileName);
    cout << "*** Open file... " << endl;
    cout << subDetectorFileName << endl;
    cout << "***" << endl;
    
    switch(i_detector) {
    case 0: {
      // subDetector = "TIB";
      // subdetector profiles
      prof_x0_TIB = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_IB  = (TH1D*)prof_x0_TIB->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP = (TH1D*)prof_x0_SUP->ProjectionX();
      hist_x0_SEN = (TH1D*)prof_x0_SEN->ProjectionX();
      hist_x0_CAB = (TH1D*)prof_x0_CAB->ProjectionX();
      hist_x0_COL = (TH1D*)prof_x0_COL->ProjectionX();
      hist_x0_ELE = (TH1D*)prof_x0_ELE->ProjectionX();
      hist_x0_OTH = (TH1D*)prof_x0_OTH->ProjectionX();
      hist_x0_OTH = (TH1D*)prof_x0_AIR->ProjectionX();
      break;
    }
    case 1: {
      // subDetector = "TIDF";
      // subdetector profiles
      prof_x0_TIDF = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_IB->Add( (TH1D*)prof_x0_TIDF->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 2: {
      // subDetector = "TIDB";
      // subdetector profiles
      prof_x0_TIDB = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_IB->Add( (TH1D*)prof_x0_TIDB->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 3: {
      // subDetector = "InnerServices";
      // subdetector profiles
      prof_x0_InnerServices = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_IB->Add( (TH1D*)prof_x0_InnerServices->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 4: {
      // subDetector = "TOB";
      // subdetector profiles
      prof_x0_TOB = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_TOB = (TH1D*)prof_x0_TOB->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 5: {
      // subDetector = "TEC";
      // subdetector profiles
      prof_x0_TEC = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_TEC =  (TH1D*)prof_x0_TEC->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 6: {
      // subDetector = "TkStrct";
      // subdetector profiles
      prof_x0_Outside = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_Outside = (TH1D*)prof_x0_Outside->ProjectionX();
      break;
    }
    case 7: {
      // subDetector = "PixBar";
      // subdetector profiles
      prof_x0_PixBar = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_Pixel  = (TH1D*)prof_x0_PixBar->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 8: {
      // subDetector = "PixFwdPlus";
      // subdetector profiles
      prof_x0_PixFwdPlus = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_Pixel->Add( (TH1D*)prof_x0_PixFwdPlus->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 9: {
      subDetector = "PixFwdMinus";
      // subdetector profiles
      prof_x0_PixFwdMinus = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_Pixel->Add( (TH1D*)prof_x0_PixFwdMinus->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_SUP->Add(   (TH1D*)prof_x0_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_AIR->ProjectionX("B")  , +1.000 );
      break;
    }
    case 10: {
      // subDetector = "BeamPipe";
      // subdetector profiles
      prof_x0_BeamPipe = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      hist_x0_BeamPipe = (TH1D*)prof_x0_BeamPipe->ProjectionX();
      break;
    }
    default: cout << " something wrong" << endl;
    }
  }
  
  // colors
  int kpipe  = kGray+2;
  int kpixel = kAzure-5;
  int ktib   = kMagenta-2;
  int ktob   = kOrange+10;
  int ktec   = kOrange-2;
  int kout   = kGray;

  int ksen = 27;
  int kele = 46;
  int kcab = kOrange-8;
  int kcol = 30;
  int ksup = 38;
  int koth = kOrange-2;

  hist_x0_BeamPipe->SetFillColor(kpipe); // Beam Pipe	 = dark gray
  hist_x0_Pixel->SetFillColor(kpixel);   // Pixel 	 = dark blue
  hist_x0_IB->SetFillColor(ktib);	 // TIB and TID  = violet
  hist_x0_TOB->SetFillColor(ktob);       // TOB          = red
  hist_x0_TEC->SetFillColor(ktec);       // TEC          = yellow gold
  hist_x0_Outside->SetFillColor(kout);   // Support tube = light gray

  hist_x0_SEN->SetFillColor(ksen); // Sensitive   = brown
  hist_x0_ELE->SetFillColor(kele); // Electronics = red
  hist_x0_CAB->SetFillColor(kcab); // Cabling     = dark orange 
  hist_x0_COL->SetFillColor(kcol); // Cooling     = green
  hist_x0_SUP->SetFillColor(ksup); // Support     = light blue
  hist_x0_OTH->SetFillColor(koth); // Other+Air   = light orange
  //
  
  
  // First Plot: BeamPipe + Pixel + TIB/TID + TOB + TEC + Outside
  // stack
  TString stackTitle_SubDetectors = Form( "Tracker Material Budget;%s;%s",abscissaName.Data(),ordinateName.Data() );
  THStack stack_x0_SubDetectors("stack_x0",stackTitle_SubDetectors);
  stack_x0_SubDetectors.Add(hist_x0_BeamPipe);
  stack_x0_SubDetectors.Add(hist_x0_Pixel);
  stack_x0_SubDetectors.Add(hist_x0_IB);
  stack_x0_SubDetectors.Add(hist_x0_TOB);
  stack_x0_SubDetectors.Add(hist_x0_TEC);
  stack_x0_SubDetectors.Add(hist_x0_Outside);
  //
  
  // canvas
  TCanvas can_SubDetectors("can_SubDetectors","can_SubDetectors",800,800);
  can_SubDetectors.Range(0,0,25,25);
  can_SubDetectors.SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  //
  
  // Draw
  stack_x0_SubDetectors.SetMinimum(ymin);
  stack_x0_SubDetectors.SetMaximum(ymax);
  stack_x0_SubDetectors.Draw("HIST");
  stack_x0_SubDetectors.GetXaxis()->SetLimits(xmin,xmax);
  //
  
  // Legenda
  TLegend* theLegend_SubDetectors = new TLegend(0.180,0.8,0.98,0.92); 
  theLegend_SubDetectors->SetNColumns(3); 
  theLegend_SubDetectors->SetFillColor(0); 
  theLegend_SubDetectors->SetFillStyle(0); 
  theLegend_SubDetectors->SetBorderSize(0); 

  theLegend_SubDetectors->AddEntry(hist_x0_Outside,   "Support Tube",  "f");
  theLegend_SubDetectors->AddEntry(hist_x0_TOB,       "TOB",           "f");
  theLegend_SubDetectors->AddEntry(hist_x0_Pixel,     "Pixel",         "f");

  theLegend_SubDetectors->AddEntry(hist_x0_TEC,       "TEC",           "f");
  theLegend_SubDetectors->AddEntry(hist_x0_IB,        "TIB and TID",   "f");
  theLegend_SubDetectors->AddEntry(hist_x0_BeamPipe,  "Beam Pipe",     "f");
  theLegend_SubDetectors->Draw();
  //

  // text
  TPaveText* text_SubDetectors = new TPaveText(0.180,0.727,0.402,0.787,"NDC");
  text_SubDetectors->SetFillColor(0);
  text_SubDetectors->SetBorderSize(0);
  text_SubDetectors->AddText("CMS Simulation");
  text_SubDetectors->SetTextAlign(11);
  text_SubDetectors->Draw();
  //
  
  // Store
  can_SubDetectors.Update();
  //  can_SubDetectors.SaveAs( Form( "%s/Tracker_SubDetectors_%s.eps",  theDirName.Data(), plot.Data() ) );
  //  can_SubDetectors.SaveAs( Form( "%s/Tracker_SubDetectors_%s.gif",  theDirName.Data(), plot.Data() ) );
  can_SubDetectors.SaveAs( Form( "%s/Tracker_SubDetectors_%s.pdf",  theDirName.Data(), plot.Data() ) );
  //  can_SubDetectors.SaveAs( Form( "%s/Tracker_SubDetectors_%s.png",  theDirName.Data(), plot.Data() ) );
  can_SubDetectors.SaveAs( Form( "%s/Tracker_SubDetectors_%s.root",  theDirName.Data(), plot.Data() ) );
  //  can_SubDetectors.SaveAs( Form( "%s/Tracker_SubDetectors_%s.C",  theDirName.Data(), plot.Data() ) );
  //
  
  
  // Second Plot: BeamPipe + SEN + ELE + CAB + COL + SUP + OTH/AIR + Outside
  // stack
  TString stackTitle_Materials = Form( "Tracker Material Budget;%s;%s",abscissaName.Data(),ordinateName.Data() );
  THStack stack_x0_Materials("stack_x0",stackTitle_Materials);
  stack_x0_Materials.Add(hist_x0_BeamPipe);
  stack_x0_Materials.Add(hist_x0_SEN);
  stack_x0_Materials.Add(hist_x0_ELE);
  stack_x0_Materials.Add(hist_x0_CAB);
  stack_x0_Materials.Add(hist_x0_COL);
  stack_x0_Materials.Add(hist_x0_SUP);
  stack_x0_Materials.Add(hist_x0_OTH);
  stack_x0_Materials.Add(hist_x0_Outside);
  //
  
  // canvas
  TCanvas can_Materials("can_Materials","can_Materials",800,800);
  can_Materials.Range(0,0,25,25);
  can_Materials.SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  //
  
  // Draw
  stack_x0_Materials.SetMinimum(ymin);
  stack_x0_Materials.SetMaximum(ymax);
  stack_x0_Materials.Draw("HIST");
  stack_x0_Materials.GetXaxis()->SetLimits(xmin,xmax);
  //
  
  // Legenda
  TLegend* theLegend_Materials = new TLegend(0.180,0.8,0.95,0.92); 
  theLegend_Materials->SetNColumns(3); 
  theLegend_Materials->SetFillColor(0); 
  theLegend_Materials->SetBorderSize(0); 

  theLegend_Materials->AddEntry(hist_x0_Outside,   "Support and Thermal Screen",  "f");
  theLegend_Materials->AddEntry(hist_x0_OTH,       "Other",                       "f");
  theLegend_Materials->AddEntry(hist_x0_SUP,       "Mechanical Structures",       "f");

  theLegend_Materials->AddEntry(hist_x0_COL,       "Cooling",                     "f");
  theLegend_Materials->AddEntry(hist_x0_CAB,       "Cables",                      "f");
  theLegend_Materials->AddEntry(hist_x0_ELE,       "Electronics",                 "f");

  theLegend_Materials->AddEntry(hist_x0_SEN,       "Sensitive",                   "f");
  theLegend_Materials->AddEntry(hist_x0_BeamPipe,  "Beam Pipe",                   "f");
  theLegend_Materials->Draw();
  //

  // text
  TPaveText* text_Materials = new TPaveText(0.180,0.727,0.402,0.787,"NDC");
  text_Materials->SetFillColor(0);
  text_Materials->SetBorderSize(0);
  text_Materials->AddText("CMS Simulation");
  text_Materials->SetTextAlign(11);
  text_Materials->Draw();
  //
  
  // Store
  can_Materials.Update();
  // can_Materials.SaveAs( Form( "%s/Tracker_Materials_%s.eps",  theDirName.Data(), plot.Data() ) );
  // can_Materials.SaveAs( Form( "%s/Tracker_Materials_%s.gif",  theDirName.Data(), plot.Data() ) );
  can_Materials.SaveAs( Form( "%s/Tracker_Materials_%s.pdf",  theDirName.Data(), plot.Data() ) );
  // can_Materials.SaveAs( Form( "%s/Tracker_Materials_%s.png",  theDirName.Data(), plot.Data() ) );
  can_Materials.SaveAs( Form( "%s/Tracker_Materials_%s.root",  theDirName.Data(), plot.Data() ) );
  // can_Materials.SaveAs( Form( "%s/Tracker_Materials_%s.C",  theDirName.Data(), plot.Data() ) );
  //
  
}
