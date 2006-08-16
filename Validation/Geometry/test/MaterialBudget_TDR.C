// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// data dirs
TString theDirName = "Images";
//

// data files
// All the rootfiles must be present:
//  TkStrct PixBar PixFwdPlus PixFwdMinus TIB TIDF TIDB TOB TEC BeamPipe
//

// histograms
TProfile* prof_x0_BeamPipe;
TProfile* prof_x0_PixBar;
TProfile* prof_x0_PixFwdPlus;
TProfile* prof_x0_PixFwdMinus;
TProfile* prof_x0_TIB;
TProfile* prof_x0_TIDF;
TProfile* prof_x0_TIDB;
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

using namespace std;

// Main
MaterialBudget_TDR() {
  TString subDetector("empty");
  for(unsigned int i_detector=0; i_detector<=9; i_detector++) {
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
      subDetector = "TOB";
      break;
    }
    case 4: {
      subDetector = "TEC";
      break;
    }
    case 5: {
      subDetector = "TkStrct";
      break;
    }
    case 6: {
      subDetector = "PixBar";
      break;
    }
    case 7: {
      subDetector = "PixFwdPlus";
      break;
    }
    case 8: {
      subDetector = "PixFwdMinus";
      break;
    }
    case 9: {
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
      prof_x0_TIB = (TProfile*)subDetectorFile->Get("10");
      hist_x0_IB  = (TH1D*)prof_x0_TIB->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      prof_x0_TIDF = (TProfile*)subDetectorFile->Get("10");
      hist_x0_IB->Add( (TH1D*)prof_x0_TIDF->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      prof_x0_IB = (TProfile*)subDetectorFile->Get("10");
      hist_x0_IB->Add( (TH1D*)prof_x0_IB->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      // subDetector = "TOB";
      // subdetector profiles
      prof_x0_TOB = (TProfile*)subDetectorFile->Get("10");
      hist_x0_TOB = (TH1D*)prof_x0_TOB->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      // subDetector = "TEC";
      // subdetector profiles
      prof_x0_TEC = (TProfile*)subDetectorFile->Get("10");
      hist_x0_TEC =  (TH1D*)prof_x0_TEC->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      // subDetector = "TkStrct";
      // subdetector profiles
      prof_x0_Outside = (TProfile*)subDetectorFile->Get("10");
      hist_x0_Outside = (TH1D*)prof_x0_Outside->ProjectionX();
      break;
    }
    case 6: {
      // subDetector = "PixBar";
      // subdetector profiles
      prof_x0_PixBar = (TProfile*)subDetectorFile->Get("10");
      hist_x0_Pixel  = (TH1D*)prof_x0_PixBar->ProjectionX();
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
    case 7: {
      // subDetector = "PixFwdPlus";
      // subdetector profiles
      prof_x0_PixFwdPlus = (TProfile*)subDetectorFile->Get("10");
      hist_x0_Pixel->Add( (TH1D*)prof_x0_PixFwdPlus->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      subDetector = "PixFwdMinus";
      // subdetector profiles
      prof_x0_PixFwdMinus = (TProfile*)subDetectorFile->Get("10");
      hist_x0_Pixel->Add( (TH1D*)prof_x0_PixFwdMinus->ProjectionX("B") , +1.000 );
      // category profiles
      prof_x0_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_AIR   = (TProfile*)subDetectorFile->Get("710");
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
      // subDetector = "BeamPipe";
      // subdetector profiles
      prof_x0_BeamPipe = (TProfile*)subDetectorFile->Get("10");
      hist_x0_BeamPipe = (TH1D*)prof_x0_BeamPipe->ProjectionX();
      break;
    }
    default: cout << " something wrong" << endl;
    }
  }
  
  // properties
  hist_x0_BeamPipe->SetFillColor(136); // Beam Pipe = dark gray blue
  hist_x0_Pixel->SetFillColor(38);     // Pixel     = light blue
  hist_x0_IB->SetFillColor(196);       // TIB+TID   = light red
  hist_x0_TOB->SetFillColor(130);      // TOB       = dark green
  hist_x0_TEC->SetFillColor(41);       // TEC       = yellow gold
  hist_x0_Outside->SetFillColor(128);  // Outside   = dark brown
  //
  hist_x0_SUP->SetFillColor(13); // Support     = dark gray
  hist_x0_SEN->SetFillColor(27); // Sensitive   = brown
  hist_x0_CAB->SetFillColor(46); // Cabling     = red
  hist_x0_COL->SetFillColor(38); // Cooling     = light blue
  hist_x0_ELE->SetFillColor(30); // Electronics = green
  hist_x0_OTH->SetFillColor(42); // Other+Air   = orange
  //
  float mbmin  =  0.0;
  float mbmax  =  1.8;
  float etamin = -3.5;
  float etamax =  3.5;
  //  
  
  // First Plot: BeamPipe + Pixel + TIB/TID + TOB + TEC + Outside
  // stack
  TString stackTitle_SubDetectors = "Tracker Material Budget;#eta;X/X_{0}";
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
  stack_x0_SubDetectors.SetMinimum(mbmin);
  stack_x0_SubDetectors.SetMaximum(mbmax);
  stack_x0_SubDetectors.Draw("HIST");
  stack_x0_SubDetectors.GetXaxis()->SetLimits(etamin,etamax);
  //
  
  // Legenda
  TLegend* theLegend_SubDetectors = new TLegend(0.70, 0.70, 0.89, 0.89);
  theLegend_SubDetectors->AddEntry(hist_x0_Outside,  "Outside",  "f");
  theLegend_SubDetectors->AddEntry(hist_x0_TEC,      "TEC",      "f");
  theLegend_SubDetectors->AddEntry(hist_x0_TOB,      "TOB",      "f");
  theLegend_SubDetectors->AddEntry(hist_x0_IB,       "TIB+TID",  "f");
  theLegend_SubDetectors->AddEntry(hist_x0_Pixel,    "Pixel",    "f");
  theLegend_SubDetectors->AddEntry(hist_x0_BeamPipe, "Beam Pipe","f");
  theLegend_SubDetectors->Draw();
  //
  
  // Store
  can_SubDetectors.Update();
  can_SubDetectors.SaveAs( Form("%s/Tracker_SubDetectors_X0.eps",  theDirName.Data()) );
  can_SubDetectors.SaveAs( Form("%s/Tracker_SubDetectors_X0.gif",  theDirName.Data()) );
  //
  
  
  // Second Plot: BeamPipe + SEN + ELE + CAB + COL + SUP + OTH/AIR + Outside
  // stack
  TString stackTitle_Materials = "Tracker Material Budget;#eta;X/X_{0}";
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
  stack_x0_Materials.SetMinimum(mbmin);
  stack_x0_Materials.SetMaximum(mbmax);
  stack_x0_Materials.Draw("HIST");
  stack_x0_Materials.GetXaxis()->SetLimits(etamin,etamax);
  //
  
  // Legenda
  TLegend* theLegend_Materials = new TLegend(0.70, 0.70, 0.89, 0.89);
  theLegend_Materials->AddEntry(hist_x0_Outside,  "Outside",         "f");
  theLegend_Materials->AddEntry(hist_x0_OTH,      "Other",       "f");
  theLegend_Materials->AddEntry(hist_x0_SUP,      "Support",     "f");
  theLegend_Materials->AddEntry(hist_x0_COL,      "Cooling",     "f");
  theLegend_Materials->AddEntry(hist_x0_CAB,      "Cables",      "f");
  theLegend_Materials->AddEntry(hist_x0_ELE,      "Electronics", "f");
  theLegend_Materials->AddEntry(hist_x0_SEN,      "Sensitive",   "f");
  theLegend_Materials->AddEntry(hist_x0_BeamPipe, "Beam Pipe",   "f");
  theLegend_Materials->Draw();
  //
  
  // Store
  can_Materials.Update();
  can_Materials.SaveAs( Form("%s/Tracker_Materials_X0.eps",  theDirName.Data()) );
  can_Materials.SaveAs( Form("%s/Tracker_Materials_X0.gif",  theDirName.Data()) );
  //
  
}

