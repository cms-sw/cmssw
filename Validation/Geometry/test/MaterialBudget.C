// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// data dirs
TString theDirName = "Images";
//

// data files
TString theDetectorFileName;
//TString theTkStructFileName;
TString theDetector;
//

// histograms
TProfile* prof_x0_det_total;
TProfile* prof_x0_det_SUP; 
TProfile* prof_x0_det_SEN;
TProfile* prof_x0_det_CAB;
TProfile* prof_x0_det_COL;
TProfile* prof_x0_det_ELE;
TProfile* prof_x0_det_OTH;
TProfile* prof_x0_det_AIR;
TProfile* prof_x0_str_total;
TProfile* prof_x0_str_SUP; 
TProfile* prof_x0_str_SEN;
TProfile* prof_x0_str_CAB;
TProfile* prof_x0_str_COL;
TProfile* prof_x0_str_ELE;
TProfile* prof_x0_str_OTH;
TProfile* prof_x0_str_AIR;
//

using namespace std;

// Main
MaterialBudget(TString detector) {
  // detector
  theDetector = detector;
  if(
     theDetector!="TIB" && theDetector!="TIDF" && theDetector!="TIDB" 
     && theDetector!="TOB" && theDetector!="TEC" && theDetector!="TkStrct" 
     && theDetector!="PixBar" && theDetector!="PixFwdPlus" && theDetector!="PixFwdMinus" 
     && theDetector!="Tracker" && theDetector!="TrackerSum"
     && theDetector!="Pixel" && theDetector!="Strip"
     ){
    cerr << "MaterialBudget - ERROR detector not found " << theDetector << endl;
    break;
  }
  //
  
  // files
  theDetectorFileName = "matbdg_" + theDetector + ".root";
  unsigned int iFirst = 1;
  unsigned int iLast  = 8;
  if(theDetector == "TrackerSum") {
    iFirst = 1;
    iLast  = 8;
    theDetectorFileName = "matbdg_TIB.root";
  }
  if(theDetector == "Pixel") {
    iFirst = 7;
    iLast  = 8;
    theDetectorFileName = "matbdg_PixBar.root";
  }
  if(theDetector == "Strip") {
    iFirst = 1;
    iLast  = 4;
    theDetectorFileName = "matbdg_TIB.root";
  }
  cout << "*** Open file... " << endl;
  cout << theDetectorFileName << endl;
  cout << "***" << endl;
  //
  
  // open root files
  TFile* theDetectorFile = new TFile(theDetectorFileName);
  //
  
  // get TProfiles
  prof_x0_det_total = (TProfile*)theDetectorFile->Get("10");
  prof_x0_det_SUP   = (TProfile*)theDetectorFile->Get("110");
  prof_x0_det_SEN   = (TProfile*)theDetectorFile->Get("210");
  prof_x0_det_CAB   = (TProfile*)theDetectorFile->Get("310");
  prof_x0_det_COL   = (TProfile*)theDetectorFile->Get("410");
  prof_x0_det_ELE   = (TProfile*)theDetectorFile->Get("510");
  prof_x0_det_OTH   = (TProfile*)theDetectorFile->Get("610");
  prof_x0_det_AIR   = (TProfile*)theDetectorFile->Get("710");
  
  // histos
  TH1D* hist_x0_total = (TH1D*)prof_x0_det_total->ProjectionX();
  TH1D* hist_x0_SUP   = (TH1D*)prof_x0_det_SUP->ProjectionX();
  TH1D* hist_x0_SEN   = (TH1D*)prof_x0_det_SEN->ProjectionX();
  TH1D* hist_x0_CAB   = (TH1D*)prof_x0_det_CAB->ProjectionX();
  TH1D* hist_x0_COL   = (TH1D*)prof_x0_det_COL->ProjectionX();
  TH1D* hist_x0_ELE   = (TH1D*)prof_x0_det_ELE->ProjectionX();
  TH1D* hist_x0_OTH   = (TH1D*)prof_x0_det_OTH->ProjectionX();
  TH1D* hist_x0_AIR   = (TH1D*)prof_x0_det_AIR->ProjectionX();
  //
  if(theDetector=="TrackerSum" || theDetector=="Pixel" || theDetector=="Strip") {
    TString subDetector = "TIB";
    for(unsigned int i_detector=iFirst; i_detector<=iLast; i_detector++) {
      switch(i_detector) {
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
      default: cout << " something wrong" << endl;
      }
      // file name
      TString subDetectorFileName = "matbdg_" + subDetector + ".root";
      // open file
      TFile* subDetectorFile = new TFile(subDetectorFileName);
      cout << "*** Open file... " << endl;
      cout << subDetectorFileName << endl;
      cout << "***" << endl;
      // subdetector profiles
      prof_x0_det_total = (TProfile*)subDetectorFile->Get("10");
      prof_x0_det_SUP   = (TProfile*)subDetectorFile->Get("110");
      prof_x0_det_SEN   = (TProfile*)subDetectorFile->Get("210");
      prof_x0_det_CAB   = (TProfile*)subDetectorFile->Get("310");
      prof_x0_det_COL   = (TProfile*)subDetectorFile->Get("410");
      prof_x0_det_ELE   = (TProfile*)subDetectorFile->Get("510");
      prof_x0_det_OTH   = (TProfile*)subDetectorFile->Get("610");
      prof_x0_det_AIR   = (TProfile*)subDetectorFile->Get("710");
      // add to summary histogram
      hist_x0_total->Add( (TH1D*)prof_x0_det_total->ProjectionX("B"), +1.000 );
      hist_x0_SUP->Add(   (TH1D*)prof_x0_det_SUP->ProjectionX("B")  , +1.000 );
      hist_x0_SEN->Add(   (TH1D*)prof_x0_det_SEN->ProjectionX("B")  , +1.000 );
      hist_x0_CAB->Add(   (TH1D*)prof_x0_det_CAB->ProjectionX("B")  , +1.000 );
      hist_x0_COL->Add(   (TH1D*)prof_x0_det_COL->ProjectionX("B")  , +1.000 );
      hist_x0_ELE->Add(   (TH1D*)prof_x0_det_ELE->ProjectionX("B")  , +1.000 );
      hist_x0_OTH->Add(   (TH1D*)prof_x0_det_OTH->ProjectionX("B")  , +1.000 );
      hist_x0_AIR->Add(   (TH1D*)prof_x0_det_AIR->ProjectionX("B")  , +1.000 );
    }
  }
  //
  
  // properties
  hist_x0_total->SetMarkerStyle(1);
  hist_x0_total->SetMarkerSize(3);
  hist_x0_total->SetMarkerColor(kBlack);
  //
  hist_x0_SUP->SetFillColor(13); // Support     = dark gray
  hist_x0_SEN->SetFillColor(27); // Sensitive   = brown
  hist_x0_CAB->SetFillColor(46); // Cabling     = red
  hist_x0_COL->SetFillColor(38); // Cooling     = light blue
  hist_x0_ELE->SetFillColor(30); // Electronics = green
  hist_x0_OTH->SetFillColor(42); // Other       = orange
  hist_x0_AIR->SetFillColor(29); // Air         = light bluegreen
  //
  
  // stack
  TString stackTitle = "Material Budget " + theDetector + ";#eta;x/X_{0}";
  THStack stack_x0("stack_x0",stackTitle);
  stack_x0.Add(hist_x0_SUP);
  stack_x0.Add(hist_x0_SEN);
  stack_x0.Add(hist_x0_CAB);
  stack_x0.Add(hist_x0_COL);
  stack_x0.Add(hist_x0_ELE);
  stack_x0.Add(hist_x0_OTH);
  stack_x0.Add(hist_x0_AIR);
  //
  
  // canvas
  TCanvas can("can","can",800,800);
  can.Range(0,0,25,25);
  can.SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  //
  
  // Draw
  stack_x0.Draw("HIST");
  //
  
  // Legenda
  TLegend* theLegend = new TLegend(0.70, 0.70, 0.89, 0.89);
  theLegend->AddEntry(hist_x0_SUP,  "Support",     "f");
  theLegend->AddEntry(hist_x0_SEN,  "Sensitive",   "f");
  theLegend->AddEntry(hist_x0_CAB,  "Cables",      "f");
  theLegend->AddEntry(hist_x0_COL,  "Cooling",     "f");
  theLegend->AddEntry(hist_x0_ELE,  "Electronics", "f");
  theLegend->AddEntry(hist_x0_OTH,  "Other",       "f");
  theLegend->AddEntry(hist_x0_AIR,  "Air",         "f");
  //  theLegend->AddEntry(hist_x0_total,"Total",       "e");
  theLegend->Draw();
  //
  
  // Store
  can.Update();
  can.SaveAs( Form("%s/%s_X0.eps",  theDirName.Data(), theDetector.Data()) );
  can.SaveAs( Form("%s/%s_X0.gif",  theDirName.Data(), theDetector.Data()) );
  //
  
}

