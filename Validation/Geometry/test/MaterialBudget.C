// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

// data dirs
TString theDirName = "Images";
//

// data files
TString theDetectorFileName;
TString theDetector;
//
TFile* theDetectorFile;
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
TProfile* prof_l0_det_total;
//
TProfile2D* prof2d_x0_det_total;
//

const std::vector<std::string> DETECTORS{"BeamPipe", "TIB", "TIDF", "TIDB",
      "InnerServices", "TOB",
      "TEC", "TkStrct", "PixBar",
      "PixFwdPlus", "PixFwdMinus",
      "Tracker", "TrackerSum",
      "Pixel", "Strip",
      "InnerTracker",
      "Phase1PixelBarrel", "Phase2OTBarrel",
      "Phase2OTForward", "Phase2PixelEndcap"};

std::map<std::string, std::vector<std::string> > COMPOUNDS {
  {"TrackerSum", {"TIB", "TIDF", "TIDB", "BeamPipe", "InnerServices", "TOB", "TEC", "TkStruct", "PixBar", "PixFwdPlus", "PixFwdMinus"}},
  {"Pixel", {"PixBar", "PixFwdMinus", "PixFwdPlus"}},
  {"Strip", {"TIB", "TIDF", "TIDB", "InnerServices", "TOB", "TEC"}},
  {"InnerTracker", {"TIB", "TIDF", "TIDB", "InnerServices"}}
};

//forward declaration of functions
void createPlots(TString plot);
void create2DPlots(TString plot);
void createRatioPlots(TString plot);
void drawEtaValues();
void makeColorTable();

using namespace std;

bool checkFile(const char * filename) {
  struct stat sb;
  if (stat(filename, &sb) == -1) {
    cerr << "Error, missing file: " << filename << endl;
    return false;
  }
  return true;
}

// Main
void MaterialBudget(TString detector) {

  //
  gROOT->SetStyle("Plain");

  // detector
  theDetector = detector;
  if( (std::find(DETECTORS.begin(), DETECTORS.end(), theDetector.Data()) == DETECTORS.end())
      && (COMPOUNDS.find(detector.Data()) == COMPOUNDS.end())) {
    cerr << "MaterialBudget - ERROR detector not found: |" << theDetector.Data() << "|" << endl;
    exit(0);
  }

  theDetectorFileName = "matbdg_" + theDetector + ".root";
  // If the user supplied a compound name, a special treatment is
  // needed, and a loop is involved. The initial filename that has to
  // be opened is the first in the list of associated compounds.
  if (COMPOUNDS.find(theDetector.Data()) != COMPOUNDS.end()) {
    theDetectorFileName = "matbdg_" + TString(COMPOUNDS[theDetector.Data()][0]) +".root";
  }

  if (! checkFile(theDetectorFileName.Data()))
    return;

  // open root files
  cout << "*** Open file... " << endl;
  cout << theDetectorFileName << endl;
  cout << "***" << endl;
  theDetectorFile = new TFile(theDetectorFileName);

  //output root file
  TString outputPlotsFileName = theDirName + "/matbdg_" + theDetector + "_plots.root";
  TFile* outputPlotsFile = new TFile(outputPlotsFileName, "RECREATE");

  // plots
  createPlots("x_vs_eta");
  createPlots("x_vs_phi");
  //  createPlots("x_vs_R");
  createPlots("l_vs_eta");
  createPlots("l_vs_phi");
  //  createPlots("l_vs_R");
  create2DPlots("x_vs_eta_vs_phi");
  create2DPlots("l_vs_eta_vs_phi");
  create2DPlots("x_vs_z_vs_R");
  create2DPlots("l_vs_z_vs_R");
  create2DPlots("x_vs_z_vs_Rsum");
  create2DPlots("l_vs_z_vs_Rsum");
  //
  createRatioPlots("x_over_l_vs_eta");
  createRatioPlots("x_over_l_vs_phi");
  //
}

// Plots
void createPlots(TString plot) {
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  if(plot.CompareTo("x_vs_eta") == 0) {
    plotNumber = 10;
    abscissaName = TString("#eta");
    ordinateName = TString("x/X_{0}");
  } else if(plot.CompareTo("x_vs_phi") == 0) {
    plotNumber = 20;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("x/X_{0}");
  } else if(plot.CompareTo("x_vs_R") == 0) {
    plotNumber = 40;
    abscissaName = TString("R [mm]");
    ordinateName = TString("x/X_{0}");
  } else if(plot.CompareTo("l_vs_eta") == 0) {
    plotNumber = 1010;
    abscissaName = TString("#eta");
    ordinateName = TString("x/#lambda_{0}");
  } else if(plot.CompareTo("l_vs_phi") == 0) {
    plotNumber = 1020;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("x/#lambda_{0}");
  } else if(plot.CompareTo("l_vs_R") == 0) {
    plotNumber = 1040;
    abscissaName = TString("R [mm]");
    ordinateName = TString("x/#lambda_{0}");
  } else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }
  
  // get TProfiles
  prof_x0_det_total = (TProfile*)theDetectorFile->Get(Form("%u", plotNumber));
  prof_x0_det_SUP   = (TProfile*)theDetectorFile->Get(Form("%u", 100 + plotNumber));
  prof_x0_det_SEN   = (TProfile*)theDetectorFile->Get(Form("%u", 200 + plotNumber));
  prof_x0_det_CAB   = (TProfile*)theDetectorFile->Get(Form("%u", 300 + plotNumber));
  prof_x0_det_COL   = (TProfile*)theDetectorFile->Get(Form("%u", 400 + plotNumber));
  prof_x0_det_ELE   = (TProfile*)theDetectorFile->Get(Form("%u", 500 + plotNumber));
  prof_x0_det_OTH   = (TProfile*)theDetectorFile->Get(Form("%u", 600 + plotNumber));
  prof_x0_det_AIR   = (TProfile*)theDetectorFile->Get(Form("%u", 700 + plotNumber));

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
  if (COMPOUNDS.find(theDetector.Data()) != COMPOUNDS.end()) {
    for (auto subDetector : COMPOUNDS[theDetector.Data()]) {
      TString subDetectorFileName = "matbdg_" + TString(subDetector) + ".root";
      // The first file, corresponding to first detector of the
      // compound, has already been opened and digested. Move to the
      // others.
      if (theDetectorFileName == subDetectorFileName)
        continue;
      // open file
      if (! checkFile(subDetectorFileName))
        continue;
      TFile* subDetectorFile = new TFile(subDetectorFileName);
      cout << "*** Open file... " << endl;
      cout << subDetectorFileName << endl;
      cout << "***" << endl;
      // subdetector profiles
      prof_x0_det_total = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      prof_x0_det_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
      prof_x0_det_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
      prof_x0_det_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
      prof_x0_det_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
      prof_x0_det_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
      prof_x0_det_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
      prof_x0_det_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
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
  TString stackTitle = "Material Budget " + theDetector + Form( ";%s;%s",abscissaName.Data(),ordinateName.Data() );
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
  TString canname("MBCan_1D_" + theDetector + "_" + plot);
  TCanvas can(canname,canname,800,800);
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
  //  can.SaveAs( Form( "%s/%s_%s.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can.SaveAs( Form( "%s/%s_%s.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can.SaveAs( Form( "%s/%s_%s.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  can.SaveAs( Form( "%s/%s_%s.png",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can.Write();
  //
  
}

// Plots
void create2DPlots(TString plot) {
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  TString quotaName = "dummy";
  Int_t zLog = 0;
  Int_t iDrawEta = 0;
  Int_t iRebin = 0; //Rebin
  Double_t histoMin = -1.;
  Double_t histoMax = -1.;
  if(plot.CompareTo("x_vs_eta_vs_phi") == 0) {
    plotNumber = 30;
    abscissaName = TString("#eta");
    ordinateName = TString("#varphi");
    quotaName    = TString("x/X_{0}");
    iRebin = 1;
    zLog = 0;
  } else if(plot.CompareTo("l_vs_eta_vs_phi") == 0) {
    plotNumber = 1030;
    abscissaName = TString("#eta");
    ordinateName = TString("#varphi");
    quotaName    = TString("x/#lambda_{0}");
    iRebin = 1;
    zLog = 0;
  } else if(plot.CompareTo("x_vs_z_vs_Rsum") == 0) {
    plotNumber = 50;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("#Sigmax/X_{0}");
    zLog = 0;
    iDrawEta = 1;
    histoMin = 0.;
    histoMax = 2.5;
  } else if(plot.CompareTo("x_vs_z_vs_R") == 0) {
    plotNumber = 60;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("1/X_{0}");
    iDrawEta = 1;
    zLog = 1;
    histoMin = 0.00001;
    histoMax = 0.01;  
  } else if(plot.CompareTo("l_vs_z_vs_Rsum") == 0) {
    plotNumber = 1050;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("#Sigmax/#lambda_{0}");
    iDrawEta = 1;
    histoMin = 0.;
    histoMax = 1.;
    zLog = 0;
  } else if(plot.CompareTo("l_vs_z_vs_R") == 0) {
    plotNumber = 1060;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("1/#lambda_{0}");
    iDrawEta = 1;
    zLog = 1;
    histoMin = 0.001;
    histoMax = 0.9;
  } else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }
  
  // get TProfiles
  prof2d_x0_det_total = (TProfile2D*)theDetectorFile->Get(Form("%u", plotNumber));
  
  // histos
  TH2D* hist_x0_total = (TH2D*)prof2d_x0_det_total->ProjectionXY();
  //
  
  if (COMPOUNDS.find(theDetector.Data()) != COMPOUNDS.end()) {
    for (auto subDetector : COMPOUNDS[theDetector.Data()]) {
      // file name
      TString subDetectorFileName = "matbdg_" + TString(subDetector) + ".root";
      // The first file, corresponding to first detector of the
      // compound, has already been opened and digested. Move to the
      // others.
      if (theDetectorFileName == subDetectorFileName)
        continue;
      // open file
      if (! checkFile(subDetectorFileName))
        continue;
      TFile* subDetectorFile = new TFile(subDetectorFileName);
      cout << "*** Open file... " << endl;
      cout << subDetectorFileName << endl;
      cout << "***" << endl;
      // subdetector profiles
      prof2d_x0_det_total = (TProfile2D*)subDetectorFile->Get(Form("%u", plotNumber));
      // add to summary histogram
      hist_x0_total->Add( (TH2D*)prof2d_x0_det_total->ProjectionXY("B"), +1.000 );
    }
  }
  //
  
  // properties
  gStyle->SetPalette(1);
  gStyle->SetStripDecimals(false);
  //
  
  // Create "null" histo
  Double_t minX = 1.03*hist_x0_total->GetXaxis()->GetXmin();
  Double_t maxX = 1.03*hist_x0_total->GetXaxis()->GetXmax();
  Double_t minY = 1.03*hist_x0_total->GetYaxis()->GetXmin();
  Double_t maxY = 1.03*hist_x0_total->GetYaxis()->GetXmax();

  //  TH2F *frame = new TH2F("frame","",10,-3100.,3100.,10,-50.,1400.); 
  TH2F *frame = new TH2F("frame","",10,minX,maxX,10,minY,maxY); 
  frame->SetMinimum(0.1);
  frame->SetMaximum(10.);
  frame->GetXaxis()->SetTickLength(frame->GetXaxis()->GetTickLength()*0.50);
  frame->GetYaxis()->SetTickLength(frame->GetXaxis()->GetTickLength()/4.);

  // Ratio
  if (iRebin){
    hist_x0_total->Rebin2D();
  }

  // stack
  TString hist2dTitle(quotaName+" "+theDetector+";"+abscissaName+";"+ordinateName+";"+quotaName);
  
  TH2D* hist2d_x0_total = hist_x0_total; //->Rebin2D(5,5);
  //  hist2d_x0_total->SetTitle(hist2dTitle);
  frame->SetTitle(hist2dTitle);
  frame->SetTitleOffset(0.5,"Y");

  if ( histoMin != -1. ) hist2d_x0_total->SetMinimum(histoMin);      
  if ( histoMax != -1. ) hist2d_x0_total->SetMaximum(histoMax);      


  //
  TString can2name("MBCan_2D_" + theDetector + "_" + plot);
  TCanvas can2(can2name,can2name,2480+248,580+58+58);
  can2.SetTopMargin(0.1);
  can2.SetBottomMargin(0.1);
  can2.SetLeftMargin(0.04);
  can2.SetRightMargin(0.06);
  can2.SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleBorderSize(0);
  //
  

  // Color palette
  gStyle->SetPalette(1);

  // Log?
  can2.SetLogz(zLog);

  // Draw in colors
  frame->Draw(); 
  hist2d_x0_total->Draw("COLZsame"); //Dummy draw to create the palette object

  // Store
  can2.Update();

  //Aesthetic
  TPaletteAxis *palette = 
    (TPaletteAxis*)hist2d_x0_total->GetListOfFunctions()->FindObject("palette");
  palette->SetX1NDC(0.945);
  palette->SetX2NDC(0.96);
  palette->SetY1NDC(0.1);
  palette->SetY2NDC(0.9);
  palette->GetAxis()->SetTickSize(.01);
  if ( zLog==1 ) {  palette->GetAxis()->SetLabelOffset(-0.01); }
  palette->GetAxis()->SetTitle("");
  TLatex *paletteTitle = new TLatex(1.12*maxX,maxY,quotaName); 
  paletteTitle->SetTextAngle(90.);
  paletteTitle->SetTextSize(0.05);
  paletteTitle->SetTextAlign(31);
  paletteTitle->Draw();     
  hist2d_x0_total->GetYaxis()->SetTickLength(hist2d_x0_total->GetXaxis()->GetTickLength()/4.);
  hist2d_x0_total->GetYaxis()->SetTickLength(hist2d_x0_total->GetXaxis()->GetTickLength()/4.);
  hist2d_x0_total->SetTitleOffset(0.5,"Y");
  hist2d_x0_total->GetXaxis()->SetNoExponent(true);
  hist2d_x0_total->GetYaxis()->SetNoExponent(true);

  //Add eta labels
  if( iDrawEta ) drawEtaValues();

  can2.Modified();

  //Color plots are not supported!
  //  can2.SaveAs( Form( "%s/%s_%s_col.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can2.SaveAs( Form( "%s/%s_%s_col.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can2.SaveAs( Form( "%s/%s_%s_col.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //
  
//  makeColorTable();  // Grayscale palette

  hist2d_x0_total->SetContour(255);      
    
  // Store
  can2.Update();

  can2.Modified();

  //  can2.SaveAs( Form( "%s/%s_%s_bw.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can2.SaveAs( Form( "%s/%s_%s_bw.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can2.SaveAs( Form( "%s/%s_%s_bw.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  can2.SaveAs( Form( "%s/%s_%s_bw.png",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can2.Write();
  //
  
  // restore properties
  gStyle->SetStripDecimals(true);
  //
}

void createRatioPlots(TString plot) {
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  ordinateName = TString("(x/X_{0})/(x/#lambda_{0})");
  if(plot.CompareTo("x_over_l_vs_eta") == 0) {
    plotNumber = 10;
    abscissaName = TString("#eta");
  } else if(plot.CompareTo("x_over_l_vs_phi") == 0) {
    plotNumber = 20;
    abscissaName = TString("#varphi [rad]");
  } else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }
  
  // get TProfiles
  prof_x0_det_total = (TProfile*)theDetectorFile->Get(Form("%u", plotNumber));
  prof_l0_det_total = (TProfile*)theDetectorFile->Get(Form("%u", 1000+plotNumber));
  
  // histos
  TH1D* hist_x0_total = (TH1D*)prof_x0_det_total->ProjectionX();
  TH1D* hist_l0_total = (TH1D*)prof_l0_det_total->ProjectionX();
  //
  if (COMPOUNDS.find(theDetector.Data()) != COMPOUNDS.end()) {
    for (auto subDetector : COMPOUNDS[theDetector.Data()]) {
      // file name
      TString subDetectorFileName = "matbdg_" + TString(subDetector) + ".root";
      // The first file, corresponding to first detector of the
      // compound, has already been opened and digested. Move to the
      // others.
      if (theDetectorFileName == subDetectorFileName)
        continue;
      // open file
      if (! checkFile(subDetectorFileName))
        continue;
      TFile* subDetectorFile = new TFile(subDetectorFileName);
      cout << "*** Open file... " << endl;
      cout << subDetectorFileName << endl;
      cout << "***" << endl;
      // subdetector profiles
      prof_x0_det_total = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
      prof_l0_det_total = (TProfile*)subDetectorFile->Get(Form("%u", 1000+plotNumber));
      // add to summary histogram
      hist_x0_total->Add( (TH1D*)prof_x0_det_total->ProjectionX("B"), +1.000 );
      hist_l0_total->Add( (TH1D*)prof_l0_det_total->ProjectionX("B"), +1.000 );
    }
  }
  //
  TH1D* hist_x0_over_l0_total = new TH1D(*hist_x0_total);
  hist_x0_over_l0_total->Divide(hist_l0_total);
  TString histTitle = Form( "Material Budget %s;%s;%s", theDetector.Data() ,abscissaName.Data(),ordinateName.Data() );
  hist_x0_over_l0_total->SetTitle(histTitle);
  // properties
  hist_x0_over_l0_total->SetMarkerStyle(1);
  hist_x0_over_l0_total->SetMarkerSize(3);
  hist_x0_over_l0_total->SetMarkerColor(kBlue);
  //
  
  // canvas
  TString canRname("MBRatio_" + theDetector + "_" + plot);
  TCanvas canR(canRname,canRname,800,800);
  canR.Range(0,0,25,25);
  canR.SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  //
  
  // Draw
  hist_x0_over_l0_total->Draw("E1");
  //
  
  // Store
  canR.Update();
  //  canR.SaveAs( Form( "%s/%s_%s.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  canR.SaveAs( Form( "%s/%s_%s.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  canR.SaveAs( Form( "%s/%s_%s.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  canR.SaveAs( Form( "%s/%s_%s.png",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  canR.Write();
}


void drawEtaValues(){

  //Add eta labels
  Float_t etas[33] = {-3.4, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3.0, 3.4};
  Float_t etax = 2940.;
  Float_t etay = 1240.;
  Float_t lineL = 100.;
  Float_t offT = 10.;
  Float_t pi = 3.1415926;



  for (Int_t ieta=0; ieta<33; ieta++){
    Float_t th = 2*atan(exp(-etas[ieta]));
    Int_t talign = 21;

    //IP
    TLine *lineh = new TLine(-20.,0.,20.,0.); 
    lineh->Draw();  
    TLine *linev = new TLine(0.,-10.,0.,10.); 
    linev->Draw();  

    Float_t x1;
    Float_t y1;
    if ( etas[ieta]>-1.6 && etas[ieta]<1.6 ){
      x1 = etay/tan(th);
      y1 = etay;
    } else if ( etas[ieta]<=-1.6 ) {
      x1 = -etax;
      y1 = -etax*tan(th);
      talign = 11;
    } else if ( etas[ieta]>=1.6 ){
      x1 = etax;
      y1 = etax*tan(th);
      talign = 31;
    }
    Float_t x2 = x1+lineL*cos(th);
    Float_t y2 = y1+lineL*sin(th);
    Float_t xt = x2;
    Float_t yt = y2+offT;
    //      cout << isign << " " << th*180./pi << " " << x1 << " " << y1 << "\n";
    TLine *line1 = new TLine(x1,y1,x2,y2); 
    line1->Draw();  
    char text[20];
    int rc = sprintf(text, "%3.1f", etas[ieta]);
    TLatex *t1;
    if ( etas[ieta] == 0 ) {
      t1 = new TLatex(xt,yt,"#eta = 0"); 
    } else {
      t1 = new TLatex(xt,yt,text); 
    }
    t1->SetTextSize(0.03);
    t1->SetTextAlign(talign);
    t1->Draw();     
    
  }

}

void makeColorTable(){

  const Int_t NRGBs = 2;
  const Int_t NCont = 255;
  Double_t stops[NRGBs] = {0.00, 1.00};
  Double_t red[NRGBs]   = { .99, 0.00};
  Double_t green[NRGBs] = { .99, 0.00};
  Double_t blue[NRGBs]  = { .99, 0.00};

  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);

}
