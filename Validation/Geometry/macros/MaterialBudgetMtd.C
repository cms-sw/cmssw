// include files

#include "tdrStyle.C"
#include "CMS_lumi.C"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "TStyle.h"

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
void createPlots(TString plot);

using namespace std;

// Main
void MaterialBudgetMtd() {
  //TDR style
  setTDRStyle();

  // plots
  createPlots("x_vs_eta");
  createPlots("x_vs_phi");
  createPlots("l_vs_eta");
  createPlots("l_vs_phi");
}

void createPlots(TString plot) {
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  if (plot.CompareTo("x_vs_eta") == 0) {
    plotNumber = 10;
    abscissaName = TString("#eta");
    ordinateName = TString("t/X_{0}");
    ymin = 0.0;
    ymax = 0.8;
    xmin = -4.0;
    xmax = 4.0;
  } else if (plot.CompareTo("x_vs_phi") == 0) {
    plotNumber = 20;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("t/X_{0}");
    ymin = 0.0;
    ymax = 0.8;
    xmin = -3.2;
    xmax = 3.2;
  } else if (plot.CompareTo("l_vs_eta") == 0) {
    plotNumber = 1010;
    abscissaName = TString("#eta");
    ordinateName = TString("t/#lambda_{I}");
    ymin = 0.0;
    ymax = 0.18;
    xmin = -4.0;
    xmax = 4.0;
  } else if (plot.CompareTo("l_vs_phi") == 0) {
    plotNumber = 1020;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("t/#lambda_{I}");
    ymin = 0.0;
    ymax = 0.18;
    xmin = -3.2;
    xmax = 3.2;
  } else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }

  // file name
  TString subDetectorFileName = "matbdg_Mtd.root";

  // open file
  TFile* subDetectorFile = new TFile(subDetectorFileName);
  cout << "*** Open file... " << endl;
  cout << subDetectorFileName << endl;
  cout << "***" << endl;

  TProfile* prof_x0 = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));
  TH1D* hist_x0 = (TH1D*)prof_x0->ProjectionX();
  // category profiles
  TProfile* prof_x0_SUP = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
  TProfile* prof_x0_SEN = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
  TProfile* prof_x0_CAB = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
  TProfile* prof_x0_COL = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
  TProfile* prof_x0_ELE = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
  TProfile* prof_x0_OTH = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
  // add to summary histogram
  TH1D* hist_x0_SUP = (TH1D*)prof_x0_SUP->ProjectionX();
  TH1D* hist_x0_SEN = (TH1D*)prof_x0_SEN->ProjectionX();
  TH1D* hist_x0_CAB = (TH1D*)prof_x0_CAB->ProjectionX();
  TH1D* hist_x0_COL = (TH1D*)prof_x0_COL->ProjectionX();
  TH1D* hist_x0_ELE = (TH1D*)prof_x0_ELE->ProjectionX();
  TH1D* hist_x0_OTH = (TH1D*)prof_x0_OTH->ProjectionX();

  // colors

  int ksen = 27;
  int kele = 46;
  int kcab = kOrange - 8;
  int kcol = 30;
  int ksup = 38;
  int koth = kOrange - 2;

  hist_x0_SEN->SetFillColor(ksen);  // Sensitive   = brown
  hist_x0_ELE->SetFillColor(kele);  // Electronics = red
  hist_x0_CAB->SetFillColor(kcab);  // Cabling     = dark orange
  hist_x0_COL->SetFillColor(kcol);  // Cooling     = green
  hist_x0_SUP->SetFillColor(ksup);  // Support     = light blue
  hist_x0_OTH->SetFillColor(koth);  // Other+Air   = light orange
  //

  TString stackTitle_Materials = Form("Mtd Material Budget;%s;%s", abscissaName.Data(), ordinateName.Data());
  THStack stack_x0_Materials("stack_x0", stackTitle_Materials);
  stack_x0_Materials.Add(hist_x0_SEN);
  stack_x0_Materials.Add(hist_x0_CAB);
  stack_x0_Materials.Add(hist_x0_COL);
  stack_x0_Materials.Add(hist_x0_ELE);
  stack_x0_Materials.Add(hist_x0_OTH);
  stack_x0_Materials.Add(hist_x0_SUP);
  //

  // canvas

  int W = 800;
  int H = 600;
  int H_ref = 600;
  int W_ref = 800;

  // references for T, B, L, R
  float T = 0.08 * H_ref;
  float B = 0.12 * H_ref;
  float L = 0.12 * W_ref;
  float R = 0.04 * W_ref;

  TCanvas* can_Materials = new TCanvas("can_Materials", "can_Materials", 50, 50, W, H);
  can_Materials->Range(0, 0, 25, 25);
  can_Materials->SetFillColor(kWhite);
  can_Materials->SetBorderMode(0);
  can_Materials->SetFrameFillStyle(0);
  can_Materials->SetFrameBorderMode(0);
  can_Materials->SetLeftMargin(L / W);
  can_Materials->SetRightMargin(R / W);
  can_Materials->SetTopMargin(T / H);
  can_Materials->SetBottomMargin(B / H);
  can_Materials->SetTickx(0);
  can_Materials->SetTicky(0);
  gStyle->SetOptStat(0);
  //

  // Draw
  stack_x0_Materials.SetMinimum(ymin);
  stack_x0_Materials.SetMaximum(ymax);
  stack_x0_Materials.Draw("HIST");
  stack_x0_Materials.GetXaxis()->SetLimits(xmin, xmax);
  //

  // Legenda
  TLegend* theLegend_Materials = new TLegend(0.14, 0.8, 0.96, 0.92);
  theLegend_Materials->SetTextAlign(22);
  theLegend_Materials->SetNColumns(3);
  theLegend_Materials->SetFillColor(0);
  theLegend_Materials->SetFillStyle(0);
  theLegend_Materials->SetBorderSize(0);

  theLegend_Materials->AddEntry(hist_x0_SEN, "Sensitive material", "f");
  theLegend_Materials->AddEntry(hist_x0_COL, "Support/cooling", "f");
  theLegend_Materials->AddEntry(hist_x0_ELE, "Electronics/services", "f");
  //  theLegend_Materials->AddEntry(hist_x0_CAB, "Services", "f");
  //  theLegend_Materials->AddEntry(hist_x0_SUP, "Support", "f");
  //  theLegend_Materials->AddEntry(hist_x0_OTH, "Other", "f");
  theLegend_Materials->Draw();

  // writing the lumi information and the CMS "logo"
  int iPeriod = 0;
  writeExtraText = true;
  extraText = "Simulation";

  // second parameter in example_plot is iPos, which drives the position of the CMS logo in the plot
  // iPos=11 : top-left, left-aligned
  // iPos=33 : top-right, right-aligned
  // iPos=22 : center, centered
  // mode generally :
  //   iPos = 10*(alignement 1/2/3) + position (1/2/3 = left/center/right)
  int iPos = 0;

  CMS_lumi(can_Materials, iPeriod, iPos);

  // Store
  can_Materials->Update();
  can_Materials->RedrawAxis();
  can_Materials->Draw();
  // can_Materials->SaveAs( Form( "%s/Mtd_Materials_%s.eps",  theDirName.Data(), plot.Data() ) );
  // can_Materials->SaveAs( Form( "%s/Mtd_Materials_%s.gif",  theDirName.Data(), plot.Data() ) );
  can_Materials->SaveAs(Form("%s/Mtd_Materials_%s.pdf", theDirName.Data(), plot.Data()));
  // can_Materials->SaveAs( Form( "%s/Mtd_Materials_%s.png",  theDirName.Data(), plot.Data() ) );
  can_Materials->SaveAs(Form("%s/Mtd_Materials_%s.root", theDirName.Data(), plot.Data()));
  // can_Materials->SaveAs( Form( "%s/Mtd_Materials_%s.C",  theDirName.Data(), plot.Data() ) );
  //

  delete can_Materials;
}
