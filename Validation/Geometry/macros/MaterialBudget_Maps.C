#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TPaveStats.h"
#include "TLegend.h"
#include "TChain.h"
#include "TVirtualFitter.h"
#include "TBox.h"
#include "TPaveText.h"
#include "TColor.h"
#include "TProfile.h"
#include "TProfile2D.h"

//#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <ctime>
#include <map>
#include <algorithm>
#include <math.h>
#include <vector>

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
  tdrStyle->SetPadTopMargin(0.07);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.13);
  tdrStyle->SetPadRightMargin(0.19);

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
  tdrStyle->SetTitleYOffset(1.);

// For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.045, "XYZ");

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


void palette(){

	const Int_t NRGBs = 5;
	const Int_t NCont = 255;

	Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
	Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
	Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
	Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
	TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
	gStyle->SetNumberContours(NCont);
}


// data dirs
TString theDirName = "Figures";
//

//
float xmin = -4.;
float xmax = 4.; 

float ymin = -3.1416;
float ymax = 3.1416;

float zmin_x0 = 0.; 
float zmax_x0;

float zmin_lambdaI = 0.; 
float zmax_lambdaI;
//

//
int rebin_x0_x = 1;
int rebin_x0_y = 1;
float norm_x0 = rebin_x0_x*rebin_x0_y;

int rebin_lambdaI_x = 1;
int rebin_lambdaI_y = 1;
float norm_lambdaI = rebin_lambdaI_x*rebin_lambdaI_y;
//

using namespace std;

// Main
void MaterialBudget_Maps() {

	//TDR style
	setTDRStyle(); 
	//palette colori
	palette();

	TString subDetectorFileName = "matbdg_Tracker.root";

	// open file
	TFile *subDetectorFile = new TFile(subDetectorFileName);
	cout << "*** Open file... " << endl;
	cout << subDetectorFileName << endl;
	cout << "***" << endl;

	//case t/X0
	TProfile2D *prof_x0_Tracker_XY = (TProfile2D*)subDetectorFile->Get("30");
	TH2D* hist_x0_Tracker_XY = (TH2D*)prof_x0_Tracker_XY->ProjectionXY();

	// canvas
	TCanvas can_x0("can_x0","can_x0",800,800);
//	can_x0.Range(0,0,25,25);
	
	// Draw
	hist_x0_Tracker_XY->GetXaxis()->SetRangeUser(xmin,xmax);
	hist_x0_Tracker_XY->GetYaxis()->SetRangeUser(ymin,ymax);
	hist_x0_Tracker_XY->Rebin2D(rebin_x0_x,rebin_x0_y);
	hist_x0_Tracker_XY->Scale(1/norm_x0);

	zmax_x0 = hist_x0_Tracker_XY->GetMaximum();
	hist_x0_Tracker_XY->SetMinimum(zmin_x0);
	hist_x0_Tracker_XY->SetMaximum(zmax_x0);

	hist_x0_Tracker_XY->GetXaxis()->SetTitle("#eta");
	hist_x0_Tracker_XY->GetYaxis()->SetTitle("#varphi [rad]");
	hist_x0_Tracker_XY->GetZaxis()->SetTitle("t/X_{0}");
	hist_x0_Tracker_XY->GetZaxis()->SetTitleOffset(1.1);

	hist_x0_Tracker_XY->Draw("zcol");

	// text
	TPaveText* text_x0 = new TPaveText(0.13,0.937,0.35,0.997,"NDC");
	text_x0->SetFillColor(0);
	text_x0->SetBorderSize(0);
	text_x0->AddText("CMS Simulation");
	text_x0->SetTextAlign(13);
	text_x0->Draw();

	// Store
	can_x0.Update();
	//  can_x0.SaveAs( Form( "%s/EtaPhiMap_x0.eps",  theDirName.Data() ) );
	//  can_x0.SaveAs( Form( "%s/EtaPhiMap_x0.gif",  theDirName.Data() ) );
	can_x0.SaveAs( Form( "%s/EtaPhiMap_x0.pdf",  theDirName.Data() ) );
	can_x0.SaveAs( Form( "%s/EtaPhiMap_x0.png",  theDirName.Data() ) );
	can_x0.SaveAs( Form( "%s/EtaPhiMap_x0.root",  theDirName.Data() ) );
	//  can_x0.SaveAs( Form( "%s/EtaPhiMap_x0.C",  theDirName.Data() ) );
	//

//----------------

	//case t/lambdaI
	TProfile2D *prof_lambdaI_Tracker_XY = (TProfile2D*)subDetectorFile->Get("1030");
	TH2D* hist_lambdaI_Tracker_XY = (TH2D*)prof_lambdaI_Tracker_XY->ProjectionXY();

	// canvas
	TCanvas can_lambdaI("can_lambdaI","can_lambdaI",800,800);
//	can_lambdaI.Range(0,0,25,25);
	
	// Draw
	hist_lambdaI_Tracker_XY->GetXaxis()->SetRangeUser(xmin,xmax);
	hist_lambdaI_Tracker_XY->GetYaxis()->SetRangeUser(ymin,ymax);
	hist_lambdaI_Tracker_XY->Rebin2D(rebin_lambdaI_x,rebin_lambdaI_y);
	hist_lambdaI_Tracker_XY->Scale(1/norm_lambdaI);

	zmax_lambdaI = hist_lambdaI_Tracker_XY->GetMaximum();
	hist_lambdaI_Tracker_XY->SetMinimum(zmin_lambdaI);
	hist_lambdaI_Tracker_XY->SetMaximum(zmax_lambdaI);

	hist_lambdaI_Tracker_XY->GetXaxis()->SetTitle("#eta");
	hist_lambdaI_Tracker_XY->GetYaxis()->SetTitle("#varphi [rad]");
	hist_lambdaI_Tracker_XY->GetZaxis()->SetTitle("t/#lambda_{I}");

	hist_lambdaI_Tracker_XY->Draw("zcol");

	// text
	TPaveText* text_lambdaI = new TPaveText(0.13,0.937,0.35,0.997,"NDC");
	text_lambdaI->SetFillColor(0);
	text_lambdaI->SetBorderSize(0);
	text_lambdaI->AddText("CMS Simulation");
	text_lambdaI->SetTextAlign(13);
	text_lambdaI->Draw();

	// Store
	can_lambdaI.Update();
	//  can_lambdaI.SaveAs( Form( "%s/EtaPhiMap_lambdaI.eps",  theDirName.Data() ) );
	//  can_lambdaI.SaveAs( Form( "%s/EtaPhiMap_lambdaI.gif",  theDirName.Data() ) );
	can_lambdaI.SaveAs( Form( "%s/EtaPhiMap_lambdaI.pdf",  theDirName.Data() ) );
	can_lambdaI.SaveAs( Form( "%s/EtaPhiMap_lambdaI.png",  theDirName.Data() ) );
	can_lambdaI.SaveAs( Form( "%s/EtaPhiMap_lambdaI.root",  theDirName.Data() ) );
	//  can_lambdaI.SaveAs( Form( "%s/EtaPhiMap_lambdaI.C",  theDirName.Data() ) );
	//
}
