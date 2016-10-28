// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include "TStyle.h"
#include <sys/types.h>
#include <sys/stat.h>

std::vector<const char * > DETECTORS{"TIB", "TIDF", "TIDB",
      "InnerServices", "TOB",
      "TEC", "TkStrct", "PixBar",
      "PixFwdPlus", "PixFwdMinus",
      "Phase1PixelBarrel", "Phase2OTBarrel",
      "Phase2OTForward", "Phase2PixelEndcap",
      "BeamPipe"};//,
      // "Tracker", "TrackerSum",
      // "Pixel", "Strip",
      // "InnerTracker"};

bool checkFile(const char * filename) {
  struct stat sb;
  if (stat(filename, &sb) == -1) {
    cerr << "Error, missing file: " << filename << endl;
    return false;
  }
  return true;
}

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
// For simulation material budget:
//  TkStrct PixBar PixFwdPlus PixFwdMinus TIB TIDF TIDB TOB TEC BeamPipe InnerServices
// For reconstruction material budget:

// histograms
TProfile* prof_x0_XXX;
//
TProfile* prof_x0_SEN;
TProfile* prof_x0_SUP;
TProfile* prof_x0_ELE;
TProfile* prof_x0_CAB;
TProfile* prof_x0_COL;
TProfile* prof_x0_OTH;
TProfile* prof_x0_AIR;
//
std::map<std::string, TH1D*> hist_x0_detectors;
TH1D* hist_x0_IB;
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

// Routine to internally create and save plots related to the
// Simulation geometry. It also returns the overall envelope of the
// tracker material budget, in order to compare it against the one
// computed from the Reconstruction geometry.

void createPlots(TString plot, TH1D** cumulative_matbdg);


// Routine to internally create and save plots related to the
// Reconstruction geometry. It also returns the overall envelope of
// the tracker material budget, in order to compare it against the one
// computed from the Simulation geometry.

void createPlotsReco(const char * reco_file, const char * label, TH1D** cumulative_matbdg);

void assignOrAddIfExists(TH1D** h, TProfile* p) {
  if ( !*h )
    *h = (TH1D*)p->ProjectionX();
  else
    (*h)->Add((TH1D*)p->ProjectionX("B"), +1.000);
}

void setColorIfExists(std::map<std::string, TH1D*> &m, const char * k, int c) {
  if (m.find(k) != m.end())
    m[k]->SetFillColor(c);
}

using namespace std;

// Main
void MaterialBudget_Simul_vs_Reco(const char * reco_file, const char * label="") {

  //TDR style
  setTDRStyle();
  TH1D * cumulative_matbdg_sim = 0;
  TH1D * cumulative_matbdg_rec = 0;

  // plots
  createPlots("x_vs_eta", &cumulative_matbdg_sim);
  createPlotsReco(reco_file, label, &cumulative_matbdg_rec);
  std::cout << "sim: " << cumulative_matbdg_sim << std::endl;
  std::cout << "reco: " << cumulative_matbdg_rec << std::endl;

  TCanvas * cc = new TCanvas("cc", "cc", 1024, 1024);
  if (cumulative_matbdg_sim != 0 && cumulative_matbdg_rec != 0) {
    cumulative_matbdg_sim->SetMinimum(0.); cumulative_matbdg_sim->SetMaximum(3.5);
    cumulative_matbdg_sim->GetXaxis()->SetRangeUser(-3.0, 3.0);
    cumulative_matbdg_sim->SetLineColor(kOrange);
    cumulative_matbdg_rec->SetMinimum(0.); cumulative_matbdg_rec->SetMaximum(3.);
    cumulative_matbdg_rec->SetLineColor(kAzure+1);
    TLegend * l = new TLegend(0.18, 0.8, 0.95, 0.92);
    l->AddEntry(cumulative_matbdg_sim, "Sim Material", "f");
    l->AddEntry(cumulative_matbdg_rec, "Reco Material", "f");
    cumulative_matbdg_sim->Draw("HIST");
    cumulative_matbdg_rec->Draw("HIST SAME");
    l->Draw();
    std::string filename = "MaterialBdg_Reco_vs_Simul_";
    filename += label;
    filename += ".png";
    cc->SaveAs(filename.data());
  }
}

void createPlotsReco(const char * reco_file, const char * label, TH1D ** cumulative_matbdg) {
  std::vector<std::string> sDETS = {"PXB", "PXF", "TIB", "TID", "TOB", "TEC"};
  std::vector<unsigned int> sLAYS = {4, 11, 4, 5, 6, 9};
  std::vector<std::string> sPREF = {"Original_RadLen_vs_Eta_", "RadLen_vs_Eta_"};
  std::vector<int> sCOLORS = {kRed, kBlue, kGreen, kYellow, kOrange, kPink};
  std::vector<TProfile*> profs;
  std::vector<TH1D*> histos;
  std::vector<TH1D*> diffs;
  std::vector<THStack *> stack;
  char name[1000];

  TCanvas * c = new TCanvas("c", "c", 1024, 1024);
  struct stat sb;
  if (stat(reco_file, &sb) == -1 ) {
    std::cerr << "Error opening file: " << reco_file << std::endl;
    return;
  }
  TFile * file = new TFile(reco_file);
  char prefix[100] = "/DQMData/Run 1/RecoMaterialFromRecoTracks/Run summary/";
  file->cd(prefix);
  file->ls();
  THStack *hs = new THStack("hs","");
  for (unsigned int s = 0; s < sPREF.size(); ++s) {
    histos.clear();
    profs.clear();
    stack.push_back(new THStack("hs",""));
    THStack * hs = stack.back();
    for (unsigned int i = 0; i < sDETS.size(); ++i) {
      for (unsigned int j = 1; j <= sLAYS[i]; ++j) {
        memset(name, 0, sizeof(name));
        snprintf(name, sizeof(name), "%s%s%s%d", prefix, sPREF[s].data(), sDETS[i].data(), j);
        profs.push_back((TProfile*)file->Get(name));
        if (profs.back()) {
          histos.push_back(profs.back()->ProjectionX("_px", "hist"));
          diffs.push_back(histos.back());
          histos.back()->SetFillColor(sCOLORS[i]+j);
          histos.back()->SetLineColor(sCOLORS[i]+j+1);
        } else {
          std::cout << "Missing profile " << name << std::endl;
        }
      }  // end of sLAYS
    }  // end of sDETS

    memset(name, 0, sizeof(name));
    snprintf(name, sizeof(name), "CumulativeRecoMatBdg_%s", sPREF[s].data());
    if (sPREF[s] == "RadLen_vs_Eta_") {
      *cumulative_matbdg = new TH1D(name,
                                    name,
                                    histos.front()->GetNbinsX(),
                                    histos.front()->GetXaxis()->GetXmin(),
                                    histos.front()->GetXaxis()->GetXmax());
    }
    for (auto h : histos) {
      hs->Add(h);
      if (*cumulative_matbdg)
        (*cumulative_matbdg)->Add(h, 1.);
    }
    hs->Draw();
    hs->GetYaxis()->SetTitle("RadLen");
    c->Update();
    c->Modified();
    c->SaveAs(std::string(sPREF[s]+std::string("stacked_")+std::string(label)+std::string(".png")).data());
  }  // end of sPREF
  stack.push_back(new THStack("diff",""));
  hs = stack.back();
  for (unsigned int d = 0; d < diffs.size()/2; ++d) {
    diffs[d+diffs.size()/2]->Add(diffs[d], -1.);
    hs->Add(diffs[d+diffs.size()/2]);
  }  // end of diffs
  hs->Draw();
  hs->GetYaxis()->SetTitle("RadLen");
  c->Update();
  c->Modified();
  std::string outfile("RadLen_difference_");
  outfile += label;
  outfile += ".png";
  c->SaveAs(outfile.data());
}

void createPlots(TString plot, TH1D ** cumulative_matbdg){
  std::cout << "Sim at entrance: " << *cumulative_matbdg << std::endl;
  std::vector<const char *> IBs{"TIB", "TIDF", "TIDB", "InnerServices", "Phase1PixelBarrel"};

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
  for (const auto detector : DETECTORS) {
    TString subDetector(detector);
    // file name
    TString subDetectorFileName = "matbdg_" + subDetector + ".root";

    // open file

    struct stat sb;
    if (!checkFile(subDetectorFileName.Data())) {
      std::cerr << "Error opening file: " << subDetectorFileName << std::endl;
      continue;
    }

    TFile* subDetectorFile = new TFile(subDetectorFileName);
    cout << "*** Open file... " << endl;
    cout << subDetectorFileName << endl;
    cout << "***" << endl;

    prof_x0_XXX = (TProfile*)subDetectorFile->Get(Form("%u", plotNumber));

    // Merge together the "inner barrel detectors".
    if (std::find(IBs.begin(), IBs.end(), detector) != IBs.end())
      assignOrAddIfExists(&hist_x0_IB, prof_x0_XXX);

    hist_x0_detectors[detector] = (TH1D*)prof_x0_XXX->ProjectionX();

    if ( *cumulative_matbdg == 0 ) {
      *cumulative_matbdg = new TH1D("CumulativeSimulMatBdg",
                                    "CumulativeSimulMatBdg",
                                    hist_x0_IB->GetNbinsX(),
                                    hist_x0_IB->GetXaxis()->GetXmin(),
                                    hist_x0_IB->GetXaxis()->GetXmax());
      std::cout << "Sim at exit: " << *cumulative_matbdg << std::endl;
    }
    // category profiles
    prof_x0_SUP   = (TProfile*)subDetectorFile->Get(Form("%u", 100 + plotNumber));
    prof_x0_SEN   = (TProfile*)subDetectorFile->Get(Form("%u", 200 + plotNumber));
    prof_x0_CAB   = (TProfile*)subDetectorFile->Get(Form("%u", 300 + plotNumber));
    prof_x0_COL   = (TProfile*)subDetectorFile->Get(Form("%u", 400 + plotNumber));
    prof_x0_ELE   = (TProfile*)subDetectorFile->Get(Form("%u", 500 + plotNumber));
    prof_x0_OTH   = (TProfile*)subDetectorFile->Get(Form("%u", 600 + plotNumber));
    prof_x0_AIR   = (TProfile*)subDetectorFile->Get(Form("%u", 700 + plotNumber));
    // add to summary histogram
    assignOrAddIfExists( &hist_x0_SUP, prof_x0_SUP );
    assignOrAddIfExists( &hist_x0_SEN, prof_x0_SEN );
    assignOrAddIfExists( &hist_x0_CAB, prof_x0_CAB );
    assignOrAddIfExists( &hist_x0_COL, prof_x0_COL );
    assignOrAddIfExists( &hist_x0_ELE, prof_x0_ELE );
    assignOrAddIfExists( &hist_x0_OTH, prof_x0_OTH );
    assignOrAddIfExists( &hist_x0_OTH, prof_x0_AIR );
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

  setColorIfExists(hist_x0_detectors, "BeamPipe", kpipe); // Beam Pipe	 = dark gray
  setColorIfExists(hist_x0_detectors, "Pixel", kpixel);   // Pixel 	 = dark blue
  setColorIfExists(hist_x0_detectors, "Phase1PixelBarrel", kpixel);
  setColorIfExists(hist_x0_detectors, "Phase2OTBarrel", ktib);
  setColorIfExists(hist_x0_detectors, "Phase2OTForward", ktec);
  setColorIfExists(hist_x0_detectors, "Phase2PixelEndcap", ktib);
  setColorIfExists(hist_x0_detectors, "TIB", ktib);	 // TIB and TID  = violet
  setColorIfExists(hist_x0_detectors, "TID", ktib);	 // TIB and TID  = violet
  setColorIfExists(hist_x0_detectors, "TOB", ktob);       // TOB          = red
  setColorIfExists(hist_x0_detectors, "TEC", ktec);       // TEC          = yellow gold
  setColorIfExists(hist_x0_detectors, "TkStrct", kout);   // Support tube = light gray

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
  for (auto const det : hist_x0_detectors) {
    stack_x0_SubDetectors.Add(det.second);
    (*cumulative_matbdg)->Add(det.second, 1);
  }
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

  for (auto const det : hist_x0_detectors)
    theLegend_SubDetectors->AddEntry(det.second, det.first.data(),  "f");

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
  stack_x0_Materials.Add(hist_x0_detectors["BeamPipe"]);
  stack_x0_Materials.Add(hist_x0_SEN);
  stack_x0_Materials.Add(hist_x0_ELE);
  stack_x0_Materials.Add(hist_x0_CAB);
  stack_x0_Materials.Add(hist_x0_COL);
  stack_x0_Materials.Add(hist_x0_SUP);
  stack_x0_Materials.Add(hist_x0_OTH);
  stack_x0_Materials.Add(hist_x0_detectors["TkStrct"]);
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

  theLegend_Materials->AddEntry(hist_x0_detectors["TkStrct"],   "Support and Thermal Screen",  "f");
  theLegend_Materials->AddEntry(hist_x0_detectors["BeamPipe"],  "Beam Pipe",                   "f");
  theLegend_Materials->AddEntry(hist_x0_OTH,       "Other",                       "f");
  theLegend_Materials->AddEntry(hist_x0_SUP,       "Mechanical Structures",       "f");
  theLegend_Materials->AddEntry(hist_x0_COL,       "Cooling",                     "f");
  theLegend_Materials->AddEntry(hist_x0_CAB,       "Cables",                      "f");
  theLegend_Materials->AddEntry(hist_x0_ELE,       "Electronics",                 "f");
  theLegend_Materials->AddEntry(hist_x0_SEN,       "Sensitive",                   "f");
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
