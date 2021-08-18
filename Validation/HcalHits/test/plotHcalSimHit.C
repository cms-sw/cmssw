// Usage:
// .L plotHcalSimHit.C+g
//             For comparing basic histograms from Sim step between 2 sources
//  plotCompare(infile1, infile2, text1, text2, type, save);
//
//  infile1, infile2  std::string  The ROOT files of the two input sources
//  text1, text2      std::string  Character strings describing the two inputs
//  type              int          Detector type (0:HB; 1:HE; 2:HO; 3:HF)
//  save              bool         Flag to indicate the canvas to be saved
//                                 as a pdf file (true) or not (false)
//
//////////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "THStack.h"
#include "TLegend.h"
#include "TMath.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

void setTDRStyle() {
  TStyle *tdrStyle = new TStyle("tdrStyle", "Style for P-TDR");

  // For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600);  //Height of canvas
  tdrStyle->SetCanvasDefW(600);  //Width of canvas
  tdrStyle->SetCanvasDefX(0);    //POsition on screen
  tdrStyle->SetCanvasDefY(0);

  // For the Pad:
  tdrStyle->SetPadBorderMode(0);
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
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

  //For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

  // For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat("mr");

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
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.7);
  tdrStyle->SetTitleYOffset(0.7);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

  // For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.03, "XYZ");

  // For the axis:
  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

  tdrStyle->cd();
}

void plotCompare(std::string infile1, std::string infile2, std::string text1, std::string text2, int type, bool save) {
  const int ndets = 4;
  const int nhist = 5;
  std::string names[ndets][nhist] = {{"Hit05", "Hit17", "Hit21", "Hit25", "Hit29"},
                                     {"Hit06", "Hit18", "Hit22", "Hit26", "Hit30"},
                                     {"Hit07", "Hit19", "Hit23", "Hit27", "Hit31"},
                                     {"Hit08", "Hit20", "Hit24", "Hit28", "Hit32"}};
  std::string xtitl[nhist] = {"Hits", "Depth", "i#eta", "i#phi", "Energy (MeV)"};
  std::string dets[ndets] = {"HB", "HE", "HO", "HF"};
  int nbins[nhist] = {10, 1, 1, 1, 1};
  bool logs[nhist] = {0, 0, 0, 0, 1};
  double xmax[nhist] = {2000, -1, -1, -1, 2.0};
  int colors[2] = {2, 4};
  int style[2] = {1, 2};

  if (type < 0 || type >= ndets)
    type = 0;
  setTDRStyle();
  TFile *file1 = new TFile(infile1.c_str());
  TFile *file2 = new TFile(infile2.c_str());
  if ((file1 != nullptr) && (file2 != nullptr)) {
    char name[20], cname[50];
    for (int k = 0; k < nhist; ++k) {
      sprintf(name, "%s", names[type][k].c_str());
      sprintf(cname, "c_%s", names[type][k].c_str());
      TH1D *hist[2];
      hist[0] = (TH1D *)(file1->FindObjectAny(name));
      hist[1] = (TH1D *)(file2->FindObjectAny(name));
      if ((hist[0] != nullptr) && (hist[1] != nullptr)) {
        TCanvas *pad = new TCanvas(cname, cname, 800, 600);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        if (logs[k] > 0)
          pad->SetLogy();
        TLegend *leg = new TLegend(0.65, 0.69, 0.89, 0.77);
        leg->SetFillColor(kWhite);
        double ymax(0.90);
        for (int i = 0; i < 2; ++i) {
          hist[i]->SetLineStyle(style[i]);
          hist[i]->SetLineColor(colors[i]);
          hist[i]->SetLineWidth(2);
          hist[i]->Rebin(nbins[k]);
          hist[i]->GetXaxis()->SetTitle(xtitl[k].c_str());
          if (xmax[k] > 0)
            hist[i]->GetXaxis()->SetRangeUser(0, xmax[k]);
          if (i == 0)
            hist[i]->Draw("HIST");
          else
            hist[i]->Draw("HIST sames");
          if (i == 0)
            sprintf(name, "%s", text1.c_str());
          else
            sprintf(name, "%s", text2.c_str());
          leg->AddEntry(hist[i], name, "lp");
          pad->Update();
          TPaveStats *st1 = (TPaveStats *)hist[i]->GetListOfFunctions()->FindObject("stats");
          if (st1 != nullptr) {
            double ymin = ymax - 0.06;
            st1->SetFillColor(kWhite);
            st1->SetLineColor(colors[i]);
            st1->SetTextColor(colors[i]);
            st1->SetY1NDC(ymin);
            st1->SetY2NDC(ymax);
            st1->SetX1NDC(0.70);
            st1->SetX2NDC(0.90);
            ymax = ymin;
          }
        }
        leg->Draw("same");
        pad->Update();
        TPaveText *txt1 = new TPaveText(0.65, 0.63, 0.89, 0.685, "blNDC");
        txt1->SetFillColor(kWhite);
        sprintf(name, "%s", dets[type].c_str());
        txt1->AddText(name);
        txt1->Draw("same");
        pad->Update();
        if (save) {
          sprintf(cname, "%s.pdf", pad->GetName());
          pad->Print(cname);
        }
      }
    }
  }
}
