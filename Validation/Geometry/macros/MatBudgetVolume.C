///////////////////////////////////////////////////////////////////////////////
//
// etaPhiPlot(fileName, plot, drawLeg, ifEta, maxEta, tag)
//      Make the plots of integrated interaction/radiation/step lengths as a
//      function of eta or phi
// fileName (TString)     Name of the input ROOT file ("matbdg_run3.root")
// plot     (std::string) Type of plot: intl/radl/step ("intl")
// drawLeg  (bool)        Flag to show the legend or not (true)
// ifEta    (bool)        Draw as a function of eta or phi (true)
// maxEta   (double)      Maximum value of x-axis: if -1 use default (5.2)
// tag      (string)      Tag to be added to the name of the canvas ("Run3")
//
// etaPhi2DPlot(fileName, plot, drawLeg, maxEta, tag)
//      Make the 2-D plots as a function of eta and phi with same parameter
//      meanings as those of *etaPhiPlot*
//
///////////////////////////////////////////////////////////////////////////////

// include files
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TROOT.h>
#include <TStyle.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

void etaPhiPlot(TString fileName = "matbdg_run3.root",
                std::string plot = "intl",
                bool drawLeg = true,
                bool ifEta = true,
                double maxEta = 5.2,
                std::string tag = "Run3");
void etaPhi2DPlot(TString fileName = "matbdg_run3.root",
		  std::string plot = "intl",
		  bool drawLeg = true,
		  double maxEta = 5.2,
		  std::string tag = "Run3");
void setStyle();

const int nlay = 13;
int colorLay[nlay] = {2, 2, 2, 2, 2, 3, 5, 4, 8, 6, 3, 7, 1};
int legends[nlay] =  {1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
std::string title[nlay] = {"Beam Pipe", "", "", "", "", "Tracker",
			   "ECAL", "HCAL", "HGCAL", "HF", "Magnet",
			   "MUON", "Forward"};
std::string names[nlay] = {"BEAM", "BEAM1", "BEAM2", "BEAM3", "BEAM4",
			   "Tracker", "ECAL", "HCal", "CALOEC", "VCAL",
			   "MGNT", "MUON", "OQUA"};


void etaPhiPlot(TString fileName,
                std::string plot,
                bool drawLeg,
                bool ifEta,
                double maxEta,
                std::string tag) {
  TFile *hcalFile = new TFile(fileName);
  hcalFile->cd("materialBudgetVolumeAnalysis");
  setStyle();

  std::string xtit = "#eta";
  std::string ztit = "Eta";
  std::string ytit = "none";
  double xh = 0.90;
  if (plot == "radl") {
    ytit = "Material Budget (X_{0})";
  } else if (plot == "step") {
    ytit = "Material Budget (Step Length)";
    xh = 0.95;
  } else {
    plot = "intl";
    ytit = "Material Budget (#lambda)";
  }
  if (!ifEta) {
    xtit = "#phi";
    ztit = "Phi";
  }

  TLegend *leg = new TLegend(xh - 0.15, 0.60, xh, 0.90);
  leg->SetBorderSize(1);
  leg->SetFillColor(10);
  leg->SetMargin(0.25);
  leg->SetTextSize(0.028);

  char hname[20], titlex[50];
  sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[0].c_str());
  TProfile* prof;
  gDirectory->GetObject(hname, prof);
  int nb = prof->GetNbinsX();
  double xlow = prof->GetBinLowEdge(1);
  double xhigh = prof->GetBinLowEdge(nb) + prof->GetBinWidth(nb);
  THStack *hs = new THStack("hs", "");
  for (int ii = 0; ii < nlay; ++ii) {
    sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[ii].c_str());
    TProfile* prof;
    gDirectory->GetObject(hname, prof);
    sprintf(hname, "%s%s%sH", plot.c_str(), ztit.c_str(), names[ii].c_str());
    TH1D* hist = new TH1D(hname, "", nb, xlow, xhigh);
    for (int k = 1; k <= nb; ++k) {
      double cont = prof->GetBinContent(k);
      hist->SetBinContent(k, cont);
    }
    hist->SetLineColor(colorLay[ii]);
    hist->SetFillColor(colorLay[ii]);
    if (ifEta && maxEta > 0)
      hist->GetXaxis()->SetRangeUser(-maxEta, maxEta);
    hs->Add(hist);
    if (legends[ii] > 0) {
      sprintf(titlex, "%s", title[ii].c_str());
      leg->AddEntry(hist, titlex, "lf");
    }
  }

  std::string cname = "c_" + plot + ztit + tag;
  TCanvas *cc1 = new TCanvas(cname.c_str(), cname.c_str(), 700, 600);
  if (xh > 0.91) {
    cc1->SetLeftMargin(0.15);
    cc1->SetRightMargin(0.05);
  } else {
    cc1->SetLeftMargin(0.10);
    cc1->SetRightMargin(0.10);
  }

  hs->Draw("");
  if (drawLeg)
    leg->Draw("sames");
  hs->GetXaxis()->SetTitle(xtit.c_str());
  hs->GetYaxis()->SetTitle(ytit.c_str());
  if (xh > 0.91) {
    hs->GetYaxis()->SetTitleOffset(2.0);
  } else {
    hs->GetYaxis()->SetTitleOffset(1.2);
  }
  cc1->Modified();
}

void etaPhi2DPlot(TString fileName,
		  std::string plot,
		  bool drawLeg,
		  double maxEta,
		  std::string tag) {
  TFile *hcalFile = new TFile(fileName);
  hcalFile->cd("materialBudgetVolumeAnalysis");
  setStyle();

  std::string xtit = "#eta";
  std::string ytit = "#phi";
  std::string ztit = "none";
  if (plot == "radl") {
    ztit = "Material Budget (X_{0})";
  } else if (plot == "step") {
    ztit = "Material Budget (Step Length)";
  } else {
    plot = "intl";
    ztit = "Material Budget (#lambda)";
  }

  TLegend *leg = new TLegend(0.84, 0.69, 0.99, 0.99);
  leg->SetBorderSize(1);
  leg->SetFillColor(10);
  leg->SetMargin(0.25);
  leg->SetTextSize(0.028);

  char hname[20], titlex[50];
  sprintf(hname, "%sEtaPhi%s", plot.c_str(), names[0].c_str());
  TProfile2D* prof;
  gDirectory->GetObject(hname, prof);
  int nx = prof->GetXaxis()->GetNbins();
  double xlow = prof->GetXaxis()->GetBinLowEdge(1);
  double xhigh = prof->GetXaxis()->GetBinUpEdge(nx);
  int ny = prof->GetYaxis()->GetNbins();
  double ylow = prof->GetYaxis()->GetBinLowEdge(1);
  double yhigh = prof->GetYaxis()->GetBinUpEdge(ny);
  std::cout << hname << " X " << nx << ":" << xlow << ":" << xhigh << " Y " << ny << ":" << ylow << ":" << yhigh << std::endl;
  THStack *hs = new THStack("hs", ztit.c_str());
  for (int ii = 0; ii < nlay; ++ii) {
    sprintf(hname, "%sEtaPhi%s", plot.c_str(), names[ii].c_str());
    gDirectory->GetObject(hname, prof);
    sprintf(hname, "%sEtaPhi%sH", plot.c_str(), names[ii].c_str());
    TH2D* hist = new TH2D(hname, "", nx, xlow, xhigh, ny, ylow, yhigh);
    for (int kx = 1; kx <= nx; ++kx) {
      for (int ky = 1; ky <= ny; ++ky) {
	double cont = prof->GetBinContent(kx, ky);
	hist->SetBinContent(kx, ky, cont);
      }
    }
    hist->SetLineColor(colorLay[ii]);
    hist->SetFillColor(colorLay[ii]);
    if (maxEta > 0)
      hist->GetXaxis()->SetRangeUser(-maxEta, maxEta);
    hs->Add(hist);
    if (legends[ii] > 0) {
      sprintf(titlex, "%s", title[ii].c_str());
      leg->AddEntry(hist, titlex, "lf");
    }
  }

  std::string cname = "c_" + plot + "EtaPhi" + tag;
  TCanvas *cc1 = new TCanvas(cname.c_str(), cname.c_str(), 700, 600);
  cc1->SetLeftMargin(0.10);
  cc1->SetRightMargin(0.10);

  hs->Draw("");
  if (drawLeg)
    leg->Draw("sames");
  hs->GetXaxis()->SetTitle(xtit.c_str());
  hs->GetYaxis()->SetTitle(ytit.c_str());
  hs->GetYaxis()->SetTitleOffset(1.2);
  cc1->Modified();
}

void setStyle() {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);
  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);
  gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);
  gStyle->SetFrameLineWidth(1);
  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);
  gStyle->SetOptTitle(0);
  gStyle->SetTitleOffset(2.5, "Y");
}
