// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

#include "TCanvas.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TStyle.h"


int colorLayer[6]   = {  2,   7,   6,   3,   4,   1};
std::string dets[6] = {"BeamPipe", "Tracker", "EM Calorimeter",
		       "Hadron Calorimeter", "Muon System", "Forward Shield"};

void etaPhiPlot(TString fileName="matbdg_Calo.root", TString plot="IntLen", 
		int ifirst=0, int ilast=5, int drawLeg=1, bool ifEta=true,
		double maxEta=-1, bool debug=false);
void etaPhiDiff(TString fileName1="matbdg_Calo1.root", 
		TString fileName2="matbdg_Calo2.root", TString plot="IntLen", 
		int itype=2, int drawLeg=1, bool ifEta=true, double maxEta=-1);
void setStyle();

void etaPhiPlot(TString fileName, TString plot, int ifirst, int ilast, 
		int drawLeg, bool ifEta, double maxEta, bool debug) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  TString xtit = TString("#eta");
  TString ytit = "none";
  int ymin = 0, ymax = 20, istart = 200;
  double xh = 0.90;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("Material Budget (X_{0})");
    ymin = 0;  ymax = 200; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 300; xh = 0.70;
  } else {
    ytit = TString("Material Budget (#lambda)");
    ymin = 0;  ymax = 20; istart = 200;
  }
  if (!ifEta) {
    istart += 300;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xh-0.25, 0.75, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.018);

  int nplots=0;
  TProfile *prof[6];
  for (int ii=ilast; ii>=ifirst; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+ii);
    gDirectory->GetObject(hname,prof[nplots]);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    if (ifEta && maxEta > 0) 
      prof[nplots]->GetXaxis()->SetRangeUser(-maxEta,maxEta);
    if (xh < 0.8) 
      prof[nplots]->GetYaxis()->SetTitleOffset(1.7);
    sprintf(title, "%s", dets[ii].c_str());
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
    if (ii == ilast && debug) {
      int    nbinX = prof[0]->GetNbinsX();
      double xmin  = prof[0]->GetXaxis()->GetXmin();
      double xmax  = prof[0]->GetXaxis()->GetXmax();
      double dx    = (xmax - xmin)/nbinX;
      std::cout << "Hist " << ii;
      for (int ibx=0; ibx<nbinX; ibx++) {
	double xx1  = xmin + ibx*dx;
	double cont = prof[0]->GetBinContent(ibx+1);
	std::cout << " | " << ibx << "(" << xx1 << ":" << (xx1+dx) << ") " 
		  << cont;
      }
      std::cout << "\n";
    }
  }

  TString cname = "c_" + plot + xtit;
  TCanvas *cc1 = new TCanvas(cname, cname, 700, 600);
  if (xh < 0.8) {
    cc1->SetLeftMargin(0.15); cc1->SetRightMargin(0.05);
  }

  prof[0]->Draw("h");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("h sames");
  if (drawLeg > 0) leg->Draw("sames");
}

void etaPhiDiff(TString fileName1, TString fileName2, TString plot,
		int itype, int drawLeg, bool ifEta, double maxEta) {

  setStyle();

  TString xtit = TString("#eta");
  TString ytit = "none";
  double xh = 0.90, ymin = -0.5, ymax = 0.5; 
  int    ihist = 200 + itype;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("Material Budget Difference (X_{0})");
    ymin = -1;  ymax = 1; ihist = 100 + itype;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("Material Budget Difference (Step Length)");
    ymin = -20;  ymax = 20; ihist = 300 + itype; xh = 0.70;
  } else {
    ytit = TString("Material Budget Difference (#lambda)");
  }
  if (!ifEta) {
    ihist += 300;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xh-0.25, 0.84, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.022);

  TProfile *prof1, *prof2;
  char hname[10], title[50];
  sprintf(hname, "%i", ihist);
  TFile* file1 = new TFile(fileName1);
  file1->cd("g4SimHits");
  gDirectory->GetObject(hname,prof1);
  TFile* file2 = new TFile(fileName2);
  file2->cd("g4SimHits");
  gDirectory->GetObject(hname,prof2);
  TH1D *prof = (TH1D*) prof1->Clone();
  prof->Add(prof2,-1);
  prof->GetXaxis()->SetTitle(xtit);
  prof->GetYaxis()->SetTitle(ytit);
  prof->GetYaxis()->SetRangeUser(ymin, ymax);
  prof->SetLineColor(colorLayer[itype]);
  for (int k=1; k<=prof->GetNbinsX(); ++k) 
    prof->SetBinError(k,0);
  if (ifEta && maxEta > 0) 
    prof->GetXaxis()->SetRangeUser(-maxEta,maxEta);
  if (xh < 0.8) 
    prof->GetYaxis()->SetTitleOffset(1.7);
  sprintf(title, "%s", dets[itype].c_str());
  leg->AddEntry(prof, title, "lf");

  TString cname = "c_dif" + plot + xtit;
  TCanvas *cc1 = new TCanvas(cname, cname, 700, 600);
  if (xh < 0.8) {
    cc1->SetLeftMargin(0.15); cc1->SetRightMargin(0.05);
  }

  prof->Draw("h");
  if (drawLeg > 0) leg->Draw("sames");
}

void setStyle () {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptStat(0);          gStyle->SetLegendBorderSize(1);
  gStyle->SetOptTitle(0);         gStyle->SetTitleOffset(2.5,"Y");

}

