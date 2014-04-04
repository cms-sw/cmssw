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

void setStyle () {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptStat(0);          gStyle->SetLegendBorderSize(1);
  gStyle->SetOptTitle(0);         gStyle->SetTitleOffset(2.5,"Y");

}

