// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

#include "TCanvas.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TStyle.h"

const int nlaymax = 25;
const int nbinmax = 41;
double mean[nlaymax][nbinmax],  diff[nlaymax][nbinmax]; 

double towLow[41]    = { 0.000,  0.087,  0.174,  0.261,  0.348, 
			 0.435,  0.522,  0.609,  0.696,  0.783,
			 0.870,  0.957,  1.044,  1.131,  1.218,
			 1.305,  1.392,  1.479,  1.566,  1.653,
			 1.740,  1.830,  1.930,  2.043,  2.172, 
			 2.322,  2.500,  2.650,  2.853,  2.964, 
			 3.139,  3.314,  3.489,  3.664,  3.839, 
			 4.013,  4.191,  4.363,  4.538,  4.716, 
			 4.889};
double towHigh[41]   = { 0.087,  0.174,  0.261,  0.348,  0.435,
			 0.522,  0.609,  0.696,  0.783,  0.870,
			 0.957,  1.044,  1.131,  1.218,  1.305,
			 1.392,  1.479,  1.566,  1.653,  1.740,
			 1.830,  1.930,  2.043,  2.172,  2.322,
			 2.500,  2.650,  3.000,  2.964,  3.139,
			 3.314,  3.489,  3.664,  3.839,  4.013,
			 4.191,  4.363,  4.538,  4.716,  4.889,
			 5.191};

int colorLayer[25] = {  2,   7,   9,  30,  34,  38,  14,  40,  41,  42,
			45,  46,   8,  49,  37,  28,   4,   1,  48,  50,
		        3,   6,   5, 156, 159};

void etaPhiPlot(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		int ifirst=0, int ilast=21, int drawLeg=1, bool ifEta=true,
		double maxEta=-1, bool debug=true);
void etaPhiPlotHO(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		  int drawLeg=1, bool ifEta=true, double maxEta=-1);
void etaPhiPlotEC(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		  int drawLeg=1, bool ifEta=true, double maxEta=-1);
void etaPhiPlotHC(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		  int drawLeg=1, bool ifEta=true, double maxEta=-1);
void etaPhi2DPlot(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		  int ifirst=0, int ilast=19, int drawLeg=1);
void etaPhi2DPlot(int nslice, int kslice, TString fileName="matbdg_HCAL.root",
		  TString plot="IntLen", int ifirst=0, int ilast=21, 
		  int drawLeg=1);
void printTable (TString fileName="matbdg_HCAL.root", 
		 TString outputFileName="hcal.txt",
		 TString inputFileName="None");
void plotDiff (TString fileName="matbdg_HCAL.root", TString plot="IntLen");
void getDiff (TString fileName="matbdg_HCAL.root", TString plot="IntLen");
void plotHE(int flag=0, int logy=0, int save=0);
void etaPhiCastorPlot(TString fileName="matbdg_Castor.root", 
		      TString plot="IntLen", TString type="All",
		      bool etaPlus=true, int drawLeg=1, bool ifEta=true,
		      bool debug=true);
void efficiencyPlot(TString fileName="matbdg_HCAL.root", TString type="All",
		    bool ifEtaPhi=true, double maxEta=-1, bool debug=false);
void etaPhiFwdPlot(TString fileName="matbdg_Fwd.root", TString plot="IntLen", 
		   int first=0, int last=9, int drawLeg=1, bool debug=false);
void setStyle ();

void standardPlot (TString fileName="matbdg_HCAL.root", 
		   TString outputFileName="hcal.txt") {

  etaPhiPlot  (fileName, "IntLen", 0, 21, 1, true,  4.8);
  etaPhiPlot  (fileName, "IntLen", 0, 20, 1, false, -1.);
  etaPhiPlot  (fileName, "RadLen", 0, 21, 1, true,  4.8);
  etaPhiPlot  (fileName, "RadLen", 0, 20, 1, false, -1);
  etaPhiPlot  (fileName, "StepLen",0, 21, 1, true,  -1);
  etaPhiPlot  (fileName, "StepLen",0, 20, 1, false, -1);
  etaPhiPlotHO(fileName, "IntLen", 1, true, 2.5);
  etaPhiPlotEC(fileName, "IntLen", 1, true, 2.5);
  plotDiff    (fileName, "IntLen");
  plotDiff    (fileName, "RadLen");
  printTable  (fileName, outputFileName);
  etaPhi2DPlot(fileName, "IntLen", 0, 21, 1);
  etaPhi2DPlot(fileName, "RadLen", 0, 21, 1);
}

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
    ytit = TString("HCal Material Budget (X_{0})");
    ymin = 0;  ymax = 200; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("HCal Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 300; xh = 0.70;
  } else {
    ytit = TString("HCal Material Budget (#lambda)");
    ymin = 0;  ymax = 20; istart = 200;
  }
  if (!ifEta) {
    istart += 400;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xh-0.13, 0.60, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.018);

  int nplots=0;
  TProfile *prof[nlaymax];
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
    int lay = ii - 1;
    if (lay > 0 && lay < 20) {
      sprintf(title, "Layer %d", lay);
    } else if (lay == 0) {
      sprintf(title, "After Crystal");
    } else if (lay >= 20 ) {
      sprintf(title, "After HF");
    } else {
      sprintf(title, "Before Crystal");
    }
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
    if (ii == ilast && debug) {
      int    nbinX = prof[0]->GetNbinsX();
      double xmin  = prof[0]->GetXaxis()->GetXmin();
      double xmax  = prof[0]->GetXaxis()->GetXmax();
      double dx    = (xmax - xmin)/nbinX;
      cout << "Hist " << ii;
      for (int ibx=0; ibx<nbinX; ibx++) {
	double xx1  = xmin + ibx*dx;
	double cont = prof[0]->GetBinContent(ibx+1);
	cout << " | " << ibx << "(" << xx1 << ":" << (xx1+dx) << ") " << cont;
      }
      cout << "\n";
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

void etaPhiPlotHO(TString fileName, TString plot, int drawLeg, bool ifEta, 
		  double maxEta) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  int ihid[3] = {2, 18, 20};
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
    istart += 400;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xh-0.25, 0.60, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.45);
  leg->SetTextSize(0.04);
  int nplots=0;
  TProfile *prof[nlaymax];
  for (int ii=2; ii>=0; ii--) {
    char hname[10], title[50];
    int idpl = istart+ihid[ii];
    sprintf(hname, "%i", idpl);
    gDirectory->GetObject(hname,prof[nplots]);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetXaxis()->SetLabelSize(0.05);
    prof[nplots]->GetYaxis()->SetLabelSize(0.05);
    prof[nplots]->GetXaxis()->SetTitleSize(0.05);
    prof[nplots]->GetYaxis()->SetTitleSize(0.05);
    prof[nplots]->GetYaxis()->SetTitleOffset(0.8);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    if (xh < 0.8) 
      prof[nplots]->GetYaxis()->SetTitleOffset(1.7);
    if (ifEta && maxEta > 0) 
      prof[nplots]->GetXaxis()->SetRangeUser(0.0,maxEta);
    if (ii >= 2) {
      sprintf(title, "With HO");
    } else if (ii == 1) {
      sprintf(title, "Without HO");
    } else {
      sprintf(title, "Before HCAL");
    }
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
    if (ii == 1) {
      for (int kk=0; kk<7; kk++) {
	idpl--;
	sprintf(hname, "%i", idpl);
	gDirectory->GetObject(hname,prof[nplots]);
	prof[nplots]->GetXaxis()->SetTitle(xtit);
	prof[nplots]->GetYaxis()->SetTitle(ytit);
	prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
	prof[nplots]->SetLineColor(colorLayer[ii]);
	prof[nplots]->SetFillColor(colorLayer[ii]);
	if (ifEta && maxEta > 0) 
	  prof[nplots]->GetXaxis()->SetRangeUser(0.0,maxEta);
	nplots++;
      }
    }
  }

  TString cname = "c_HO" + plot + xtit;
  new TCanvas(cname, cname, 700, 400);

  prof[0]->Draw("h");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("h sames");
  if (drawLeg > 0) leg->Draw("sames");
}

void etaPhiPlotEC(TString fileName, TString plot, int drawLeg, bool ifEta,
		  double maxEta) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  int ihid[3] = {0, 1, 2};
  TString xtit = TString("#eta");
  TString ytit = "none";
  int ymin = 0, ymax = 7, istart = 200, ymax1 = 5;
  double xh = 0.90, xh1 = 0.90;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("Material Budget (X_{0})");
    ymin = 0;  ymax = 70; istart = 100; ymax1 = 50;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("Material Budget (Step Length)");
    ymin = 0;  ymax = 6000; istart = 300; xh = 0.35; ymax1 = 2500;
  } else {
    ytit = TString("Material Budget (#lambda)");
    ymin = 0;  ymax = 7; istart = 200;
  }
  if (!ifEta) {
    istart += 400;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xh-0.25, 0.60, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.30);
  leg->SetTextSize(0.04);
  int nplots=0;
  TProfile *prof[nlaymax];
  for (int ii=2; ii>=0; ii--) {
    char hname[10], title[50];
    int idpl = istart+ihid[ii];
    sprintf(hname, "%i", idpl);
    gDirectory->GetObject(hname,prof[nplots]);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetXaxis()->SetLabelSize(0.05);
    prof[nplots]->GetYaxis()->SetLabelSize(0.05);
    prof[nplots]->GetXaxis()->SetTitleSize(0.05);
    prof[nplots]->GetYaxis()->SetTitleSize(0.05);
    prof[nplots]->GetYaxis()->SetTitleOffset(0.8);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    if (xh < 0.8) 
      prof[nplots]->GetYaxis()->SetTitleOffset(1.05);
    if (ifEta && maxEta > 0) 
      prof[nplots]->GetXaxis()->SetRangeUser(0.0,maxEta);
    if (ii >= 2) {
      sprintf(title, "Front of HCAL");
    } else if (ii == 1) {
      sprintf(title, "After Crystals");
    } else {
      sprintf(title, "Before Crystals");
    }
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
  }

  TString cname = "c_EC1" + plot + xtit;
  new TCanvas(cname, cname, 700, 400);

  prof[0]->Draw("h");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("h sames");
  if (drawLeg > 0) leg->Draw("sames");

  double xmin  = prof[2]->GetXaxis()->GetXmin();
  double xmax  = prof[2]->GetXaxis()->GetXmax();
  int    nbins = prof[2]->GetNbinsX();
  TH1D *prof1 = new TH1D("Temp01", "Temp01", nbins, xmin, xmax);
  for (int ii=0; ii<nbins; ii++) {
    double x1 = prof[0]->GetBinLowEdge(ii+1);
    double x2 = prof[0]->GetBinWidth(ii+1);
    double v0 = prof[0]->GetBinContent(ii+1);
    double v1 = prof[1]->GetBinContent(ii+1);
    double v2 = prof[2]->GetBinContent(ii+1);
    double xx = x1+0.5*x2;
    double cont = v0 - v1;
    prof1->Fill(xx,cont);
    std::cout << "Bin " << ii << " Eta/Phi " << std::setw(4) << xx << " Material " << std::setw(6) << cont << "  " << std::setw(6) << (v0-v2) << "\n";
  }
  prof1->GetXaxis()->SetTitle(xtit);
  prof1->GetYaxis()->SetTitle(ytit);
  prof1->GetXaxis()->SetLabelSize(0.05);
  prof1->GetYaxis()->SetLabelSize(0.05);
  prof1->GetXaxis()->SetTitleSize(0.05);
  prof1->GetYaxis()->SetTitleSize(0.05);
  prof1->GetYaxis()->SetTitleOffset(0.8);
  prof1->GetYaxis()->SetRangeUser(ymin, ymax1);
  prof1->SetLineColor(colorLayer[3]);
  prof1->SetFillColor(colorLayer[3]);
  if (ifEta && maxEta > 0) prof1->GetXaxis()->SetRangeUser(0.0,maxEta);
  if (xh < 0.8)            prof1->GetYaxis()->SetTitleOffset(1.05);

  TLegend *mlg = new TLegend(xh1-0.3, 0.80, xh1, 0.90);
  char title[100];
  sprintf (title, "End crystal to Layer 0");
  mlg->SetBorderSize(1); mlg->SetFillColor(10); mlg->SetMargin(0.30);
  mlg->SetTextSize(0.04); mlg->AddEntry(prof1, title, "lf");

  cname        = "c_EC2" + plot + xtit;
  new TCanvas(cname, cname, 700, 400);
  prof1->Draw();
  if (drawLeg > 0) mlg->Draw("sames");
}

void etaPhiPlotHC(TString fileName, TString plot, int drawLeg, bool ifEta, 
		  double maxEta) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  int ihid[20] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  TString xtit = TString("#eta");
  TString ytit = "none";
  int ymin = 0, ymax = 20, istart = 200;
  double xh = 0.90;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("Material Budget (X_{0})");
    ymin = 0;  ymax = 200; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 300; xh = 0.35;
  } else {
    ytit = TString("Material Budget (#lambda)");
    ymin = 0;  ymax = 20; istart = 200;
  }
  if (!ifEta) {
    istart += 400;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xh-0.25, 0.70, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.30);
  leg->SetTextSize(0.04);
  int nplots=0;
  TProfile *prof[nlaymax];
  for (int ii=19; ii>=0; ii--) {
    char hname[10], title[50];
    int idpl = istart+ihid[ii];
    sprintf(hname, "%i", idpl);
    int icol = colorLayer[0];
    if (ii >= 18)    icol = colorLayer[2];
    else if (ii > 0) icol = colorLayer[1];
    gDirectory->GetObject(hname,prof[nplots]);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetXaxis()->SetLabelSize(0.05);
    prof[nplots]->GetYaxis()->SetLabelSize(0.05);
    prof[nplots]->GetXaxis()->SetTitleSize(0.05);
    prof[nplots]->GetYaxis()->SetTitleSize(0.05);
    prof[nplots]->GetYaxis()->SetTitleOffset(0.8);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(icol);
    prof[nplots]->SetFillColor(icol);
    if (xh < 0.8) 
      prof[nplots]->GetYaxis()->SetTitleOffset(1.05);
    if (ifEta && maxEta > 0) 
      prof[nplots]->GetXaxis()->SetRangeUser(0.0,maxEta);
    if (ii == 19) {
      sprintf(title, "End of HF");
      leg->AddEntry(prof[nplots], title, "lf");
    } else if (ii == 1) {
      sprintf(title, "After HB/HE/HO");
      leg->AddEntry(prof[nplots], title, "lf");
    } else if (ii == 0) {
      sprintf(title, "Before HCAL");
      leg->AddEntry(prof[nplots], title, "lf");
    }
    nplots++;
  }

  TString cname = "c_HC" + plot + xtit;
  new TCanvas(cname, cname, 700, 400);

  prof[0]->Draw("h");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("h sames");
  if (drawLeg > 0) leg->Draw("sames");
}
 
void etaPhi2DPlot(TString fileName, TString plot, int ifirst, int ilast, 
		  int drawLeg) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  TString xtit = TString("#eta");
  TString ytit = TString("#phi");
  TString ztit = TString("HCal Material Budget (#lambda)");
  int ymin = 0, ymax = 20, istart = 1000;
  double xh=0.95, yh=0.95;
  if (plot.CompareTo("RadLen") == 0) {
    ztit = TString("HCal Material Budget (X_{0})");
    ymin = 0;  ymax = 200; istart = 900;
  } else if (plot.CompareTo("StepLen") == 0) {
    ztit = TString("HCal Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 1100; 
  }
  
  TLegend *leg = new TLegend(xh-0.13, yh-0.30, xh, yh);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.018);

  int nplots=0;
  TProfile2D *prof[nlaymax];
  for (int ii=ilast; ii>=ifirst; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+ii);
    gDirectory->GetObject(hname,prof[nplots]);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetZaxis()->SetTitle(ztit);
    prof[nplots]->GetZaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    prof[nplots]->GetZaxis()->SetTitleOffset(1.4);
    int lay = ii - 1;
    if (lay > 0 && lay < 20) {
      sprintf(title, "Layer %d", lay);
    } else if (lay == 0) {
      sprintf(title, "After Crystal");
    } else if (lay >= 20 ) {
      sprintf(title, "After HF");
    } else {
      sprintf(title, "Before Crystal");
    }
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
  }

  TString cname = "c_" + plot + xtit + ytit;
  TCanvas *cc1 = new TCanvas(cname, cname, 700, 600);
  cc1->SetLeftMargin(0.15); cc1->SetRightMargin(0.05);

  prof[0]->Draw("lego fb bb");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("lego fb bb sames");
  if (drawLeg > 0) leg->Draw("sames");
}

void etaPhi2DPlot(int nslice, int kslice, TString fileName, TString plot,
		  int ifirst, int ilast, int drawLeg) {

  char hname[200], title[50];

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  TString xtit = TString("#eta");
  TString ytit;
  int ymin, ymax, istart;
  double xh=0.95, yh=0.90;
  char type[10];
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("HCal Material Budget (X_{0})");
    ymin = 0;  ymax = 200; istart = 900;
    sprintf (type, "Radlen");
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("HCal Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 1100; 
    sprintf (type, "Steplen");
  } else {
    ytit = TString("HCal Material Budget (#lambda)");
    ymin = 0;  ymax = 20;    istart = 1000;
    sprintf (type, "Intlen");
  }

  int nplots=0;
  TProfile2D *prof[nlaymax];
  for (int ii=ilast; ii>=ifirst; ii--) {
    sprintf(hname, "%i", istart+ii);
    gDirectory->GetObject(hname,prof[nplots]);
    nplots++;
  }
  
  double xmin  = prof[0]->GetXaxis()->GetXmin();
  double xmax  = prof[0]->GetXaxis()->GetXmax();
  int    nbinX = prof[0]->GetNbinsX();
  double ymin1 = prof[0]->GetYaxis()->GetXmin();
  double ymax1 = prof[0]->GetYaxis()->GetXmax();
  int    nbinY = prof[0]->GetNbinsY();
  int    ngroup= nbinY/nslice;
  double dy    = (ymax1-ymin1)/nbinY;
  cout << "X " << nbinX << "/" << xmin << "/" << xmax << " Slice " << nbinY << "/" << nslice << "/" << ngroup << " " << nplots << " " << dy << "\n";

  istart= 0;
  TLegend *leg[360];
  TH1D    *hist[nlaymax][360];
  for (int is=0; is<nslice; is++) {
    leg[is] = new TLegend(xh-0.13, yh-0.43, xh, yh);
    leg[is]->SetBorderSize(1); leg[is]->SetFillColor(10); 
    leg[is]->SetMargin(0.25);  leg[is]->SetTextSize(0.023);
    double y1 = (ymin1 + istart*dy)*180./3.1415926;
    double y2 = y1 + ngroup*dy*180./3.1415926;
    if (y1 < 0) {
      y1 += 360.;
      y2 += 360.;
    }
    sprintf (title, "#phi = %6.1f :%6.1f", y1, y2);
    leg[is]->SetHeader(title);

    for (int ii=0; ii<nplots; ii++) {
      sprintf(hname, "Hist%iSlice%i", ii, is);
      hist[ii][is] = new TH1D(hname, hname, nbinX, xmin, xmax);
      //      cout << "Hist " << ii;
      for (int ibx=0; ibx<nbinX; ibx++) {
	double contb = 0;
	//	cout << " / " << ibx;
	for (int iby=0; iby<ngroup; iby++) {
	  int    ibin = iby+istart;
	  double cont = prof[ii]->GetBinContent(ibx+1, ibin+1);
	  //	  cout << "/" << ibin << " " << cont;
	  contb += cont;
	}
	contb /= ngroup;
	//	cout << " " << contb;
	hist[ii][is]->SetBinContent(ibx+1, contb);
      }
      //      cout << "\n";
      hist[ii][is]->GetXaxis()->SetTitle(xtit);
      hist[ii][is]->GetYaxis()->SetTitle(ytit);
      hist[ii][is]->GetYaxis()->SetRangeUser(ymin, ymax);
      hist[ii][is]->SetLineColor(colorLayer[ilast-ii]);
      hist[ii][is]->SetFillColor(colorLayer[ilast-ii]);
      hist[ii][is]->GetYaxis()->SetTitleOffset(0.8);
      int lay = ilast - ii - 1;
      if (lay > 0 && lay < 20) {
	sprintf(title, "Layer %d", lay);
      } else if (lay == 0) {
	sprintf(title, "After Crystal");
      } else if (lay >= 20 ) {
	sprintf(title, "After HF");
      } else {
	sprintf(title, "Before Crystal");
      }
      leg[is]->AddEntry(hist[ii][is], title, "lf");
    }
    istart += ngroup;
  }

  cout << "All histograms created now plot\n";
  TCanvas *cc1[360];
  int ismin=0, ismax=nslice;
  if (kslice >=0 && kslice <= nslice) {
    ismin = kslice;
    ismax = ismin+1;
  }
  for (int is=ismin; is<ismax; is++) {
    sprintf (hname, "c_%s%i", type, is);
    cc1[is] = new TCanvas(hname, hname, 700, 400);
    cc1[is]->SetLeftMargin(0.15); cc1[is]->SetRightMargin(0.05);
    hist[0][is]->Draw();
    for(int i=1; i<nplots; i++)
      hist[i][is]->Draw("sames");
    if (drawLeg > 0) leg[is]->Draw("sames");
  }
}

void printTable (TString fileName, TString outputFileName,
		 TString inputFileName) {

  double radl[nlaymax][nbinmax],  intl[nlaymax][nbinmax]; 
  bool compare = false;
  if (inputFileName != "None") {
    ifstream inp(inputFileName, ios::in);
    cout << "Opens " << inputFileName << "\n";
    if (inp) {
      TString line;
      int     tower;
      double  eta;
      for (int i = 0; i < 23; i++) 
	inp >> line;
      for (int itow=0; itow<nbinmax; itow++) {
	inp >> tower >> eta;
	int laymax=19;
	if (itow > 29)     laymax = 2;
	else if (itow > 3) laymax = 18;
	for (int ilay=0; ilay<laymax; ilay++)
	  inp >> intl[ilay][tower];
      }
      for (int i = 0; i < 23; i++) 
	inp >> line;
      for (int itow=0; itow<nbinmax; itow++) {
	inp >> tower >> eta;
	int laymax=19;
	if (itow > 29)     laymax = 2;
	else if (itow > 3) laymax = 18;
	for (int ilay=0; ilay<laymax; ilay++)
	  inp >> radl[ilay][tower];
      }
      compare = true;
      inp.close();
    }
  }
  std::ofstream os;
  os.open(outputFileName);

  int nbadI=0;
  getDiff (fileName, "IntLen");
  os << "Interaction Length\n" << "==================\n"
     << "Eta Tower/Layer   0      1       2       3       4       5     "
     << "  6       7       8       9     10      11      12      13     "
     << " 14      15      16      17\n";
  for (int itow=0; itow<nbinmax; itow++) {
    os << setw(3)<< itow << setw(7) << setprecision(3) 
       << 0.5*(towLow[itow]+towHigh[itow]);
    int laymax=19, ioff=1;
    if (itow > 29)     {laymax = 2;  ioff=0;}
    else if (itow > 3) {laymax = 18; ioff=1;}
    for (int ilay=0; ilay<laymax; ilay++) {
      os << setw(8) << setprecision(4) <<  diff[ilay+ioff][itow];
      if (compare) {
	double num = (diff[ilay+ioff][itow] - intl[ilay][itow]);
	double den = (diff[ilay+ioff][itow] + intl[ilay][itow]);
	double dd  = (den == 0.? 0. : 2.0*num/den);
	if (dd > 0.01) {
	  nbadI++;
	  cout << "Lambda::Tower " << setw(3) << itow << " Layer " << setw(3) 
	       << ilay << " Old" << setw(8) << setprecision(4) 
	       << intl[ilay][itow] << " New" << setw(8) << setprecision(4) 
	       << diff[ilay+ioff][itow] << " Diff"<< setw(8) << setprecision(4)
	       << dd << "\n";
	}
      }
    }
    os << "\n";
  }

  int nbadR = 0;
  getDiff (fileName, "RadLen");
  os << "\n\nRadiation Length\n" << "================\n"
     << "Eta Tower/Layer   0      1       2       3       4       5     "
     << "  6       7       8       9     10      11      12      13     "
     << " 14      15      16      17\n";
  for (int itow=0; itow<nbinmax; itow++) {
    os << setw(3)<< itow << setw(7) << setprecision(3) 
       << 0.5*(towLow[itow]+towHigh[itow]);
    int laymax=19, ioff=1;
    if (itow > 29)     {laymax = 2;  ioff=0;}
    else if (itow > 3) {laymax = 18; ioff=1;}
    for (int ilay=0; ilay<laymax; ilay++) {
      os << setw(8) << setprecision(4) <<  diff[ilay+ioff][itow];
      if (compare) {
	double num = (diff[ilay+ioff][itow] - radl[ilay][itow]);
	double den = (diff[ilay+ioff][itow] + radl[ilay][itow]);
	double dd  = (den == 0.? 0. : 2.0*num/den);
	if (dd > 0.01) {
	  nbadR++;
	  cout << "X0::Tower " << setw(3) << itow << " Layer " << setw(3) 
	       << ilay << " Old" << setw(8) << setprecision(4) 
	       << radl[ilay][itow] << " New" << setw(8) << setprecision(4) 
	       << diff[ilay+ioff][itow] << " Diff"<< setw(8) << setprecision(4)
	       << dd << "\n";
	}
      }
    }
    os << "\n";
  }
  os.close();

  cout << "Comparison Results " << nbadI << " discrepancies for Lambda and "
       << nbadR << " discrepancies for X0\n";
}

void plotDiff (TString fileName, TString plot) {

  setStyle();
  gStyle->SetTitleOffset(1.0,"Y");
  getDiff (fileName, plot);
  TString xtit = TString("Layer Number");
  TString ytit = TString("HCal Material Budget (#lambda)");
  if (plot.CompareTo("RadLen") == 0) 
    ytit = TString("HCal Material Budget (X_{0})");

  TMultiGraph *mg = new TMultiGraph();
  TLegend *leg_mg = new TLegend(.5,.5,.75,.80);
  leg_mg->SetFillColor(10);  leg_mg->SetBorderSize(1);
  leg_mg->SetMargin(0.25);   leg_mg->SetTextSize(0.04);
  leg_mg->SetHeader(ytit);

  double diff_lay[19],  idx[19];
  for (int ilay=1; ilay<20; ilay++) {
    diff_lay[ilay-1] = diff[ilay][0];
    idx[ilay-1] = ilay-1;
  }
  TGraph *gr_eta1 = new TGraph(19, idx, diff_lay);
  gr_eta1->SetMarkerStyle(20);
  gr_eta1->SetMarkerColor(2);
  gr_eta1->SetLineColor(2);
  gr_eta1->GetXaxis()->SetTitle(xtit);
  gr_eta1->GetYaxis()->SetTitle(ytit);
  mg->Add(gr_eta1,  "pc");
  leg_mg->AddEntry(gr_eta1, "HB #eta = 1");

  for (int ilay=1; ilay<20; ilay++) 
    diff_lay[ilay-1] = diff[ilay][6];
  TGraph *gr_eta7 = new TGraph(18, idx, diff_lay);
  gr_eta7->SetMarkerStyle(22);
  gr_eta7->SetMarkerColor(4);
  gr_eta7->SetLineColor(4);
  mg->Add(gr_eta7,  "pc");
  gr_eta7->GetXaxis()->SetTitle(xtit);
  gr_eta7->GetYaxis()->SetTitle(ytit);
  leg_mg->AddEntry(gr_eta7, "HB #eta = 7");
  
  for (int ilay=1; ilay<20; ilay++) 
    diff_lay[ilay-1] = diff[ilay][12];
  TGraph *gr_eta13 = new TGraph(18, idx, diff_lay);
  gr_eta13->SetMarkerStyle(29);
  gr_eta13->SetMarkerColor(kGreen);
  gr_eta13->SetLineColor(kGreen);
  gr_eta13->GetXaxis()->SetTitle(xtit);
  gr_eta13->GetYaxis()->SetTitle(ytit);
  mg->Add(gr_eta13, "pc");
  leg_mg->AddEntry(gr_eta13,"HB #eta = 13");

  for (int ilay=1; ilay<20; ilay++) 
    diff_lay[ilay-1] = diff[ilay][19];
  TGraph *gr_eta19 = new TGraph(18, idx, diff_lay);
  gr_eta19->SetMarkerStyle(24);
  gr_eta19->SetMarkerColor(kCyan);
  gr_eta19->SetLineColor(kCyan);
  gr_eta19->GetXaxis()->SetTitle(xtit);
  gr_eta19->GetYaxis()->SetTitle(ytit);
  mg->Add(gr_eta19, "pc");
  leg_mg->AddEntry(gr_eta19,"HE #eta = 20");

  for(int ilay=1; ilay<20; ilay++) 
    diff_lay[ilay-1] = diff[ilay][25];
  TGraph *gr_eta25 = new TGraph(18, idx, diff_lay);
  gr_eta25->SetMarkerStyle(26);
  gr_eta25->SetMarkerColor(kCyan);
  gr_eta25->SetLineColor(kCyan);
  gr_eta25->GetXaxis()->SetTitle(xtit);
  gr_eta25->GetYaxis()->SetTitle(ytit);
  mg->Add(gr_eta25, "pc");
  leg_mg->AddEntry(gr_eta25,"HE #eta = 26");

  TString cname = "c_diff_" + plot;
  new TCanvas(cname, cname, 700, 400);
  mg->Draw("a");
  mg->GetXaxis()->SetTitle(xtit);
  mg->GetYaxis()->SetTitle(ytit);
  leg_mg->Draw("same");
}

void getDiff (TString fileName, TString plot) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");

  int    istart = 200;
  if (plot.CompareTo("RadLen") == 0) {
    istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    istart = 300; 
  }

  for (int ilay=0; ilay<22; ilay++) {
    char hname[10];
    sprintf(hname, "%i", istart+ilay+1);
    TProfile *prof;
    gDirectory->GetObject(hname,prof);
    int      nbins = prof->GetNbinsX();
    for (int itow=0; itow<nbinmax; itow++) {
      double ent = 0, value = 0;
      for (int ii=0; ii<nbins; ii++) {
	double xl = prof->GetBinLowEdge(ii+1);
	double xu = prof->GetBinWidth(ii+1);
	if (xl >= 0) { xu += xl;}
	else         { double tmp = xu; xu =-xl; xl = xu-tmp;}
	double cont = (prof->GetBinContent(ii+1));
	double dx   = 1;
	if (cont > 0) {
	  if (xl >= towLow[itow] && xu <= towHigh[itow]) {
	    ent += dx; value += cont;
	  } else if (xl < towLow[itow] && xu > towLow[itow]) {
	    dx   = (xu-towLow[itow])/(xu-xl);
	    ent += dx; value += dx*cont;
	  } else if (xu > towHigh[itow] && xl < towHigh[itow]) {
	    dx   = (towHigh[itow]-xl)/(xu-xl);
	    ent += dx; value += dx*cont;
	  }
	}
      }
      if (ent > 0) mean[ilay][itow] = value/ent;
      else         mean[ilay][itow] = 0.;
    }
  }

  for (int itow=30; itow<nbinmax; itow++) {
    mean[0][itow] = mean[19][itow];
    mean[1][itow] = mean[20][itow];
    mean[19][itow] = 0;
    mean[20][itow] = 0;
  }

  mean[0][15] = 0.5*(mean[0][14] + mean[0][17]);
  mean[1][15] = 0.5*(mean[1][14] + mean[1][17]);
  mean[2][15] = 0.5*(mean[2][14] + mean[2][17]);
  /*
  for (int itow=0; itow<nbinmax; itow++) {
    std::cout << "Tower " << itow;
    for (int ilay=0; ilay<22; ilay++) 
      cout << " " << ilay << " " << mean[ilay][itow];
    std::cout << "\n";
  }
  */
  for (int itow=0; itow<30; itow++) {
    for (int ilay=20; ilay>0; ilay--) {
      if (mean[ilay-1][itow] <= 0) {
	mean[ilay-1][itow] = mean[ilay][itow];
      }
    }
  }

  for (int itow=0; itow<nbinmax; itow++) {
    if (itow > 4 && itow < 26) mean[19][itow] = 0;
    diff[0][itow] = mean[0][itow];
  }

  for (int itow=15; itow<17; itow++) {
    for (int ilay=1; ilay<22; ilay++) {
      if (mean[ilay][itow] > mean[ilay+1][itow]) {
	for (int jlay=ilay+1; jlay<22; jlay++)
	  mean[jlay][itow] = 0;
	break;
      }
    }
  }

  for (int ilay=1; ilay<22; ilay++) {
    for (int itow=0; itow<nbinmax; itow++) {
      diff[ilay][itow] = mean[ilay][itow]-mean[ilay-1][itow];
      if (diff[ilay][itow] < 0) diff[ilay][itow] = 0;
    }
  }
  /*  
  for (int itow=0; itow<nbinmax; itow++) {
    std::cout << "Tower " << itow;
    for (int ilay=0; ilay<22; ilay++) {
      cout << " " << ilay << " " << mean[ilay][itow] << " " << diff[ilay][itow];
    }
    std::cout << "\n";
  }
  */
}

void plotHE(int flag, int logy, int save) { 

  double angle[31] = {-2.5,-2.25,-2.00,-1.75,-1.50,-1.25,-1.00,-0.75,-0.50,
		      -0.25,-0.20,-0.15,-0.10,-0.05,-0.025,0,0.025,0.05,0.10,
		      0.15,0.20,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,
		      2.50};
  double lAir[31]  = {11440,11440,11440,11440,11440,11440,11440,11440,11440,
		      11440,11440,11440,11440,11440,11440,11569,11569,11440,
		      11440,11440,11440,11440,11440,11440,11440,11440,11440,
		      11440,11440,11440,11440};
  double xAir[31]  = {0.037941,0.037941,0.037941,0.037941,0.037941,0.037941,
		      0.037941,0.037941,0.037941,0.037941,0.037941,0.037941,
		      0.037941,0.037941,0.037941,0.038369,0.038369,0.037941,
		      0.037941,0.037941,0.037941,0.037941,0.037941,0.037941,
		      0.037941,0.037941,0.037941,0.037941,0.037941,0.037941,
		      0.037941};
  double iAir[31]  = {0.0162481,0.0162481,0.0162481,0.0162481,0.0162481,
		      0.0162481,0.0162481,0.0162481,0.0162481,0.0162481,
		      0.0162481,0.0162481,0.0162481,0.0162481,0.0162481,
		      0.0164314,0.0164314,0.0162481,0.0162481,0.0162481,
		      0.0162481,0.0162481,0.0162481,0.0162481,0.0162481,
		      0.0162481,0.0162481,0.0162481,0.0162481,0.0162481,
		      0.0162481};
  double lPol[31]  = {60.8381,60.8381,60.8381,60.8381,60.8381,60.8381,60.8381,
		      60.8381,60.8381,60.8381,60.8381,60.8381,60.8381,60.8381,
		      60.8381,0.00000,0.00000,60.8381,60.8381,60.8381,60.8381,
		      60.8381,60.8381,60.8381,60.8381,60.8381,60.8381,60.8381,
		      60.8381,60.8381,60.8381};
  double xPol[31]  = {0.129083,0.129083,0.129083,0.129083,0.129083,0.129083,
		      0.129083,0.129083,0.129083,0.129083,0.129083,0.129083,
		      0.129083,0.129083,0.129083,0.000000,0.000000,0.129083,
		      0.129083,0.129083,0.129083,0.129083,0.129083,0.129083,
		      0.129083,0.129083,0.129083,0.129083,0.129083,0.129083,
		      0.129083};
  double iPol[31]  = {0.0854129,0.0854129,0.0854129,0.0854129,0.0854129,
		      0.0854129,0.0854129,0.0854129,0.0854129,0.0854129,
		      0.0854129,0.0854129,0.0854129,0.0854129,0.0854129,
		      0.0000000,0.0000000,0.0854129,0.0854129,0.0854129,
		      0.0854129,0.0854129,0.0854129,0.0854129,0.0854129,
		      0.0854129,0.0854129,0.0854129,0.0854129,0.0854129,
		      0.0854129};
  double lScn[31]  = {68.2125,68.2125,68.2125,68.2125,68.2125,68.2125,68.2125,
		      68.2125,68.2125,68.2125,68.2125,68.2125,68.2125,68.2125,
		      68.2125,0.00000,0.00000,68.2125,68.2125,68.2125,68.2125,
		      68.2125,68.2125,68.2125,68.2125,68.2125,68.2125,68.2125,
		      68.2125,68.2125,68.2125};
  double xScn[31]  = {0.160352,0.160352,0.160352,0.160352,0.160352,0.160352,
		      0.160352,0.160352,0.160352,0.160352,0.160352,0.160352,
		      0.160352,0.160352,0.160352,0.000000,0.000000,0.160352,
		      0.160352,0.160352,0.160352,0.160352,0.160352,0.160352,
		      0.160352,0.160352,0.160352,0.160352,0.160352,0.160352,
		      0.160352};
  double iScn[31]  = {0.0973967,0.0973967,0.0973967,0.0973967,0.0973967,
		      0.0973967,0.0973967,0.0973967,0.0973967,0.0973967,
		      0.0973967,0.0973967,0.0973967,0.0973967,0.0973967,
		      0.0000000,0.0000000,0.0973967,0.0973967,0.0973967,
		      0.0973967,0.0973967,0.0973967,0.0973967,0.0973967,
		      0.0973967,0.0973967,0.0973967,0.0973967,0.0973967,
		      0.0973967};
  double lBra[31]  = {1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,
		      1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,
		      1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,
		      1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,1444.41,
		      1444.41,1444.41,1444.41};
  double xBra[31]  = {96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,
		      96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,
		      96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,
		      96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,96.7848,
		      96.7848,96.7848,96.7848};
  double iBra[31]  = {8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,
		      8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,
		      8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,
		      8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,8.79637,
		      8.79637,8.79637,8.79637};
  std::string nameMat[4]   = {"Air", "Polythene", "Scintillator", "Brass"};
  int         colMat[4]    = {1, 2, 6, 4};
  int         symbMat[4]   = {24, 29, 25, 27};
  
  setStyle(); gStyle->SetTitleOffset(1.2,"Y");
  char name[30], title[60], gname[12];
  TGraph *gr[4];
  int kfirst = 3;
  double ymi=0, ymx=100;
  if (flag < 0) {
    sprintf (name, "Step Length");
    sprintf (title, "Step Length (mm)");
    sprintf (gname, "stepLength");
    gr[0] = new TGraph(31, angle, lAir);
    gr[1] = new TGraph(31, angle, lPol);
    gr[2] = new TGraph(31, angle, lScn);
    gr[3] = new TGraph(31, angle, lBra);
    kfirst = 0;
    if (logy == 0) {
      ymx = 12000;
    } else {
      ymi = 10.0; ymx = 20000;
    }
  } else if (flag>0) {
    sprintf (name, "# Interaction Length");
    sprintf (title, "Number of Interaction Length");
    sprintf (gname, "intLength");
    gr[0] = new TGraph(31, angle, iAir);
    gr[1] = new TGraph(31, angle, iPol);
    gr[2] = new TGraph(31, angle, iScn);
    gr[3] = new TGraph(31, angle, iBra);
    if (logy == 0) {
      ymx = 10;
    } else {
      ymi = 0.01; ymx = 20;
    }
  } else {
    sprintf (name, "# Radiation Length");
    sprintf (title, "Number of Radiation Length");
    sprintf (gname, "radLength");
    gr[0] = new TGraph(31, angle, xAir);
    gr[1] = new TGraph(31, angle, xPol);
    gr[2] = new TGraph(31, angle, xScn);
    gr[3] = new TGraph(31, angle, xBra);
    if (logy == 0) {
      ymx = 100;
    } else {
      ymi = 0.01; ymx = 200;
    }
  }
  gr[kfirst]->GetXaxis()->SetTitle("#phi ( ^{o})");
  gr[kfirst]->GetYaxis()->SetTitle(title);
  gr[kfirst]->SetTitle("");
  for (int i=0; i<4; i++) {
    int icol = colMat[i];
    int type = symbMat[i];
    gr[i]->SetMarkerSize(1.2);    gr[i]->SetLineColor(icol);
    gr[i]->SetLineStyle(i+1);     gr[i]->SetLineWidth(2);
    gr[i]->SetMarkerColor(icol);  gr[i]->SetMarkerStyle(type);
    gr[i]->GetYaxis()->SetRangeUser(ymi,ymx);
    gr[i]->GetXaxis()->SetRangeUser(-3.0,+3.0);
  }

  TCanvas *c1 = new TCanvas("c1", name, 800, 500);
  if (logy != 0) gPad->SetLogy(1); 
  gr[kfirst]->Draw("alp");
  for (int i=0; i<4; i++) {
    if (i != kfirst) gr[i]->Draw("lp");
  }

  double ylow = 0.4;
  char list[20];
  TLegend *leg1 = new TLegend(0.60,ylow,0.90,ylow+0.2);
  for (int i=0; i<4; i++) {
    sprintf (list, "%s", nameMat[i].c_str());
    leg1->AddEntry(gr[i],list,"LP");
  }
  leg1->SetHeader(name); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

  if (save != 0) {
    char fname[20];
    if (save > 0) sprintf (fname, "%s.eps", gname);
    else          sprintf (fname, "%s.gif", gname);
    c1->SaveAs(fname);
  }


}

void etaPhiCastorPlot(TString fileName, TString plot, TString type,
		      bool etaPlus, int drawLeg, bool ifEta, bool debug) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();
  if (debug) std::cout << fileName << " is opened at " << hcalFile << std::endl;

  TString xtit = TString("#eta");
  char ytit[80], ytpart[10];
  int ymin = 0, ymax = 20, istart = 200, ifirst=0;
  double xh = 0.90;
  if (!etaPlus) ifirst = 10;
  if (type.CompareTo("EC") == 0) {
    sprintf (ytpart, "(EC)"); ifirst += 2;
  } else if (type.CompareTo("HC") == 0) {
    sprintf (ytpart, "(HC)"); ifirst += 4;
  } else if (type.CompareTo("ED") == 0) {
    sprintf (ytpart, "(Dead EC)"); ifirst += 6;
  } else if (type.CompareTo("HD") == 0) {
    sprintf (ytpart, "(Dead HC)"); ifirst += 8;
  } else {
    sprintf (ytpart, "(All)");
  }
  if (debug) std::cout << type << " Gives " << ifirst << " Title " << ytpart << std::endl;
  if (plot.CompareTo("RadLen") == 0) {
    sprintf(ytit, "Castor %s Material Budget (X_{0})", ytpart);
    ymin = 0;  ymax = 200; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    sprintf(ytit, "Castor %s Material Budget (Step Length)", ytpart);
    ymin = 0;  ymax = 15000; istart = 300; xh = 0.70;
  } else {
    sprintf(ytit, "Castor %s Material Budget (#lambda)", ytpart);
    ymin = 0;  ymax = 20; istart = 200;
  }
  if (!ifEta) {
    istart += 400;
    xtit    = TString("#phi"); 
  }
  if (debug) std::cout << "Title (x) " << xtit << " (y) " << ytit << " First " << ifirst << ":" << istart << std::endl;
  
  TLegend *leg = new TLegend(xh-0.13, 0.80, xh, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.025);

  int nplots=0;
  TProfile *prof[2];
  for (int ii=ifirst+1; ii>=ifirst; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+ii);
    if (debug) std::cout << "[" << nplots << "] " << ii << " " << hname << "\n";
    gDirectory->GetObject(hname,prof[nplots]);
    if (debug) std::cout << "Histogram[" << nplots << "] : " << hname << " at " << prof[nplots] << std::endl;
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[nplots]);
    prof[nplots]->SetFillColor(colorLayer[nplots]);
    if (xh < 0.8) 
      prof[nplots]->GetYaxis()->SetTitleOffset(1.7);
    if (ii == ifirst) {
      sprintf(title, "Front");
    } else {
      sprintf(title, "End");
    }
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
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

void efficiencyPlot(TString fileName, TString type, bool ifEtaPhi, 
		    double maxEta, bool debug) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  int id0=1300, idpl1=8, idpl2=0;
  char hname[100], title[100];
  if      (type.CompareTo("HB") == 0) {
    idpl1 = 1; idpl2 =2; sprintf (title, "Efficiency for HB");
  } else if (type.CompareTo("HE") == 0) {
    idpl1 = 3; idpl2 = 4; sprintf (title, "Efficiency for HE");
  } else if (type.CompareTo("HO") == 0) {
    idpl1 = 5; idpl2 = 6; sprintf (title, "Efficiency for HO");
  } else if (type.CompareTo("HF") == 0) {
    idpl1 = 7; sprintf (title, "Efficiency for HF");
  } else {
    sprintf (title, "Efficiency for HCAL");
  }
  TLegend *leg = new TLegend(0.70, 0.82, 0.90, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.03);

  if (ifEtaPhi) {
    id0 = 1400;
    TH2F *hist0, *hist1, *hist2;
    sprintf(hname, "%i", id0);
    gDirectory->GetObject(hname, hist0);
    sprintf(hname, "%i", id0+idpl1);
    gDirectory->GetObject(hname, hist1);
    if (idpl2 > 0) {
      sprintf(hname, "%i", id0+idpl2);
      gDirectory->GetObject(hname, hist2);
    } else {
      hist2 = 0;
    }
    if (debug) std::cout << "Get Histos at " <<hist0 << " and " <<hist1 <<"\n";
    if (hist0 && hist1) {
      double xmin  = hist0->GetXaxis()->GetXmin();
      double xmax  = hist0->GetXaxis()->GetXmax();
      int    nbinX = hist0->GetNbinsX();
      double ymin  = hist0->GetYaxis()->GetXmin();
      double ymax  = hist0->GetYaxis()->GetXmax();
      int    nbinY = hist0->GetNbinsY();
      if (debug) std::cout <<"NbinX " <<nbinX <<" range "<<std::setprecision(5)
			   <<xmin <<":" <<std::setprecision(5) <<xmax << " "
			   <<"NbinY " <<nbinY <<" range "<<std::setprecision(5)
			   <<ymin <<":" <<std::setprecision(5) <<ymax <<"\n";
      TH2D *hist = new TH2D("hist", title, nbinX,xmin,xmax, nbinY,ymin,ymax);
      TH2D *histe= 0;
      if (hist2) histe = new TH2D("histe", title, nbinX,xmin,xmax, nbinY,ymin,ymax);
      for (int ibx=0; ibx<nbinX; ++ibx) {
	for (int iby=0; iby<nbinY; ++iby) {
	  double contN = hist1->GetBinContent(ibx+1,iby+1);
	  double contD = hist0->GetBinContent(ibx+1,iby+1);
	  double cont  = contN/std::max(contD,1.0);
	  hist->SetBinContent(ibx+1, iby+1, cont);
	  if (hist2) {
	    contN = hist2->GetBinContent(ibx+1,iby+1);
	    cont  = contN/std::max(contD,1.0);
	    histe->SetBinContent(ibx+1, iby+1, cont);
	  }
	}
      }
      hist->GetXaxis()->SetTitle("#eta");hist->GetYaxis()->SetTitle("#phi"); 
      hist->GetZaxis()->SetTitle(title);hist->GetZaxis()->SetTitleOffset(.8);
      new TCanvas(title, title, 700, 400);
      hist->SetLineColor(2); hist->SetLineStyle(1); hist->SetLineWidth(1);
      if (maxEta > 0) hist->GetXaxis()->SetRangeUser(-maxEta,maxEta);
      hist->Draw("lego fb bb"); leg->AddEntry(hist, "At least 1 layer", "l");
      if (histe) {
	histe->SetLineColor(4);histe->SetLineStyle(2);histe->SetLineWidth(1);
	if (maxEta > 0) histe->GetXaxis()->SetRangeUser(-maxEta,maxEta);
	histe->Draw("lego fb bb sames");leg->AddEntry(histe,"All layers","l");
      }
      leg->Draw("sames");
    }
  } else {
    TH1F *hist0, *hist1, *hist2;
    sprintf(hname, "%i", id0);
    gDirectory->GetObject(hname, hist0);
    sprintf(hname, "%i", id0+idpl1);
    gDirectory->GetObject(hname, hist1);
    if (idpl2 > 0) {
      sprintf(hname, "%i", id0+idpl2);
      gDirectory->GetObject(hname, hist2);
    } else {
      hist2 = 0;
    }
    if (debug) std::cout << "Get Histos at " <<hist0 << " and " <<hist1 <<"\n";
    if (hist0 && hist1) {
      double xmin  = hist0->GetXaxis()->GetXmin();
      double xmax  = hist0->GetXaxis()->GetXmax();
      int    nbinX = hist0->GetNbinsX();
      if (debug) std::cout <<"Nbin " <<nbinX <<" range " <<std::setprecision(5)
			   <<xmin <<":" <<std::setprecision(5) <<xmax <<"\n";
      TH1D *hist = new TH1D("hist", title, nbinX, xmin, xmax);
      TH1D *histe= 0;
      if (hist2) histe = new TH1D("histe", title, nbinX, xmin, xmax);
      for (int ib=0; ib<nbinX; ++ib) {
	double contN = hist1->GetBinContent(ib+1);
	double contD = hist0->GetBinContent(ib+1);
	double cont  = contN/std::max(contD,1.0);
	hist->SetBinContent(ib+1, cont);
	/*
	double eror  = std::sqrt(contN)/std::max(contD,1.0);
	hist->SetBinError(ib+1, eror);
	*/
	if (hist2) {
	  contN = hist2->GetBinContent(ib+1);
	  cont  = contN/std::max(contD,1.0);
	  histe->SetBinContent(ib+1, cont);
	}
      }
      hist->GetXaxis()->SetTitle("#eta");
      hist->GetYaxis()->SetTitle(title);
      hist->GetYaxis()->SetTitleOffset(0.8);
      new TCanvas(title, title, 700, 400);
      hist->SetLineColor(2); hist->SetLineStyle(1); hist->SetLineWidth(1);
      if (maxEta > 0) hist->GetXaxis()->SetRangeUser(-maxEta,maxEta);
      hist->Draw(); leg->AddEntry(hist, "At least 1 layer", "l");
      if (histe) {
	histe->SetLineColor(4); histe->SetLineStyle(2); histe->SetLineWidth(1);
	if (maxEta > 0) histe->GetXaxis()->SetRangeUser(-maxEta,maxEta);
	histe->Draw("sames"); leg->AddEntry(histe, "All layers", "l");
      }
      leg->Draw("sames");
    }
  }
}  

void etaPhiFwdPlot(TString fileName, TString plot, int first, int last, 
		   int drawLeg, bool debug) {

  TFile* hcalFile = new TFile(fileName);
  hcalFile->cd("g4SimHits");
  setStyle();

  TString xtit = TString("#eta");
  TString ytit = "none";
  int ymin = 0, ymax = 20, istart = 200;
  double xh = 0.90, xl=0.1;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("Material Budget (X_{0})");
    ymin = 0;  ymax = 400; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("Material Budget (Step Length)");
    ymin = 0;  ymax = 20000; istart = 300; xh = 0.70, xl=0.15;
  } else {
    ytit = TString("Material Budget (#lambda)");
    ymin = 0;  ymax = 30; istart = 200;
  }

  int index[10] = {9, 0, 1, 2, 3, 8, 4, 7, 5, 6};
  std::string label[10] = {"Empty", "Beam Pipe", "Tracker", "EM Calorimeter",
			   "Hadron Calorimeter", "Muon System", 
			   "Forward Hadron Calorimeter", "Shielding", "TOTEM",
			   "CASTOR"};
  
  TLegend *leg = new TLegend(xl, 0.75, xl+0.3, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.25);
  leg->SetTextSize(0.018);

  int nplots=0;
  TProfile *prof[nlaymax];
  for (int ii=last; ii>=first; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+index[ii]);
    gDirectory->GetObject(hname,prof[nplots]);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    if (xh < 0.8) 
      prof[nplots]->GetYaxis()->SetTitleOffset(1.7);
    sprintf(title, "%s", label[ii].c_str());
    leg->AddEntry(prof[nplots], title, "lf");
    if (debug) {
      int    nbinX = prof[nplots]->GetNbinsX();
      double xmin  = prof[nplots]->GetXaxis()->GetXmin();
      double xmax  = prof[nplots]->GetXaxis()->GetXmax();
      double dx    = (xmax - xmin)/nbinX;
      cout << "Hist " << ii;
      for (int ibx=0; ibx<nbinX; ibx++) {
	double xx1  = xmin + ibx*dx;
	double cont = prof[nplots]->GetBinContent(ibx+1);
	cout << " | " << ibx << "(" << xx1 << ":" << (xx1+dx) << ") " << cont;
      }
      cout << "\n";
    }
    nplots++;
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

