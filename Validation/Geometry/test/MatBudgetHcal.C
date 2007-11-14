// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

const int nlaymax = 25;
const int nbinmax = 260;
double meanIntLen[nlaymax][nbinmax],  diffIntLen[nlaymax][nbinmax]; 
double meanRadLen[nlaymax][nbinmax],  diffRadLen[nlaymax][nbinmax]; 
double meanStepLen[nlaymax][nbinmax], diffStepLen[nlaymax][nbinmax];

double towLow[81]    = {-4.889, -4.716, -4.538, -4.363, -4.191, 
			-4.013, -3.839,	-3.664, -3.489, -3.314,
			-3.139, -2.964,	-2.853, -2.650, -2.500,
			-2.322, -2.172,	-2.043, -1.930, -1.830,
			-1.740, -1.653,	-1.566, -1.479, -1.392,
			-1.305, -1.218,	-1.131, -1.044, -0.957,
			-0.870, -0.783,	-0.696, -0.609, -0.522,
			-0.435, -0.348, -0.261, -0.174, -0.087,  
 			 0.000,  0.087,  0.174,  0.261,  0.348, 
			 0.435,  0.522,  0.609,  0.696,  0.783,
			 0.870,  0.957,  1.044,  1.131,  1.218,
			 1.305,  1.392,  1.479,  1.566,  1.653,
			 1.740,  1.830,  1.930,  2.043,  2.172, 
			 2.322,  2.500,  2.650,  2.853,  2.964, 
			 3.139,  3.314,  3.489,  3.664,  3.839, 
			 4.013,  4.191,  4.363,  4.538,  4.716, 
			 4.889};
int colorLayer[25] = {152, 107,   9,  30,  34,  38,  14,  40,  41,  42,
		       45,  46,  48,  49,  37,  28,   4, 154, 104,  50,
		        3,   5,   6, 156, 159};

void etaPhiPlot(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		int ifirst=0, int ilast=19, int drawLeg=1, bool ifEta=true,
		double maxEta=-1) {

  TFile* hcalFile = new TFile(fileName);
  setStyle();

  TString xtit = TString("#eta");
  TString ytit = "none";
  int ymin = 0, ymax = 20, istart = 200;
  double xl = 0.74;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("HCal Material Budget X_{0}");
    ymin = 0;  ymax = 200; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("HCal Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 300; xl = 0.61;
  } else {
    ytit = TString("HCal Material Budget (#lambda)");
    ymin = 0;  ymax = 20; istart = 200;
  }
  if (!ifEta) {
    istart += 400;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xl, 0.60, xl+0.09, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.6);
  
  int nplots=0;
  TProfile *prof[nlaymax];
  for (int ii=ilast; ii>=ifirst; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+ii);
    prof[nplots] = (TProfile*)hcalFile->Get(hname);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    if (ifEta && maxEta > 0) 
      prof[nplots]->GetXaxis()->SetRangeUser(-maxEta,maxEta);
    sprintf(title, "Layer %d", ii+1);
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
  }

  TString cname = "c_" + plot + xtit;
  TCanvas *cc2 = new TCanvas(cname, cname, 700, 400);

  prof[0]->Draw("h");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("h sames");
  if (drawLeg > 0) leg->Draw("sames");
}


void setStyle() {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptStat(0);          gStyle->SetLegendBorderSize(1);
  gStyle->SetOptTitle(0);         gStyle->SetTitleOffset(2.5,"Y");

}

