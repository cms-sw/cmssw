#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TRefArray.h"
#include "TStyle.h"
#include "TGraph.h"
#include <string>

int         colrs[5]  = {1, 2, 6, 3, 7};
int         symbol[5] = {24, 29, 25, 27, 26};
double      cutv[5]   = {10., 1., 0.1, 0.01, 0.001};

void PlotEcal(int ien=10, int birk=0, int cuts=1, int logy=0, int save=0) {

  setStyle();
  char fileName[50];
  TH1F *hi[5];
  int   nh=5;

  std::string flag="";
  TFile *file;
  if (birk != 2) {
    cuts = 0;
    if (birk == 0) flag = "No";
    for (int i=0; i<5; i++) {
      int cut=i+1;
      sprintf (fileName,"el%dGeVCut%d%sBirk_2.root",ien,cut,flag.c_str());
      std::cout << fileName << "\n";
      file = new TFile (fileName);
      hi[i] = (TH1F*) file->Get("DQMData/HcalTB06Histo/edecS");
    }
  } else {
    sprintf (fileName,"el%dGeVCut%dBirk_2.root",ien,cuts);
    std::cout << fileName << "\n";
    file = new TFile (fileName);
    hi[1] = (TH1F*) file->Get("DQMData/HcalTB06Histo/edecS");
    sprintf (fileName,"el%dGeVCut%dNoBirk_2.root",ien,cuts);
    std::cout << fileName << "\n";
    file = new TFile (fileName);
    hi[0] = (TH1F*) file->Get("DQMData/HcalTB06Histo/edecS");
    nh = 2;
  }

  double xmin = 0.8*ien;
  double xmax = 1.05*ien;
  double ymax = 0, ymin = 100;
  for (int i=0; i<nh; i++) {
    hi[i]->Rebin(2); hi[i]->SetTitle("");
    int nx = hi[i]->GetNbinsX();
    for (int k=1; k <= nx; k++) {
      double xx = hi[i]->GetBinCenter(k);
      double yy = hi[i]->GetBinContent(k);
      if (xx > xmin && xx < xmax) {
        if (yy > ymax) ymax = yy;
        if (yy < ymin && yy > 0) ymin = yy;
      }
    }
  }
  if (logy == 0) {ymax *= 1.2; ymin *= 0.8;}
  else           {ymax *=10.0; ymin *= 0.2; }

  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);

  for (int i=0; i<nh; i++) {
    hi[i]->SetLineStyle(1);  hi[i]->SetLineWidth(1); 
    hi[i]->SetLineColor(colrs[i]); hi[i]->GetXaxis()->SetRangeUser(xmin,xmax);
    hi[i]->GetYaxis()->SetRangeUser(ymin,ymax);
    hi[i]->GetXaxis()->SetTitle("Energy Depost (GeV)");
    hi[i]->GetYaxis()->SetTitle("Events");
    if (i == 0) hi[i]->Draw();
    else        hi[i]->Draw("same");
  }

  char title[60], list[60];
  TLegend *leg1;
  if (birk == 2) {
    leg1 = new TLegend(0.15,0.80,0.62,0.90);
    sprintf(title, "Effect of Birks Law for electron with cut %6.3f mm", cutv[cuts-1]);
  } else {
    leg1 = new TLegend(0.15,0.70,0.55,0.90);
    if (birk == 0) sprintf(title, "Birks Law switched Off (electron)");
    else           sprintf(title, "Birks Law switched On (electron)");
  }
  leg1->SetHeader(title); leg1->SetFillColor(0);  leg1->SetTextSize(0.03);
  if (birk == 2) {
    sprintf(list, "Birks Law switched Off"); leg1->AddEntry(hi[0],list,"F");
    sprintf(list, "Birks Law switched On");  leg1->AddEntry(hi[1],list,"F");
  } else {
    for (int i=0; i<nh; i++) {
      sprintf (list, "Production cut set to %6.3f mm", cutv[i]);
      leg1->AddEntry(hi[i],list,"F");
    }
  }
  leg1->Draw();

  if (save != 0) {
    char fname[100];
    if (save > 0) sprintf (fname, "plot%dGeVBirk%dCut%d.eps", ien, birk, cuts);
    else          sprintf (fname, "plot%dGeVBirk%dCut%d.gif", ien, birk, cuts);
    myc->SaveAs(fname);
  }
}

void PlotMean(int ien=10, int birk=0, int save=0) {

  setStyle();
  char fileName[50];
  TH1F *hi;
  int   nh=5, nen=1, iene[5], nset=1, ienb=0;
  double mean[5][5];

  std::string flag="";
  TFile *file;
  if (ien > 0) {
    iene[0] = ien;
  } else {
    nen = 5;
    iene[0] = 1;
    iene[1] = 3;
    iene[2] = 10;
    iene[3] = 30;
    iene[4] = 100;
    if (ien == -2) ienb = 1;
  }

  if (birk != 2) {
    if (birk == 0) flag = "No";
    for (int k=0; k<nen; k++) {
      ien = iene[k];
      for (int i=0; i<nh; i++) {
	int cut=i+1;
	sprintf (fileName,"el%dGeVCut%d%sBirk_2.root",ien,cut,flag.c_str());
	std::cout << fileName << "\n";
	file = new TFile (fileName);
	hi = (TH1F*) file->Get("DQMData/HcalTB06Histo/edecS");
	mean[k][i]  = hi->GetMean(1);
	mean[k][i] /= (double)(ien);
      }
    }
    nset = nen;
  } else {
    ien = iene[0];
    for (int i=0; i<nh; i++) {
      int cut=i+1;
      sprintf (fileName,"el%dGeVCut%dBirk_2.root",ien,cut);
      std::cout << fileName << "\n";
      file = new TFile (fileName);
      hi = (TH1F*) file->Get("DQMData/HcalTB06Histo/edecS");
      mean[0][i]  = hi->GetMean(1);
      mean[0][i] /= (double)(ien);
      sprintf (fileName,"el%dGeVCut%dNoBirk_2.root",ien,cut);
      std::cout << fileName << "\n";
      file = new TFile (fileName);
      hi = (TH1F*) file->Get("DQMData/HcalTB06Histo/edecS");
      mean[1][i]  = hi->GetMean(1);
      mean[1][i] /= (double)(ien);
    }
    nset = 2;
  }

  TGraph *gr[5];
  double val[5], ymax=0.95, ymin=0.85;
  if (birk == 0 || birk == 2) ymax = 1.05;
  if (birk == 0)              ymin = 0.90;
  std::cout << nset << " set(s) of " << nh << " means\n";
  for (int k=0; k<nset; k++) {
    std::cout << "Set " << k;
    for (int i=0; i<nh; i++) {
      std::cout << " Mean[" << i << "] = " << mean[k][i];
      val[i] = mean[k][i];
    }
    std::cout << "\n";
    gr[k] = new TGraph(nh, cutv, val); gr[k]->SetMarkerSize(1.5);
    gr[k]->SetTitle(""); gr[k]->SetLineColor(colrs[k]);
    gr[k]->SetLineStyle(k+1); gr[k]->SetLineWidth(2);
    gr[k]->SetMarkerColor(colrs[k]);  gr[k]->SetMarkerStyle(symbol[k]);
    gr[k]->GetXaxis()->SetTitle("Production cut value (mm)");
    gr[k]->GetYaxis()->SetTitle("Mean Energy/Incident Energy");
    gr[k]->GetXaxis()->SetRangeUser(0.0005, 20.0);
    gr[k]->GetYaxis()->SetRangeUser(ymin, ymax);
  }

  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  gPad->SetLogx(1);
  gr[ienb]->GetXaxis()->SetRangeUser(0.0005, 20.0);
  gr[ienb]->Draw("alp");
  for (int k=ienb+1; k<nset; k++) 
    gr[k]->Draw("lp");

  TLegend *leg1;
  if (birk == 2) leg1 = new TLegend(0.15,0.80,0.50,0.90);
  else           leg1 = new TLegend(0.15,0.75,0.37,0.90);
  leg1->SetFillColor(0);  leg1->SetTextSize(0.03); 
  char list[60];
  if (birk == 2) {
    ien = iene[0];
    sprintf(list, "Birks Law On  at %d GeV",ien); leg1->AddEntry(gr[0],list,"P");
    sprintf(list, "Birks Law Off at %d GeV",ien); leg1->AddEntry(gr[1],list,"P");
  } else {
    if (birk == 0) flag = "Off";
    else           flag = "On";
    sprintf(list, "Birks Law %s", flag.c_str()); leg1->SetHeader(list);
    for (int k=ienb; k<nset; k++) {
      sprintf (list, "%d GeV electron",iene[k]); leg1->AddEntry(gr[k],list,"P");
    }
  }
  leg1->Draw();

  if (save != 0) {
    char fname[100];
    ien = iene[ienb];
    if (save > 0) {
      if (nen == 1) sprintf (fname, "plotRatio%dGeVBirk%d.eps", ien, birk);
      else          sprintf (fname, "plotRatio%dCut%d.eps",     ienb,birk);
    } else {
      if (nen == 1) sprintf (fname, "plotRatio%dGeVBirk%d.gif", ien, birk);
      else          sprintf (fname, "plotRatio%dCut%d.gif",     ienb,birk);
    }
    myc->SaveAs(fname);
  }
}

void setStyle() {
  
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetTitleOffset(1.6,"Y");  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);

}
