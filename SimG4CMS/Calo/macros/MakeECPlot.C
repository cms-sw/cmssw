#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

void makePlots(std::string fname="runWithGun_Fix.root", 
	       std::string text="Fixed Depth Calculation",
	       std::string prefix="Fix", bool save=false) {
  std::string name[4] = {"ECLL_EB", "ECLL_EBref", "ECLL_EE", "ECLL_EERef"};
  double xrnglo[4] = {1200.0,1200.0,3100.0,3100.0};
  double xrnghi[4] = {1600.0,1600.0,3600.0,3600.0};
  std::string xtitl[4] = {"R_{Cyl} (mm)","R_{Cyl} (mm)","|z| (mm)", "|z| (mm)"};
  std::string ytitl[4] = {"# X_{0} (*100)", "# X_{0} (*100)", "# X_{0} (*100)", 
			  "# X_{0} (*100)"};
  std::string title[4] = {"EB (Normal)", "EB (Reflected)", "EE (Normal)", 
			  "EE (Reflected)"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(11110);
  TFile      *file = new TFile(fname.c_str());
  if (file) {
    char namep[100];
    for (int k=0; k<4; ++k) {
      TH2D* hist(0);
      for (int i=0; i<4; ++i) {
	if (i == 0) sprintf (namep, "%s", name[k].c_str());
	else        sprintf (namep, "%s;%d", name[k].c_str(),i);
	hist = (TH2D*)file->FindObjectAny(name[k].c_str());
	std::cout << namep << " read out at " << hist << std::endl;
	if (hist != 0) {
	  std::cout << "Entries " << hist->GetEntries() << std::endl;
	  if (hist->GetEntries() > 0) break;
	}
      }
      if (hist != 0) {
	sprintf (namep,"%s%s",name[k].c_str(),prefix.c_str());
	TCanvas *pad = new TCanvas(namep,namep,500,500);
	pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
	hist->GetYaxis()->SetTitle(ytitl[k].c_str());
	hist->GetXaxis()->SetTitle(xtitl[k].c_str());
	hist->SetTitle(title[k].c_str()); 
	hist->GetXaxis()->SetRangeUser(xrnglo[k],xrnghi[k]);
	hist->GetYaxis()->SetTitleOffset(1.4);
	hist->Draw();
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetY1NDC(0.70); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	}
	TPaveText *txt1 = new TPaveText(0.50,0.60,0.90,0.65,"blNDC");
	txt1->SetFillColor(0);
	txt1->AddText(text.c_str());
	pad->Update();
	if (save) {
	  sprintf (namep, "c_%s%s.gif",name[k].c_str(),prefix.c_str());
	  pad->Print(namep);
	}
      }
    }
  }
}

void comparePlots(std::string fname="elec10", int mom=10, bool save=false) {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1100);       gStyle->SetOptFit(0);

  std::string tags[2]   = {"Old", "New"};
  int  color[2]         = {2,4};
  int  marker[2]        = {20,21};
  int  style[2]         = {1,2};
  int  rebin[16]        = {20,20,20,20, 2, 2, 2, 2, 2, 2,10,10,10,10,10,10};
  int  type[16]         = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int  edgex[16]        = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::string name1[16] = {"Etot0",    "Etot1",    "EtotG0",   "EtotG1",
			 "r1by250",  "r1by251",  "r1by90",   "r1by91", 
			 "r9by250",  "r9by251",	 "sEtaEta0", "sEtaEta1",
			 "sEtaPhi0", "sEtaPhi1", "sPhiPhi0", "sPhiPhi1"};
  char name[100];
  TFile *file[2];
  for (int i=0; i<2; ++i) {
    sprintf (name, "%s%d%s.root", fname.c_str(), mom, tags[i].c_str());
    file[i] = new TFile(name);
  }
  for (int k=0; k<16; ++k) {
    TH1D* hist[2];
    sprintf (name, "%s", name1[k].c_str());
    for (int i=0; i<2; ++i) {
      hist[i] = (TH1D*)file[i]->FindObjectAny(name);
      if (hist[i] != 0) {
	if (rebin[k] > 1) hist[i]->Rebin(rebin[k]);
	hist[i]->GetXaxis()->SetLabelOffset(0.005);
	hist[i]->GetXaxis()->SetTitleOffset(1.40);
	hist[i]->GetYaxis()->SetLabelOffset(0.005);
	hist[i]->GetYaxis()->SetTitleOffset(1.40);
	hist[i]->SetMarkerStyle(marker[i]);
	hist[i]->SetMarkerColor(color[i]);
	hist[i]->SetLineColor(color[i]);
	hist[i]->SetLineStyle(style[i]);
	hist[i]->SetLineWidth(2);
      }
    }
    if (hist[0] != 0 && hist[1] != 0) {
      double xmin = (edgex[k] == 0) ? 0.65 : 0.11;
      TLegend *legend = new TLegend(xmin, 0.70, xmin+0.25, 0.79);
      legend->SetFillColor(kWhite);
      sprintf (name, "c_%sE%d", name1[k].c_str(),mom);
      TCanvas *pad = new TCanvas(name, name, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if (type[k] != 0) pad->SetLogy();
      double ytop(0.90), dy(0.05);
      for (int i=0; i<2; ++i) {
	if (i == 0) hist[i]->Draw("hist");
	else        hist[i]->Draw("sameshist");
	pad->Update();
	sprintf (name, "%d GeV Electron (%s)", mom, tags[i].c_str());
	legend->AddEntry(hist[i],name,"lp");
	TPaveStats* st1 = (TPaveStats*)hist[i]->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetLineColor(color[i]); st1->SetTextColor(color[i]);
	  st1->SetY1NDC(ytop-dy); st1->SetY2NDC(ytop);
	  st1->SetX1NDC(xmin); st1->SetX2NDC(xmin+0.25);
	  ytop -= dy;
	}
	pad->Modified();
	pad->Update();
      }
      legend->Draw("same");
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf", pad->GetName());
	pad->Print(name);
      }
    }
  }
}
