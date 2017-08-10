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


void makePlotsCalib(std::string fname="RelValDoubleEle200PU.root",
		    std::string tag="All",
		    std::string text="Double Electron",
		    std::string dirnm="hgcalHitCalibration", 
		    bool save=false) {
  std::string names[4] = {"h_EoP_CPene_100_calib_fraction_",
			  "h_EoP_CPene_200_calib_fraction_",
			  "h_EoP_CPene_300_calib_fraction_",
			  "h_LayerOccupancy_"};
  std::string xname[4] = {"in 100#mum Silicon", "in 200#mum Silicon",
			  "in 300#mum Silicon", "Occupancy"};
  std::string xtitl[4] = {"E/E_{True}", "E/E_{True}", "E/E_{True}", "Layer #"};
  std::string ytitl[4] = {"Clusters", "Clusters", "Clusters", "Clusters"}; 
  int         type1[4] = {0,0,0,0};
  int         type2[4] = {5,5,5,1};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  TFile      *file = new TFile(fname.c_str());
  if (file) {
    TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
    if (dir) {
      char name[100];
      for (int k=0; k<4; ++k) {
	sprintf (name, "%s%s", names[k].c_str(), tag.c_str());
	TH1D* hist = (TH1D*)dir->FindObjectAny(name);
	std::cout << name << " read out at " << hist << std::endl;
        if (hist != 0) {
	  TCanvas *pad = new TCanvas(name,name,500,500);
	  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
          hist->GetYaxis()->SetTitle(ytitl[k].c_str());
          hist->GetXaxis()->SetTitle(xtitl[k].c_str());
	  hist->SetTitle(""); hist->Rebin(type2[k]);
	  hist->GetYaxis()->SetTitleOffset(1.5);
	  if (type1[k] == 1) pad->SetLogy();
	  hist->Draw();
	  pad->Update();
	  TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	  if (st1 != NULL) {
	    st1->SetY1NDC(0.70); st1->SetY2NDC(0.90);
	    st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	  }
	  TPaveText *txt1 = new TPaveText(0.11,0.84,0.64,0.89,"blNDC");
	  txt1->SetFillColor(0);
	  char txt[200];
	  sprintf (txt, "%s %s (%s)",tag.c_str(),xname[k].c_str(),text.c_str());
	  txt1->AddText(txt);
	  txt1->Draw("same");
	  pad->Modified();
	  pad->Update();
 	  if (save) {
	    sprintf (name, "c_%s%s.gif", names[k].c_str(), tag.c_str());
	    pad->Print(name);
	  }
	}
      }
    }
  }
}
