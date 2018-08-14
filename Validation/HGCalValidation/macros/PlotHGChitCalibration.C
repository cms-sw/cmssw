#include "TH1.h"
#include "TH2.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TFile.h"
#include "TString.h"
#include "TList.h"
#include "TStyle.h"
#include "TClass.h"
#include "TKey.h"
#include "TDirectory.h"

#include <cstdio>
#include <string>
#include <iostream>


void PlotHGChitCalibration(const TString valfile, const TString reffile, std::string val_ver, std::string ref_ver) {

  TFile *f1 = new TFile(valfile);
  TFile *f2 = new TFile(reffile);

  //1D Histos
  const int type = 4;
  const int part = 6;
  const int NHist = type*part;
  std::string particle[part] = {"All", "Electron", "Muon", "Photon", "ChgHad",
				"NeutHad"};
  std::string xtitl[type] = {"E/E_{True}", "E/E_{True}", "E/E_{True}",
			     "Layer #"};
  std::string ytitl[type] = {"Clusters", "Clusters", "Clusters", "Clusters"}; 
  std::string xname[type] = {"in 100#mum Silicon", "in 200#mum Silicon",
			     "in 300#mum Silicon", "Occupancy"};
  TH1* f1_hist[NHist];
  TH1* f2_hist[NHist];
  char name[64];
  std::vector<std::string>label;
  for (int i=0; i<part; ++i) {
    sprintf(name, "h_EoP_CPene_100_calib_fraction_%s",particle[i].c_str());
    label.push_back(name);
    sprintf(name,"h_EoP_CPene_200_calib_fraction_%s",particle[i].c_str());
    label.push_back(name);
    sprintf(name,"h_EoP_CPene_300_calib_fraction_%s",particle[i].c_str());
    label.push_back(name);
    sprintf(name,"h_LayerOccupancy_%s",particle[i].c_str());
    label.push_back(name);
  }
  for (int i=0; i<NHist; ++i) {
    int it = (i%type);
    int ip = (i - it)/type;
    sprintf(name, "hgcalHitCalibration/%s",label[i].c_str());
    
    //getting hist
    f1_hist[i] = (TH1*)f1->Get(name);
    f2_hist[i] = (TH1*)f2->Get(name);

    sprintf(name, "%s",label[i].c_str());

    //Drawing
    TCanvas *myc = new TCanvas("myc","",800,600);
    gStyle->SetOptStat(1111);

    f1_hist[i]->SetStats(kTRUE);   // stat box  
    f2_hist[i]->SetStats(kTRUE);  

    char txt[200];
    sprintf (txt, "%s %s",particle[ip].c_str(),xname[it].c_str());
    f1_hist[i]->SetTitle(txt);
    f2_hist[i]->SetTitle(txt);
     
    f1_hist[i]->SetLineWidth(2); 
    f2_hist[i]->SetLineWidth(2); 
     
    // diffferent histo colors and styles
    f1_hist[i]->SetLineColor(4);
    f1_hist[i]->SetLineStyle(1); 
    f1_hist[i]->GetYaxis()->SetTitle(ytitl[it].c_str());
    f1_hist[i]->GetXaxis()->SetTitle(xtitl[it].c_str());
     
    f2_hist[i]->SetLineColor(1);
    f2_hist[i]->SetLineStyle(2);  
    f2_hist[i]->GetYaxis()->SetTitle(ytitl[it].c_str());
    f2_hist[i]->GetXaxis()->SetTitle(xtitl[it].c_str());

    //Set maximum to the larger of the two
    if (f1_hist[i]->GetMaximum() < f2_hist[i]->GetMaximum()) f1_hist[i]->SetMaximum(1.05 * f2_hist[i]->GetMaximum());

    TLegend *leg = new TLegend(0.0, 0.91, 0.3, 0.99, "","brNDC");

    leg->SetBorderSize(2);
    //  leg->SetFillColor(51); 
    leg->SetFillStyle(1001); //
    leg->AddEntry(f1_hist[i],("CMSSW_"+val_ver).c_str(),"l");
    leg->AddEntry(f2_hist[i],("CMSSW_"+ref_ver).c_str(),"l"); 

    TPaveStats *ptstats = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
    ptstats->SetTextColor(4);
    f1_hist[i]->GetListOfFunctions()->Add(ptstats);
    ptstats->SetParent(f1_hist[i]->GetListOfFunctions());
    TPaveStats *ptstats2 = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
    ptstats2->SetTextColor(1);
    f2_hist[i]->GetListOfFunctions()->Add(ptstats2);
    ptstats2->SetParent(f2_hist[i]->GetListOfFunctions());
         
    f1_hist[i]->Draw(""); // "stat"   
    f2_hist[i]->Draw("histsames");   
     
    leg->Draw();   
    
    myc->SaveAs((std::string(name)+".png").c_str());
    if(myc) delete myc;

  }
}
