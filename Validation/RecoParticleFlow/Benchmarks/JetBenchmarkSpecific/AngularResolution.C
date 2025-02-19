#include <vector>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TF1.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include "TCanvas.h"
#include <iostream>
#include <math.h>

void AngularResolution() { 

  TFile *f1 = new TFile("JetBenchmarkGeneric.root");
  f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen"); 
  TH2F* hEtaPF = (TH2F*) gDirectory->Get("DeltaEtavsEt");
  TH2F* hPhiPF = (TH2F*) gDirectory->Get("DeltaPhivsEt");
  f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen"); 
  TH2F* hEtaCALO = (TH2F*) gDirectory->Get("DeltaEtavsEt");
  TH2F* hPhiCALO = (TH2F*) gDirectory->Get("DeltaPhivsEt");
  f1->cd("DQMData/PFTask/Benchmarks/JetPlusTrackZSPCorJetIcone5/Gen"); 
  TH2F* hEtaJPT = (TH2F*) gDirectory->Get("DeltaEtavsEt");
  TH2F* hPhiJPT = (TH2F*) gDirectory->Get("DeltaPhivsEt");
  
  vector<TH1F*> histEtaPF;
  vector<TH1F*> histPhiPF;
  vector<TH1F*> histEtaJPT;
  vector<TH1F*> histPhiJPT;
  vector<TH1F*> histEtaCALO;
  vector<TH1F*> histPhiCALO;
  vector<double> pt, rmsEtaPF, rmsPhiPF, rmsEtaJPT, rmsPhiJPT, rmsEtaCALO, rmsPhiCALO;
  for ( unsigned bin=10; bin<250; ) { 

    string pfEtahname = "ResolutionEtaPF";
    string jptEtahname = "ResolutionEtaJPT";
    string caloEtahname = "ResolutionEtaCALO";
    string pfPhihname = "ResolutionPhiPF";
    string jptPhihname = "ResolutionPhiJPT";
    string caloPhihname = "ResolutionPhiCALO";
    char type[3];
    sprintf(type,"%i_%i",bin,bin+1);
    pfEtahname += type;
    jptEtahname += type;
    caloEtahname += type;
    pfPhihname += type;
    jptPhihname += type;
    caloPhihname += type;

    histEtaPF.push_back((TH1F*)hEtaPF->ProjectionY(pfEtahname.c_str(),bin,bin+2));
    histPhiPF.push_back((TH1F*)hPhiPF->ProjectionY(pfPhihname.c_str(),bin,bin+2));
    histEtaJPT.push_back((TH1F*)hEtaJPT->ProjectionY(jptEtahname.c_str(),bin,bin+2));
    histPhiJPT.push_back((TH1F*)hPhiJPT->ProjectionY(jptPhihname.c_str(),bin,bin+2));
    histEtaCALO.push_back((TH1F*)hEtaCALO->ProjectionY(caloEtahname.c_str(),bin,bin+2));
    histPhiCALO.push_back((TH1F*)hPhiCALO->ProjectionY(caloPhihname.c_str(),bin,bin+2));

    histPhiPF.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    histPhiCALO.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    histPhiJPT.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    histEtaPF.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    histEtaCALO.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    histEtaJPT.back()->GetXaxis()->SetRangeUser(-0.2,0.2);

    pt.push_back(2*bin+1);
    // std::cout << "Pt = " << pt.back() << std::endl;
    rmsEtaPF.push_back(histEtaPF.back()->GetRMS());
    rmsPhiPF.push_back(histPhiPF.back()->GetRMS());
    rmsEtaJPT.push_back(histEtaJPT.back()->GetRMS());
    rmsPhiJPT.push_back(histPhiJPT.back()->GetRMS());
    rmsEtaCALO.push_back(histEtaCALO.back()->GetRMS());
    rmsPhiCALO.push_back(histPhiCALO.back()->GetRMS());
    
    if ( pt.back() < 100. ) 
      bin += 1;
    else if ( pt.back() < 200 ) 
      bin += 2;
    else 
      bin += 5;


  }
  
  TGraph* resoEtaPF = new TGraph(pt.size(),&pt[0],&rmsEtaPF[0]);
  TGraph* resoPhiPF = new TGraph(pt.size(),&pt[0],&rmsPhiPF[0]);
  TGraph* resoEtaJPT = new TGraph(pt.size(),&pt[0],&rmsEtaJPT[0]);
  TGraph* resoPhiJPT = new TGraph(pt.size(),&pt[0],&rmsPhiJPT[0]);
  TGraph* resoEtaCALO = new TGraph(pt.size(),&pt[0],&rmsEtaCALO[0]);
  TGraph* resoPhiCALO = new TGraph(pt.size(),&pt[0],&rmsPhiCALO[0]);

  TH2F* hresEta = new TH2F("hresEta","",100,0,300,100,0.,0.07);
  TH2F* hresPhi = new TH2F("hresEta","",100,0,300,100,0.,0.10);

  TCanvas *CEta = new TCanvas("CEta","",1000, 700);
  TCanvas *CPhi = new TCanvas("CPhi","",1000, 700);

  CEta->cd();
  hresEta->SetStats(0);
  hresEta->SetTitle( "Eta Resolution" );
  hresEta->SetXTitle("p_{T} [GeV/c]" );
  hresEta->SetYTitle("Eta Resolution");
  hresEta->Draw();
  gPad->SetGridx();
  gPad->SetGridy();

  resoEtaPF->SetMarkerStyle(22);						
  resoEtaPF->SetMarkerSize(1.5);						
  resoEtaPF->SetMarkerColor(2);						
  resoEtaPF->Draw("P");

  resoEtaJPT->SetMarkerStyle(3);						
  resoEtaJPT->SetMarkerSize(1.5);						
  resoEtaJPT->SetMarkerColor(4);						
  resoEtaJPT->Draw("P");

  resoEtaCALO->SetMarkerStyle(30);						
  resoEtaCALO->SetMarkerSize(1.5);						
  resoEtaCALO->SetMarkerColor(1);						
  resoEtaCALO->Draw("P");

  TLegend *leg=new TLegend(0.70,0.65,0.85,0.85);
  leg->AddEntry(resoEtaCALO, "Calo", "p");
  leg->AddEntry(resoEtaJPT, "JPT", "p");
  leg->AddEntry(resoEtaPF, "PF 02/09", "p");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("EtaResolutionBarrel.png");

  CPhi->cd();
  hresPhi->SetStats(0);
  hresPhi->SetTitle( "Phi Resolution in the Barrel" );
  hresPhi->SetXTitle("p_{T} [GeV/c]" );
  hresPhi->SetYTitle("Phi Resolution (Rad.)");
  hresPhi->Draw();
  gPad->SetGridx();
  gPad->SetGridy();

  TF1* fPF = new TF1("fPF","[0]+[1]/x+[2]/(x*x)",20,300);
  fPF->SetParameters(0.01,0.70,0.);
  fPF->SetLineColor(2);
  fPF->SetLineWidth(4);
  resoPhiPF->SetMarkerStyle(22);						
  resoPhiPF->SetMarkerSize(1.2);						
  resoPhiPF->SetMarkerColor(2);						
  resoPhiPF->Fit("fPF","","",20,300);
  resoPhiPF->Draw("P");

  TF1* fJPT = new TF1("fJPT","[0]+[1]/x+[2]/(x*x)",20,300);
  fJPT->SetParameters(0.02,1.20,0.);
  fJPT->SetLineColor(4);
  fJPT->SetLineWidth(4);
  resoPhiJPT->SetMarkerStyle(23);						
  resoPhiJPT->SetMarkerSize(1.2);	
  resoPhiJPT->SetMarkerColor(4);						
  resoPhiJPT->Fit("fJPT","","",20,300);
  resoPhiJPT->Draw("P");

  TF1* fCALO = new TF1("fCALO","[0]+[1]/x+[2]/(x*x)",20,300);
  fCALO->SetParameters(0.02,1.20,0.);
  fCALO->SetLineColor(1);
  fCALO->SetLineWidth(4);
  resoPhiCALO->SetMarkerStyle(8);						
  resoPhiCALO->SetMarkerSize(1.2);						
  resoPhiCALO->SetMarkerColor(1);						
  resoPhiCALO->Fit("fCALO","","",20,300);
  resoPhiCALO->Draw("P");

  TLegend *legPhi=new TLegend(0.50,0.65,0.85,0.85);
  legPhi->AddEntry(resoPhiCALO, "Calorimeter Jets", "p");
  legPhi->AddEntry(resoPhiJPT, "Jet-plus-Track Jets", "p");
  legPhi->AddEntry(resoPhiPF, "Particle-Flow Jets, 02/09", "p");
  legPhi->SetTextSize(0.03);
  legPhi->Draw();

  gPad->SaveAs("PhiResolutionBarrel.png");

}
