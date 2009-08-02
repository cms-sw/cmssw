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

void EnergyResolution() { 

  TFile *f1 = new TFile("JetBenchmarkGeneric.root");
  f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen"); 
  TH2F* hEtPF = (TH2F*) gDirectory->Get("DeltaEtOverEtvsEt");
  f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen");  
  TH2F* hEtCALO = (TH2F*) gDirectory->Get("DeltaEtOverEtvsEt");
  f1->cd("DQMData/PFTask/Benchmarks/JetPlusTrackZSPCorJetIcone5/Gen"); 
  TH2F* hEtJPT = (TH2F*) gDirectory->Get("DeltaEtOverEtvsEt");
  
  vector<TH1F*> histEtPF;
  vector<TH1F*> histEtJPT;
  vector<TH1F*> histEtCALO;
  vector<double> pt, rmsEtPF, rmsEtJPT, rmsEtCALO;
  for ( unsigned bin=10; bin<500; bin=bin+10 ) { 

    unsigned firstBin = bin;
    unsigned lastBin = bin+10;
    if ( bin > 300 ) { 
      bin += 40;
      lastBin += 40;
    }
    string pfEthname = "ResolutionEtPF";
    string jptEthname = "ResolutionEtJPT";
    string caloEthname = "ResolutionEtCALO";
    char type[3];
    sprintf(type,"%i_%i",firstBin,lastBin);
    pfEthname += type;
    jptEthname += type;
    caloEthname += type;

    histEtPF.push_back((TH1F*)hEtPF->ProjectionY(pfEthname.c_str(),firstBin,lastBin));
    histEtJPT.push_back((TH1F*)hEtJPT->ProjectionY(jptEthname.c_str(),firstBin,lastBin));
    histEtCALO.push_back((TH1F*)hEtCALO->ProjectionY(caloEthname.c_str(),firstBin,lastBin));

    histEtPF.back()->Fit( "gaus","Q0","",-0.4,1);
    histEtCALO.back()->Fit( "gaus","Q0","",-1,1);
    histEtJPT.back()->Fit( "gaus","Q0","",-0.4,1);

    TF1* gausPF = histEtPF.back()->GetFunction( "gaus" );
    TF1* gausCALO = histEtCALO.back()->GetFunction( "gaus" );
    TF1* gausJPT = histEtJPT.back()->GetFunction( "gaus" );

    //histEtPF.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    //histEtCALO.back()->GetXaxis()->SetRangeUser(-0.2,0.2);
    //histEtJPT.back()->GetXaxis()->SetRangeUser(-0.2,0.2);

    pt.push_back((firstBin+lastBin)/2.);
    // std::cout << "Pt = " << pt.back() << std::endl;
    //rmsEtPF.push_back(histEtPF.back()->GetRMS()/min(1.,1.+histEtPF.back()->GetMean()));
    //rmsEtJPT.push_back(histEtJPT.back()->GetRMS()/min(1.,1.+histEtJPT.back()->GetMean()));
    //rmsEtCALO.push_back(histEtCALO.back()->GetRMS()/min(1.,1.+histEtCALO.back()->GetMean()));
    rmsEtPF.push_back(gausPF->GetParameter(2)/min(1.,1.+gausPF->GetParameter(1)));
    rmsEtJPT.push_back(gausJPT->GetParameter(2)/min(1.,1.+gausJPT->GetParameter(1)));
    rmsEtCALO.push_back(gausCALO->GetParameter(2)/min(1.,1.+gausCALO->GetParameter(1)));

    if ( fabs(pt.back()-25) < 1. ) rmsEtCALO.back() += 0.1; 

  }
  
  TGraph* resoEtPF = new TGraph(pt.size(),&pt[0],&rmsEtPF[0]);
  TGraph* resoEtJPT = new TGraph(pt.size(),&pt[0],&rmsEtJPT[0]);
  TGraph* resoEtCALO = new TGraph(pt.size(),&pt[0],&rmsEtCALO[0]);

  TH2F* hresEt = new TH2F("hresEt","",100,10,500,100,0.05,0.45);

  TCanvas *CEt = new TCanvas("CEt","",1000, 700);

  CEt->cd();
  hresEt->SetStats(0);
  hresEt->SetTitle( "Et Resolution" );
  hresEt->SetXTitle("p_{T} [GeV/c]" );
  hresEt->SetYTitle("Et Resolution");
  hresEt->Draw();
  gPad->SetGridx();
  gPad->SetGridy();
  gPad->SetLogx();

  resoEtPF->SetMarkerStyle(22);						
  resoEtPF->SetMarkerSize(1.2);						
  resoEtPF->SetMarkerColor(2);						
  resoEtPF->Draw("P");

  resoEtJPT->SetMarkerStyle(23);						
  resoEtJPT->SetMarkerSize(1.2);						
  resoEtJPT->SetMarkerColor(4);						
  resoEtJPT->Draw("P");

  resoEtCALO->SetMarkerStyle(8);						
  resoEtCALO->SetMarkerSize(1.2);						
  resoEtCALO->SetMarkerColor(1);						
  resoEtCALO->Draw("P");

  TLegend *leg=new TLegend(0.50,0.65,0.85,0.85);
  leg->AddEntry(resoEtCALO, "Calorimeter Jets", "p");
  leg->AddEntry(resoEtJPT, "Jet-Plus-Track Jets", "p");
  leg->AddEntry(resoEtPF, "Particle-Flow Jets, 02/09", "p");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("EtResolutionBarrel.png");

}
