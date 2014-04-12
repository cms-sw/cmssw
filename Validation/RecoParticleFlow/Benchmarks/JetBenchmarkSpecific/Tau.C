#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TF1.h"
#include "TH2F.h"
#include "TLegend.h"


void Tau() {

  TFile *f1 = new TFile("TauBenchmarkGeneric.root");
  f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen"); 
  TH1F* DeltaEtPF = (TH1F*) gDirectory->Get("DeltaEt");
  f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen"); 
  TH1F* DeltaEtCALO = (TH1F*) gDirectory->Get("DeltaEt");
  f1->cd("DQMData/PFTask/Benchmarks/JetPlusTrackZSPCorJetIcone5/Gen"); 
  TH1F* DeltaEtJPT = (TH1F*) gDirectory->Get("DeltaEt");

  
  DeltaEtPF->SetStats(0);
  DeltaEtPF->SetTitle( "Taus with E_{T} = 50 GeV. Reconstructed Energy - True Energy" );
  DeltaEtPF->SetXTitle("#Delta E_{T} [GeV]" );
  DeltaEtPF->SetYTitle("Events");
  DeltaEtPF->Rebin(2);
  DeltaEtPF->SetLineColor(2);
  DeltaEtPF->SetLineWidth(2);  
  DeltaEtPF->SetFillStyle(3002);
  DeltaEtPF->SetFillColor(2);
  DeltaEtPF->GetXaxis()->SetRangeUser(-30.,+30.);
  DeltaEtPF->SetMaximum(200.);
  DeltaEtPF->Draw();

  DeltaEtCALO->Rebin(2);
  DeltaEtCALO->SetLineColor(1);
  DeltaEtCALO->SetLineWidth(2);
  DeltaEtCALO->SetFillStyle(3002);
  DeltaEtCALO->SetFillColor(1);
  DeltaEtCALO->GetXaxis()->SetRangeUser(-30.,+30.);
  DeltaEtCALO->Draw("same");

  DeltaEtJPT->Rebin(2);
  DeltaEtJPT->SetLineColor(4);
  DeltaEtJPT->SetLineWidth(2);
  DeltaEtJPT->SetFillStyle(3002);
  DeltaEtJPT->SetFillColor(4);
  DeltaEtJPT->GetXaxis()->SetRangeUser(-30.,+30.);
  DeltaEtJPT->Draw("same");

  TLegend *leg=new TLegend(0.55,0.65,0.85,0.85);
  leg->AddEntry(DeltaEtCALO, "Calorimeter Taus", "lf");
  leg->AddEntry(DeltaEtJPT, "Jet-plus-Track Taus", "lf");
  leg->AddEntry(DeltaEtPF, "Particle-Flow Taus", "lf");
  leg->SetTextSize(0.03);
  leg->Draw();

  gPad->SaveAs("Tau.png");
}
