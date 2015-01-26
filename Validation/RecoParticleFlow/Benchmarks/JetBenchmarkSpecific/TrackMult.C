#include "TProfile.h"

void drawFF(TProfile* fast, TProfile* full,int max=24, int min=0) { 
  fast->SetMaximum(max);
  fast->SetMinimum(min);
  fast->SetMarkerColor(4);						
  fast->SetMarkerStyle(25);
  fast->SetMarkerSize(0.8);
  fast->SetLineWidth(2);
  fast->SetLineColor(4);
  fast->Draw("erro");
  
  full->SetMaximum(max);
  full->SetMinimum(min);
  full->SetMarkerColor(2);						
  full->SetMarkerStyle(23);
  full->SetMarkerSize(0.8);
  full->SetLineWidth(2);
  full->SetLineColor(2);
  full->Draw("same");
}

void TrackMult(const char* fileFast, const char* fileFull)
{

gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");
//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();

TFile* fast = new TFile(fileFast);
gStyle->SetOptStat(0);
TProfile* fastNchEta  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCHvsEta"))->ProfileX();
TProfile* fastNch0Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH0vsEta"))->ProfileX();
TProfile* fastNch1Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH1vsEta"))->ProfileX();
TProfile* fastNch2Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH2vsEta"))->ProfileX();
TProfile* fastNch3Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH3vsEta"))->ProfileX();
TProfile* fastNch4Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH4vsEta"))->ProfileX();
TProfile* fastNch5Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH5vsEta"))->ProfileX();
TProfile* fastNch6Eta = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH6vsEta"))->ProfileX();

TProfile* fastNchPt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCHvsPt"))->ProfileX();
TProfile* fastNch0Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH0vsPt"))->ProfileX();
TProfile* fastNch1Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH1vsPt"))->ProfileX();
TProfile* fastNch2Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH2vsPt"))->ProfileX();
TProfile* fastNch3Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH3vsPt"))->ProfileX();
TProfile* fastNch4Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH4vsPt"))->ProfileX();
TProfile* fastNch5Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH5vsPt"))->ProfileX();
TProfile* fastNch6Pt  = ((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH6vsPt"))->ProfileX();

TFile* full = new TFile(fileFull);
TProfile* fullNchEta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCHvsEta"))->ProfileX();
TProfile* fullNch0Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH0vsEta"))->ProfileX();
TProfile* fullNch1Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH1vsEta"))->ProfileX();
TProfile* fullNch2Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH2vsEta"))->ProfileX();
TProfile* fullNch3Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH3vsEta"))->ProfileX();
TProfile* fullNch4Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH4vsEta"))->ProfileX();
TProfile* fullNch5Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH5vsEta"))->ProfileX();
TProfile* fullNch6Eta = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/NCH6vsEta"))->ProfileX();

TProfile* fullNchPt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCHvsPt"))->ProfileX();
TProfile* fullNch0Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH0vsPt"))->ProfileX();
TProfile* fullNch1Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH1vsPt"))->ProfileX();
TProfile* fullNch2Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH2vsPt"))->ProfileX();
TProfile* fullNch3Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH3vsPt"))->ProfileX();
TProfile* fullNch4Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH4vsPt"))->ProfileX();
TProfile* fullNch5Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH5vsPt"))->ProfileX();
TProfile* fullNch6Pt  = ((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ENCH6vsPt"))->ProfileX();

TCanvas* c  = new TCanvas();
c->cd();
drawFF(fastNchEta,fullNchEta);
gPad->SaveAs("NchvsEta_All.png");

TCanvas* c0  = new TCanvas();
c0->cd();
drawFF(fastNch0Eta,fullNch0Eta);
gPad->SaveAs("NchvsEta_Iter0.png");

TCanvas* c1  = new TCanvas();
c1->cd();
drawFF(fastNch1Eta,fullNch1Eta);
gPad->SaveAs("NchvsEta_Iter1.png");

TCanvas* c2  = new TCanvas();
c2->cd();
drawFF(fastNch2Eta,fullNch2Eta);
gPad->SaveAs("NchvsEta_Iter2.png");

TCanvas* c3  = new TCanvas();
c3->cd();
drawFF(fastNch3Eta,fullNch3Eta);
gPad->SaveAs("NchvsEta_Iter3.png");

TCanvas* c4  = new TCanvas();
c4->cd();
drawFF(fastNch4Eta,fullNch4Eta);
gPad->SaveAs("NchvsEta_Iter4.png");

TCanvas* c5  = new TCanvas();
c5->cd();
drawFF(fastNch5Eta,fullNch5Eta);
gPad->SaveAs("NchvsEta_Iter5.png");

TCanvas* c5b  = new TCanvas();
c5b->cd();
drawFF(fastNch6Eta,fullNch6Eta);
gPad->SaveAs("NchvsEta_Iter6.png");

TCanvas* c6  = new TCanvas();
c6->cd();
drawFF(fastNchPt,fullNchPt,35);
gPad->SaveAs("NchvsPt_All.png");

TCanvas* c7  = new TCanvas();
c7->cd();
drawFF(fastNch0Pt,fullNch0Pt,35);
gPad->SaveAs("NchvsPt_Iter0.png");

TCanvas* c8  = new TCanvas();
c8->cd();
drawFF(fastNch1Pt,fullNch1Pt,35);
gPad->SaveAs("NchvsPt_Iter1.png");

TCanvas* c9  = new TCanvas();
c9->cd();
drawFF(fastNch2Pt,fullNch2Pt,35);
gPad->SaveAs("NchvsPt_Iter2.png");

TCanvas* c10  = new TCanvas();
c10->cd();
drawFF(fastNch3Pt,fullNch3Pt,35);
gPad->SaveAs("NchvsPt_Iter3.png");

TCanvas* c11  = new TCanvas();
c11->cd();
drawFF(fastNch4Pt,fullNch4Pt,35);
gPad->SaveAs("NchvsPt_Iter4.png");

TCanvas* c12  = new TCanvas();
c12->cd();
drawFF(fastNch5Pt,fullNch5Pt,35);
gPad->SaveAs("NchvsPt_Iter5.png");

TCanvas* c13  = new TCanvas();
c13->cd();
drawFF(fastNch6Pt,fullNch6Pt,35);
gPad->SaveAs("NchvsPt_Iter6.png");

}



