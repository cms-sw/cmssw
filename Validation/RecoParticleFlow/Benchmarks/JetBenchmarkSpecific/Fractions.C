#include "TProfile.h"

void drawFrac(TProfile* fast, TProfile* full, float max=1., float min=-1.) { 

  gPad->SetGridx();
  gPad->SetGridy();

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

void Fractions(const char* fileFast, const char* fileFull)
{

gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");
//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();

TFile* fast = new TFile(fileFast);
gStyle->SetOptStat(0);
TProfile* fastChBrFrac  = dynamic_cast<TProfile*>(((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/BRCHEvsPt"))->ProfileX()->Rebin(5));
TProfile* fastEmBrFrac  = dynamic_cast<TProfile*>(((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/BRNEEvsPt"))->ProfileX()->Rebin(5));
TProfile* fastHaBrFrac  = dynamic_cast<TProfile*>(((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/BRNHEvsPt"))->ProfileX()->Rebin(5));
TProfile* fastChEnFrac  = dynamic_cast<TProfile*>(((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ERCHEvsPt"))->ProfileX()->Rebin(5));
TProfile* fastEmEnFrac  = dynamic_cast<TProfile*>(((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ERNEEvsPt"))->ProfileX()->Rebin(5));
TProfile* fastHaEnFrac  = dynamic_cast<TProfile*>(((TH2F*) fast->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ERNHEvsPt"))->ProfileX()->Rebin(5));

TFile* full = new TFile(fileFull);
TProfile* fullChBrFrac  = dynamic_cast<TProfile*>(((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/BRCHEvsPt"))->ProfileX()->Rebin(5));
TProfile* fullEmBrFrac  = dynamic_cast<TProfile*>(((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/BRNEEvsPt"))->ProfileX()->Rebin(5));
TProfile* fullHaBrFrac  = dynamic_cast<TProfile*>(((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/BRNHEvsPt"))->ProfileX()->Rebin(5));
TProfile* fullChEnFrac  = dynamic_cast<TProfile*>(((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ERCHEvsPt"))->ProfileX()->Rebin(5));
TProfile* fullEmEnFrac  = dynamic_cast<TProfile*>(((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ERNEEvsPt"))->ProfileX()->Rebin(5));
TProfile* fullHaEnFrac  = dynamic_cast<TProfile*>(((TH2F*) full->Get("DQMData/PFTask/Benchmarks/ak5PFJets/Gen/ERNHEvsPt"))->ProfileX()->Rebin(5));

TCanvas* c  = new TCanvas();
c->cd();
drawFrac(fastChBrFrac,fullChBrFrac);
gPad->SaveAs("ChBrFrac.png");

TCanvas* c0  = new TCanvas();
c0->cd();
drawFrac(fastEmBrFrac,fullEmBrFrac);
gPad->SaveAs("EmBrFrac.png");

TCanvas* c1  = new TCanvas();
c1->cd();
drawFrac(fastHaBrFrac,fullHaBrFrac);
gPad->SaveAs("HaBrFrac.png");

TCanvas* c2  = new TCanvas();
c2->cd();
drawFrac(fastChEnFrac,fullChEnFrac);
gPad->SaveAs("ChEnFrac.png");

TCanvas* c3  = new TCanvas();
c3->cd();
drawFrac(fastEmEnFrac,fullEmEnFrac);
gPad->SaveAs("EmEnFrac.png");

TCanvas* c4  = new TCanvas();
c4->cd();
drawFrac(fastHaEnFrac,fullHaEnFrac);
gPad->SaveAs("HaEnFrac.png");

}



