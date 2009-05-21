{
  TFile *f = new TFile("benchmark.root"); 
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();

  f->cd("DQMData/PFTask/Benchmarks/PFlowElectrons/Gen");  
  Style *spf = spred;  
  Style *sgen = sback;
  
  TCanvas *c1 = new TCanvas;
  FormatPad(c1,false);  

  TH1F* h = (TH1F*) gDirectory->Get("DeltaEt");
  h->Rebin(2);
  FormatHisto(h,spf);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(-30,30);
  h->Draw();
  SavePlot("electronbenchmark","Plots");
  c1->Clear();

  c1->SetLogy(1);
  TH1F* h = (TH1F*) gDirectory->Get("DeltaEta");
  h->Rebin(2);
  FormatHisto(h,spf);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(-0.3,0.3);
  h->Draw();
  SavePlot("deltaeta","Plots");
  c1->Clear();
  //c1->SetLogy(0);

  //c1->SetLogy(1);
  TH1F* h = (TH1F*) gDirectory->Get("DeltaPhi");
  h->Rebin(2);
  FormatHisto(h,spf);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(-0.3,0.3);
  h->Draw();
  SavePlot("deltaphi","Plots");
  c1->Clear();
  c1->SetLogy(0);

  TH1F* h = (TH1F*) gDirectory->Get("EtGen");
  //h->Rebin(2);
  FormatHisto(h,sgen);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(0,120);
  h->Draw();
  SavePlot("etgen","Plots");
  c1->Clear();

  TH1F* h = (TH1F*) gDirectory->Get("EtaGen");
  //h->Rebin(2);
  FormatHisto(h,sgen);
  h->SetStats(1111);
  //h->GetXaxis()->SetRangeUser(-4,4);
  h->Draw();
  SavePlot("etagen","Plots");
  c1->Clear();
  
  TH2F* g = (TH2F*) gDirectory->Get("DeltaEtOverEtvsEt");
  TH1D* p = g->ProjectionY();
  p->Rebin(4);
  FormatHisto(p,spf);
  p->SetStats(1111);
  //p->GetXaxis()->SetRangeUser(-1,1);
  p->Draw();
  SavePlot("etresolution","Plots");
  c1->Clear();

}


