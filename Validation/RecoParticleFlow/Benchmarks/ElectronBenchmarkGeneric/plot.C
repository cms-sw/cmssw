{
  // Get input file name from environment variable; set a default otherwise
  TString file_name = gSystem->Getenv("TEST_OUTPUT_FILE");
  if(file_name=="")
    file_name="benchmark.root";

  TFile *f = new TFile(file_name); 
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();

  // Get environment variables for output directory name
  TString dir_name = gSystem->Getenv("DIR_NAME");
  TString release_name = gSystem->Getenv("DBS_RELEASE");
  if(release_name=="")
    release_name="";
  if(dir_name=="")
    {
      cout << " No directory name in DIR_NAME; using Plots " << endl;
      dir_name="Plots";
    }
  else
    {
      gSystem->MakeDirectory(release_name);
      dir_name=release_name+"/"+dir_name;
      gSystem->MakeDirectory(dir_name);
      cout << dir_name << endl;
    }

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
  SavePlot("electronbenchmark",dir_name);
  c1->Clear();

  c1->SetLogy(1);
  TH1F* h = (TH1F*) gDirectory->Get("DeltaEta");
  h->Rebin(2);
  FormatHisto(h,spf);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(-0.3,0.3);
  h->Draw();
  SavePlot("deltaeta",dir_name);
  c1->Clear();
  //c1->SetLogy(0);

  //c1->SetLogy(1);
  TH1F* h = (TH1F*) gDirectory->Get("DeltaPhi");
  h->Rebin(2);
  FormatHisto(h,spf);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(-0.3,0.3);
  h->Draw();
  SavePlot("deltaphi",dir_name);
  c1->Clear();
  c1->SetLogy(0);

  TH1F* h = (TH1F*) gDirectory->Get("EtGen");
  //h->Rebin(2);
  FormatHisto(h,sgen);
  h->SetStats(1111);
  h->GetXaxis()->SetRangeUser(0,120);
  h->Draw();
  SavePlot("etgen",dir_name);
  c1->Clear();

  TH1F* h = (TH1F*) gDirectory->Get("EtaGen");
  //h->Rebin(2);
  FormatHisto(h,sgen);
  h->SetStats(1111);
  //h->GetXaxis()->SetRangeUser(-4,4);
  h->Draw();
  SavePlot("etagen",dir_name);
  c1->Clear();
  
  TH2F* g = (TH2F*) gDirectory->Get("DeltaEtOverEtvsEt");
  TH1D* p = g->ProjectionY();
  p->Rebin(4);
  FormatHisto(p,spf);
  p->SetStats(1111);
  //p->GetXaxis()->SetRangeUser(-1,1);
  p->Draw();
  SavePlot("etresolution",dir_name);
  c1->Clear();

  //   Efficiency plots
  // Eta 
  TH1F* EffEtaNum = (TH1F*) gDirectory->Get("EtaSeen")->Clone();
  TH1F* EffEtaDenom = (TH1F*) gDirectory->Get("EtaGen")->Clone();
  TH1F* EffEta = (TH1F*) gDirectory->Get("EtaSeen")->Clone();
  EffEtaNum->Rebin(2);
  EffEtaDenom->Rebin(2);
  EffEta->Rebin(2);
  EffEta->Clear();
  //  FormatHisto(EffEta,sgen);
  EffEta->SetMarkerStyle(23);
  EffEta->SetMarkerColor(2);
  EffEta->GetXaxis()->SetRangeUser(-3.,3.);
  EffEta->GetXaxis()->SetTitle("#eta_{gen}");
  EffEta->GetYaxis()->SetTitle("Efficiency");
  gStyle->SetOptStat(0);

  EffEtaNum->Sumw2();
  EffEtaDenom->Sumw2();
  EffEta->Divide(EffEtaNum,EffEtaDenom,1.,1.,"B");
  EffEta->SetMinimum(0.7);
  EffEta->Draw("E");
  SavePlot("etaeff",dir_name);
  c1->Clear();

  // Phi
  TH1F* EffPhiNum = (TH1F*) gDirectory->Get("PhiSeen")->Clone();
  TH1F* EffPhiDenom = (TH1F*) gDirectory->Get("PhiGen")->Clone();
  TH1F* EffPhi = (TH1F*) gDirectory->Get("PhiSeen")->Clone();
  EffPhiNum->Rebin(4);
  EffPhiDenom->Rebin(4);
  EffPhi->Rebin(4);
  EffPhi->Clear();
  //  FormatHisto(EffPhi,sgen);
  EffPhi->SetMarkerStyle(23);
  EffPhi->SetMarkerColor(2);
  EffPhi->GetXaxis()->SetRangeUser(-TMath::Pi(),TMath::Pi());
  EffPhi->GetXaxis()->SetTitle("#phi_{gen}");
  EffPhi->GetYaxis()->SetTitle("Efficiency");
  gStyle->SetOptStat(0);

  EffPhiNum->Sumw2();
  EffPhiDenom->Sumw2();
  EffPhi->Divide(EffPhiNum,EffPhiDenom,1.,1.,"B");
  EffPhi->SetMinimum(0.7);
  EffPhi->Draw("E");
  SavePlot("phieff",dir_name);
  c1->Clear();

  // Et - low
  TH1F* EffEtNum = (TH1F*) gDirectory->Get("EtSeen")->Clone();
  TH1F* EffEtDenom = (TH1F*) gDirectory->Get("EtGen")->Clone();
  TH1F* EffEt = (TH1F*) gDirectory->Get("EtSeen")->Clone();
  EffEtNum->Rebin(5);
  EffEtDenom->Rebin(5);
  EffEt->Rebin(5);
  EffEt->Clear();
  EffEtNum->Sumw2();
  EffEtDenom->Sumw2();
  //  FormatHisto(EffEt,sgen);
  EffEt->SetMarkerStyle(23);
  EffEt->SetMarkerColor(2);
  EffEt->GetXaxis()->SetRangeUser(0,40.);
  EffEt->GetXaxis()->SetTitle("p_{T}_{gen}");
  EffEt->GetYaxis()->SetTitle("Efficiency");
  gStyle->SetOptStat(0);

  //  EffEtNum->Sumw2();
  //  EffEtDenom->Sumw2();
  EffEt->Divide(EffEtNum,EffEtDenom,1.,1.,"B");
  EffEt->SetMinimum(0.7);
  EffEt->Draw("E");
  SavePlot("ptefflow",dir_name);
  c1->Clear();

  // Et - high
  EffEtNum = (TH1F*) gDirectory->Get("EtSeen")->Clone();
  EffEtDenom = (TH1F*) gDirectory->Get("EtGen")->Clone();
  EffEt = (TH1F*) gDirectory->Get("EtSeen")->Clone();
  EffEtNum->Rebin(2,"EffHighEtNum");
  EffEtDenom->Rebin(2,"EffHighEtDenom");
  EffEt->Rebin(2,"EffHighEt");

  TH1F * EffHighEtNum=gDirectory->Get("EffHighEtNum");
  TH1F * EffHighEtDenom=gDirectory->Get("EffHighEtDenom");
  TH1F * EffHighEt=gDirectory->Get("EffHighEt");
  EffHighEt->Clear();
//  //  FormatHisto(EffEtHigh,sgen);
  EffHighEt->SetMarkerStyle(23);
  EffHighEt->SetMarkerColor(2);
  EffHighEt->GetXaxis()->SetRangeUser(0,100.);
  EffHighEt->GetXaxis()->SetTitle("p_{T}_{gen}");
  EffHighEt->GetYaxis()->SetTitle("Efficiency");
  gStyle->SetOptStat(0);


  EffHighEtNum->Sumw2();
  EffHighEtDenom->Sumw2();
  EffHighEt->Divide(EffHighEtNum,EffHighEtDenom,1.,1.,"B");
  EffHighEt->SetMinimum(0.7);
  EffHighEt->Draw("E");
  SavePlot("pteffhigh",dir_name);
  c1->Clear();

  cout << "ok " << endl;
  gROOT->ProcessLine(".q");
}


