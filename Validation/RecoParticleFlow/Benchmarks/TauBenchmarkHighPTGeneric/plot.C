{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();

//TFile f("tauBenchmarkGeneric.root");
TFile f("benchmark.root");

TCanvas c1;
Styles::FormatPad( &c1, false );

f.cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen");
TH2F* hpf = (TH2F*) gDirectory.Get("DeltaEtOverEtvsEt");
TH1D* hpfy = hpf->ProjectionY("ppf");
Styles::FormatHisto(hpfy, s1);
hpfy.Draw();

f.cd("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen");
TH2F* hcalo = (TH2F*) gDirectory.Get("DeltaEtOverEtvsEt");
TH1D* hcaloy = hcalo->ProjectionY("pcalo");
Styles::FormatHisto(hcaloy, s2);
hcaloy.Draw("same");

gPad->SaveAs("tauBenchmarkGeneric.png");

}
