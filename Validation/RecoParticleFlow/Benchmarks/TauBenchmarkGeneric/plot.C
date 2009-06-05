{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile f("tauBenchmarkGeneric.root");


TCanvas c1;
FormatPad( &c1, false );

f.cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen");
TH1F* hpf = (TH1F*) gDirectory.Get("DeltaEt");
hpf.Rebin(2);
FormatHisto(hpf, s1);
hpf.Draw();
f.cd("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen");
TH1F* hcalo = (TH1F*) gDirectory.Get("DeltaEt");
hcalo.Rebin(2);
FormatHisto(hcalo, s2);
hcalo.Draw("same");

gPad->SaveAs("tauBenchmarkGeneric.png");

}
