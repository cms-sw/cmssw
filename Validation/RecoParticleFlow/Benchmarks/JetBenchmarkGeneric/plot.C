{

gROOT->LoadMacro("Macros/NicePlot.C");
InitNicePlot();

TFile f("jetBenchmarkGeneric.root");


TCanvas c1;
FormatPad( &c1, false );

f.cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen");
TH2F* hpf = (TH2F*) gDirectory.Get("DeltaEtOverEtvsEt");
hpf.RebinX(2);
hpf.RebinY(2);
FormatHisto(hpf, s1);
hpf->Draw();

hpf->FitSlicesY();
TH1* hpf_1 = (TH1*) gROOT->FindObject("DeltaEtOverEtvsEt_1");
FormatHisto(hpf_1, s2);
hpf_1->Draw("same");


gPad->SaveAs("jetBenchmarkGeneric.png");

}
