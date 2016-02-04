
void compare() {
  gROOT->Reset();
  gROOT->LoadMacro("../Tools/Comparator.C");
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  
  Comparator c("benchmark_0.root", "benchmark_1.root");
  c.cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen");

  TCanvas *c1 = new TCanvas();
  FormatPad( c1, false );

  c.Draw("DeltaEt");
  
  // not very clean.. need a better way to apply a style
  FormatHisto( c.h0_, sback );
  FormatHisto( c.h1_, s2 );

  gPad->SetLogy();
  gPad->SaveAs("c_tauBenchmarkGeneric.png");
  
}
