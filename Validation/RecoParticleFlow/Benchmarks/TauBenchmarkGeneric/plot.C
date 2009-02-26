{
  
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  gROOT->LoadMacro("../Tools/Comparator2.C");
  //  Manager::Init();

  Comparator comp("benchmark.root",
		  "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen",
		  "benchmark.root",
		  "DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen");
  comp.SetStyles(s1, s2);

  Comparator::Mode mode = Comparator::SCALE;
  
  TCanvas c1("c1", "Tau benchmark");
  FormatPad( &c1, false ); 
  comp.Draw("DeltaEt", mode);
  gPad->SaveAs("tauBenchmarkGeneric.png");

  TCanvas c2("c2", "Eta resolution");
  FormatPad( &c2, false );
  comp.Draw("DeltaEta", mode);
  gPad->SaveAs("deltaEta.png");

  TCanvas c3("c3", "Phi resolution");
  FormatPad( &c3, false );
  comp.Draw("DeltaPhi", mode);
  gPad->SaveAs("deltaPhi.png");

  TCanvas c4("c4", "Efficiency");
  FormatPad( &c4, false );
  unsigned rebin = 10;

  Comparator comp2("benchmark.root",
		   "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen",
		   "benchmark.root",
		   "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen" ); 
  comp2.SetStyles(s1, s2);
  comp2.Draw("EtSeen","EtGen", Comparator::EFF);
  gPad->SaveAs("efficiency_vs_pT.png");
}
