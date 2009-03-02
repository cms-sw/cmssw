{
  
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  gROOT->LoadMacro("../Tools/Comparator2.C");
  //  Manager::Init();

  gStyle->SetOptStat(1111);

  Comparator comp("benchmark.root",
		  "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen",
		  "benchmark.root",
		  "DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen");
  comp.SetStyles(s1, s2, "Particle Flow Taus", "Calo Taus");

  Comparator::Mode mode = Comparator::NORMAL;


  comp.SetAxis(5);

  TCanvas c0("c0", "legend", 400, 200);
  FormatPad( &c0, false ); 
  comp.Legend().Draw();
  gPad->SaveAs("legend.png");
    
  TCanvas c1a("c1a", "Tau benchmark, low pT");
  FormatPad( &c1a, false ); 
  comp.DrawSlice("DeltaEtOverEtvsEt", 10, 50, mode);
  gPad->SaveAs("tauBenchmarkGeneric_lowpT.png");

  TCanvas c1b("c1b", "Tau benchmark, high pT");
  FormatPad( &c1b, false );
  comp.SetAxis(5);
  comp.DrawSlice("DeltaEtOverEtvsEt", 400, 600, mode);
  gPad->SaveAs("tauBenchmarkGeneric_highpT.png");

  comp.SetAxis(1, -0.1,0.1);

  TCanvas c2a("c2a", "Eta resolution, low pT");
  FormatPad( &c2a, false );
  comp.DrawSlice("DeltaEtavsEt", 10, 50, mode);
  gPad->SaveAs("deltaEta_lowpT.png");

  TCanvas c2b("c2b", "Eta resolution, high pT");
  FormatPad( &c2b, false );
  comp.DrawSlice("DeltaEtavsEt", 400, 600, mode);
  gPad->SaveAs("deltaEta_highpT.png");


  TCanvas c3a("c3a", "Phi resolution, low pT");
  FormatPad( &c3a, false );
  comp.DrawSlice("DeltaPhivsEt", 10, 50, mode);
  gPad->SaveAs("deltaPhi_lowpT.png");

  TCanvas c3b("c3b", "Phi resolution, high pT");
  FormatPad( &c3b, false );
  comp.DrawSlice("DeltaPhivsEt", 400, 600, mode);
  gPad->SaveAs("deltaPhi_highpT.png");

  comp.SetAxis(1);


  bool efficiencies = false;

  if (efficiencies ) {

  TCanvas c4("c4", "Efficiency PF");
  FormatPad( &c4, false );
  unsigned rebin = 10;

  Comparator comp2("benchmark.root",
		   "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen",
		   "benchmark.root",
		   "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen" ); 
  comp2.SetStyles(s1, s2);
  comp2.Draw("EtSeen","EtGen", Comparator::EFF);
  gPad->SaveAs("efficiency_vs_pT_pf.png");

  TCanvas c5("c5", "Efficiency Calo");
  FormatPad( &c5, false );

  Comparator comp3("benchmark.root",
		   "DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen",
		   "benchmark.root",
		   "DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen" ); 
  comp3.SetStyles(s1, s2);
  comp3.Draw("EtSeen","EtGen", Comparator::EFF);
  gPad->SaveAs("efficiency_vs_pT_calo.png");
}

  TCanvas c6("c6", "pT");
  FormatPad( &c6, false );

  comp.Draw("EtGen", mode);
  gPad->SaveAs("etgen.png");

  TCanvas c7("c7", "eta");
  FormatPad( &c7, false );

  comp.Draw("EtaGen", mode);
  gPad->SaveAs("etagen.png");

  
}
