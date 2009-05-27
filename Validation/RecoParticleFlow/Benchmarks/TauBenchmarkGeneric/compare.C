
void compare() {

  gROOT->Reset();
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  gROOT->LoadMacro("../Tools/Comparator.C");
  gStyle->SetOptStat(1111);
  
  Style* style1 = sback;
  Style* style2 = spblue;
  Comparator::Mode mode = Comparator::SCALE;


  Comparator comp("benchmark_0.root", 
	       "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen",
	       "benchmark_1.root",
	       "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen"
	       );
  comp.SetStyles(style1, style2, "Particle Flow Taus", "Calo Taus");

  TCanvas *c1 = new TCanvas();
  FormatPad( c1, false );

  comp.SetAxis(2, -20, 20);
  comp.DrawSlice("DeltaEtvsEt", 0, 50, mode);
  
  gPad->SetLogy();
  gPad->SaveAs("c_tauBenchmarkGeneric.png");
  
}
