
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
  comp.SetStyles(style1, style2, "ZTT_SignalCone 0.15 (Barrel)", "ZTT_Isolated SignalCone 0.15 (Barrel)");

 TCanvas c0("c0", "legend", 400, 200);
  FormatPad( &c0, false ); 
  comp.Legend().Draw();
  gPad->SaveAs("c_legend.png");

  TCanvas *c1 = new TCanvas();
  FormatPad( c1, false );

  comp.SetAxis(4, -30, 30);
  comp.DrawSlice("DeltaEtvsEt", 0, 50, mode);
  
//  gPad->SetLogy();
  gPad->SaveAs("c_tauBenchmarkGeneric.png");
  
}
