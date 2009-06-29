
void compare() {

  gROOT->Reset();
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  gROOT->LoadMacro("../Tools/Comparator.C");
  gStyle->SetOptStat(1111);
  
  Style* style1 = spred;
  Style* style2 = spblue;
  Comparator::Mode mode = Comparator::SCALE;


  Comparator comp("benchmark_0.root", 
	       "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen",
	       "benchmark_1.root",
	       "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen"
	       );
  comp.SetStyles(style1, style2, "CMSSW_3_1_0_pre10", "CMSSW_3_1_0_pre11");

 TCanvas c0("c0", "legend", 400, 200);
  FormatPad( &c0, false ); 
  comp.Legend().Draw();
  gPad->SaveAs("c_legend.png");

  TCanvas *c1 = new TCanvas();
  FormatPad( c1, false );

  comp.SetAxis(4, -30, 30);
  comp.DrawSlice("DeltaEtvsEt", 0, 9999, mode);
  
//  gPad->SetLogy();
  gPad->SaveAs("c_tauBenchmarkGeneric.png");
  
}
