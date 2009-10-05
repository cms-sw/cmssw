
void compare() {

  gROOT->Reset();
  gSystem->Load("libFWCoreFWLite.so");
  gSystem->Load("libValidationRecoParticleFlow.so");
  gSystem->Load("libCintex.so");
  ROOT::Cintex::Cintex::Enable();

  gStyle->SetOptStat(1111);
  
  Styles styles;
  Style* style1 = styles.spred;
  Style* style2 = styles.spblue;
  Comparator::Mode mode = Comparator::SCALE;
  

  Comparator comp("benchmark_0.root", 
	       "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen",
		  "benchmark_1.root",
	       "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen"
		  );
  comp.SetStyles(style1, style2, "310_pre10", "310_pre11");

  TCanvas c0("c0", "legend", 400, 200);
  styles.FormatPad( &c0, false, false, true ); 
  comp.Legend().Draw();
  gPad->SaveAs("c_legend.png");

  TCanvas *c1 = new TCanvas();
  styles.FormatPad( c1, false );

  comp.SetAxis(4, -30, 30);
  comp.DrawSlice("DeltaEtvsEt", 0, 9999, mode);
  
  gPad->SetLogy();
  gPad->SaveAs("c_tauBenchmarkGeneric.png");
  
}
