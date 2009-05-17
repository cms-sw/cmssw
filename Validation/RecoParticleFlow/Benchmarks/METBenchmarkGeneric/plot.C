{
  
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  gROOT->LoadMacro("../Tools/Comparator.C");

  gStyle->SetOptStat(1111);

  string dir1 = "DQMData/PFTask/Benchmarks/pfMet/Gen";
  string dir2 = "DQMData/PFTask/Benchmarks/met/Gen";
  const char* file1 = "benchmark.root";
  const char* file2 = "benchmark.root";

  float ptMin = 0;
  float ptMax = 9999;
 
  Style* style1 = spred;
  Style* style2 = spblue;
  Comparator::Mode mode = Comparator::SCALE;

  string outdir = "Plots";

  Comparator comp(file1,
		  dir1.c_str(),
		  file2,
		  dir2.c_str());
  comp.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");


  TCanvas c0("c0", "legend", 400, 200);
  FormatPad( &c0, false ); 
  comp.Legend().Draw();
  SavePlot("legend", outdir.c_str() );
    
  comp.SetAxis(10, -200,200);
  TCanvas c1a("c1a", "DeltaMET");
  FormatPad( &c1a, false); 
  comp.DrawSlice("DeltaEtvsEt", 20., 10000., mode);
//comp.Draw("DeltaEt", mode);
  SavePlot("deltaMET", outdir.c_str() );
  comp.SetAxis(5);

  TCanvas c1al("c1al", "DeltaMET_log");
  FormatPad( &c1al, false , false, true  ); 
  comp.DrawSlice("DeltaEtvsEt", 20., 10000., mode);
  SavePlot("deltaMET_log", outdir.c_str() );
  comp.SetAxis(5);

  TCanvas c1b("c1b", "MET");
  FormatPad( &c1b, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtRec", mode);
  SavePlot("MET", outdir.c_str() );

  TCanvas c2a("c2a", "MEX");
  FormatPad( &c2a, false );
  comp.Draw("ExRec", mode);
  SavePlot("MEX", outdir.c_str() );

  TCanvas c2b("c2b", "MEY");
  FormatPad( &c2b, false );
  comp.Draw("EyRec", mode);
  SavePlot("MEY", outdir.c_str() );

  comp.SetAxis(10, -3.2,3.2);
  TCanvas c3a("c3a", "DeltaPhi");
  FormatPad( &c3a, false );
  comp.DrawSlice("DeltaPhivsEt", 20., 10000., mode);
//comp.Draw("DeltaPhi", mode);
  SavePlot("deltaPhi", outdir.c_str() );

  TCanvas c3b("c3b", "pf vs Gen");
  FormatPad( &c3b, false );
  comp.Draw2D_file1("EtRecvsEt", mode);
  SavePlot("PF_vs_Gen", outdir.c_str() );

  TCanvas c4("c4", "Calo vs Gen");
  FormatPad( &c4, false );
  comp.Draw2D_file2("EtRecvsEt", mode);
  SavePlot("Calo_vs_Gen", outdir.c_str() );

  TCanvas c5("c5", "MET response");
  FormatPad( &c5, false );
  comp.DrawResp("DeltaEtOverEtvsEt", 0, 200, mode);
  SavePlot("MET_Response", outdir.c_str() );

  TCanvas c6("c6", "genMET");
  FormatPad( &c6, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtGen", mode);
  SavePlot("genMET", outdir.c_str() );
  comp.SetAxis(1);

  comp.SetAxis(10, -200,200);
  TCanvas c7("c7", "DeltaMET_20_50");
  FormatPad( &c7, false );
  comp.DrawSlice("DeltaEtvsEt", 20., 50., mode);
  SavePlot("deltaMET_20_50", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c8("c8", "DeltaMET_50_100");
  FormatPad( &c8, false );
  comp.DrawSlice("DeltaEtvsEt", 50., 100., mode);
  SavePlot("deltaMET_50_100", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c9("c9", "DeltaMET_100_200");
  FormatPad( &c9, false );
  comp.DrawSlice("DeltaEtvsEt", 100., 200., mode);
  SavePlot("deltaMET_100_200", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c10("c10", "DeltaPhi_20_50");
  FormatPad( &c10, false );
  comp.DrawSlice("DeltaPhivsEt", 20., 50., mode);
  SavePlot("deltaPhi_20_50", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c11("c11", "DeltaPhi_50_100");
  FormatPad( &c11, false );
  comp.DrawSlice("DeltaPhivsEt", 50., 100., mode);
  SavePlot("deltaPhi_50_100", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c12("c12", "DeltaPhi_100_200");
  FormatPad( &c12, false );
  comp.DrawSlice("DeltaPhivsEt", 100., 200., mode);
  SavePlot("deltaPhi_100_200", outdir.c_str() );

}
