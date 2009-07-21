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

  comp.SetAxis(5, -200,200);
  TCanvas c1a("c1a", "DeltaMET");
  FormatPad( &c1a, false); 
  comp.DrawSlice("DeltaEtvsEt", 0., 10000., mode);
//comp.Draw("DeltaEt", mode);
  SavePlot("deltaMET", outdir.c_str() );

  comp.SetAxis(5, -500,500);
  TCanvas c1al("c1al", "DeltaMET_log");
  FormatPad( &c1al, false , false, true  ); 
  comp.DrawSlice("DeltaEtvsEt", 0., 10000., mode);
  SavePlot("deltaMET_log", outdir.c_str() );

  comp.SetAxis(5);
  TCanvas c1b("c1b", "MET");
  FormatPad( &c1b, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtRec", mode);
  SavePlot("MET", outdir.c_str() );

  TCanvas c2a("c2a", "MEX");
  FormatPad( &c2a, false );
  comp.SetAxis(2,-100,100);
  comp.Draw("ExRec", mode);
  SavePlot("MEX", outdir.c_str() );

  TCanvas c2b("c2b", "MEY");
  FormatPad( &c2b, false );
  comp.Draw("EyRec", mode);
  SavePlot("MEY", outdir.c_str() );

  TCanvas c2c("c2c", "DeltaMEX");
  FormatPad( &c2c, false );
  comp.SetAxis(2,-100,100);
  comp.Draw("DeltaEx", mode);
  SavePlot("DeltaMEX", outdir.c_str() );

  TCanvas c2d("c2d", "DeltaMEY");
  FormatPad( &c2d, false );
  comp.Draw("DeltaEy", mode);
  SavePlot("DeltaMEY", outdir.c_str() );

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

//  TCanvas c5("c5", "MET response");
//  FormatPad( &c5, false );
//  comp.DrawResp("DeltaEtOverEtvsEt", 0, 200, mode, -0.4, 0.4);
//  SavePlot("MET_Response", outdir.c_str() );

  TCanvas c6("c6", "genMET");
  FormatPad( &c6, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtGen", mode);
  SavePlot("genMET", outdir.c_str() );
  comp.SetAxis(1);

//  comp.SetAxis(10, -200,200);
//  TCanvas c7("c7", "DeltaMET_20_50");
//  FormatPad( &c7, false );
//  comp.DrawSlice("DeltaEtvsEt", 20., 50., mode);
//  SavePlot("deltaMET_20_50", outdir.c_str() );
//
//  comp.SetAxis(10, -200,200);
//  TCanvas c8("c8", "DeltaMET_50_100");
//  FormatPad( &c8, false );
//  comp.DrawSlice("DeltaEtvsEt", 50., 100., mode);
//  SavePlot("deltaMET_50_100", outdir.c_str() );
//
//  comp.SetAxis(10, -200,200);
//  TCanvas c9("c9", "DeltaMET_100_200");
//  FormatPad( &c9, false );
//  comp.DrawSlice("DeltaEtvsEt", 100., 200., mode);
//  SavePlot("deltaMET_100_200", outdir.c_str() );
//
//  comp.SetAxis(10, -200,200);
//  TCanvas c10("c10", "DeltaPhi_20_50");
//  FormatPad( &c10, false );
//  comp.DrawSlice("DeltaPhivsEt", 20., 50., mode);
//  SavePlot("deltaPhi_20_50", outdir.c_str() );
//
//  comp.SetAxis(10, -200,200);
//  TCanvas c11("c11", "DeltaPhi_50_100");
//  FormatPad( &c11, false );
//  comp.DrawSlice("DeltaPhivsEt", 50., 100., mode);
//  SavePlot("deltaPhi_50_100", outdir.c_str() );
//
//  comp.SetAxis(10, -200,200);
//  TCanvas c12("c12", "DeltaPhi_100_200");
//  FormatPad( &c12, false );
//  comp.DrawSlice("DeltaPhivsEt", 100., 200., mode);
//  SavePlot("deltaPhi_100_200", outdir.c_str() );

  comp.SetAxis(20, 0,3000);
  TCanvas c13t("c13t", "TrueSumEt");
  FormatPad( &c13t, false );
  comp.Draw("TrueSumEt", mode);
  SavePlot("TrueSumEt", outdir.c_str() );

  TCanvas c13("c13", "SumEt");
  FormatPad( &c13, false );
  comp.Draw("SumEt", mode);
  SavePlot("SumEt", outdir.c_str() );

  comp.SetAxis(5, -1000,1000);
  TCanvas c13b("c13b", "DeltaSumEt");
  FormatPad( &c13b, false );
  comp.DrawSlice("DeltaSetvsSet", 0., 3000., mode);
  SavePlot("DeltaSumEt", outdir.c_str() );

  mode = Comparator::NORMAL;

  TCanvas c14("c14", "SET response");
  FormatPad( &c14, false );
  comp.DrawResp("DeltaSetOverSetvsSet", 0., 3000., mode, -1., 0.);
  SavePlot("SET_Response", outdir.c_str() );

  TCanvas c15("c15", "sigmaDeltaMEX");
  FormatPad( &c15, false );
  comp.DrawSigmaEt2("DeltaMexvsSet", 0., 3000., mode);
  SavePlot("sigmaDeltaMEX", outdir.c_str() );

  TCanvas c16("c16", "<recoSet/TrueSet>");
  FormatPad( &c16, false );
  comp.DrawResp("RecSetOverTrueSetvsTrueSet", 0., 3000., mode, 0., 1., "trueSET");
  SavePlot("recSetOverTrueSet", outdir.c_str() );

  TCanvas c17("c17", "sigmaDeltaMEX / <recoSet/TrueSet>");
  FormatPad( &c17, false );
  comp.DrawSigmaEt_var("DeltaMexvsSet", "RecSetOverTrueSetvsTrueSet", 0., 3000., mode);
  SavePlot("sigmaDeltaMEX_ratio", outdir.c_str() );


//  TCanvas c13("c13", "sigmaMET_MET");
//  FormatPad( &c13, false );
//  comp.DrawSigmaEt_Et("DeltaEtvsEt", 20., 200., mode);
//  SavePlot("sigmaMET_MET", outdir.c_str() );

//  TCanvas c14("c14", "sigmaPhi");
//  FormatPad( &c14, false );
//  comp.DrawSigmaEt("DeltaPhivsEt", 20., 200., mode);
//  SavePlot("sigmaPhi", outdir.c_str() );

// mode = Comparator::SCALE;

//  TCanvas c14("c14", "test");
//  FormatPad( &c14, false );
//  comp.Draw2D_file1("DeltaEtvsEt", mode);
//  comp.DrawSlice("DeltaEtvsEt", 40., 60., mode);
//  SavePlot("test", outdir.c_str() );

}
