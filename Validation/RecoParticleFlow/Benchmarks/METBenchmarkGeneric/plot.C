{
  gSystem->Load("libFWCoreFWLite.so");
  gSystem->Load("libValidationRecoParticleFlow.so");

  gStyle->SetOptStat(1111);

//string dir1 = "DQMData/PFTask/Benchmarks/pfMet/Gen";
//string dir2 = "DQMData/PFTask/Benchmarks/met/Gen";
  const string dir1 = "DQMData/Run\ 1/PFTask/Run\ summary/pfMet";
  const string dir2 = "DQMData/Run\ 1/PFTask/Run\ summary/metMuonJESCorAK5";
  //  const string dir2 = "DQMData/Run\ 1/PFTask/Run\ summary/met";
  const string dir3 = "DQMData/Run\ 1/PFTask/Run\ summary/genMetTrue";

//const char* file1 = "benchmark.root";
//const char* file2 = "benchmark.root";
  const char* file1 = "DQM_V0001_R000000001__A__B__C.root";
  const char* file2 = "DQM_V0001_R000000001__A__B__C.root";

  float ptMin = 0;
  float ptMax = 9999;
 
  Styles styles;
  Style* style1 = styles.spred;
  Style* style2 = styles.spblue;
  Comparator::Mode mode = Comparator::SCALE;

  string outdir = "Plots";

  Comparator comp(file1,
		  dir1.c_str(),
		  file2,
		  dir2.c_str());
  comp.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");

  TCanvas c0("c0", "legend", 400, 200);
  Styles::FormatPad( &c0, false ); 
  comp.Legend().Draw();
  Styles::SavePlot("legend", outdir.c_str() );

  comp.SetAxis(1, -200,200);
  TCanvas c1a("c1a", "DeltaMET");
  Styles::FormatPad( &c1a, false); 
  comp.DrawSlice("delta_et_VS_et_", 3, 1000, mode);
//comp.Draw("DeltaEt", mode);
  Styles::SavePlot("deltaMET", outdir.c_str() );

  comp.SetAxis(1, -400,400);
  TCanvas c1al("c1al", "DeltaMET_log");
  Styles::FormatPad( &c1al, false , false, true  ); 
  comp.DrawSlice("delta_et_VS_et_", 3, 1000, mode);
  Styles::SavePlot("deltaMET_log", outdir.c_str() );
//comp.SetAxis(1);

  TCanvas c1b("c1b", "MET");
  Styles::FormatPad( &c1b, false );
  comp.SetAxis(1,0,120);
  comp.Draw("pt_", mode);
  Styles::SavePlot("MET", outdir.c_str() );

  TCanvas c2a("c2a", "MEX");
  Styles::FormatPad( &c2a, false );
  comp.SetAxis(1,-100,100);
  comp.Draw("px_", mode);
  Styles::SavePlot("MEX", outdir.c_str() );

//TCanvas c2b("c2b", "MEY");
//Styles::FormatPad( &c2b, false );
//comp.Draw("EyRec", mode);
//Styles::SavePlot("MEY", outdir.c_str() );

  TCanvas c2c("c2c", "DeltaMEX");
  Styles::FormatPad( &c2c, false );
  comp.SetAxis(1,-100,100);
  comp.Draw("delta_ex_", mode);
  Styles::SavePlot("DeltaMEX", outdir.c_str() );

//TCanvas c2d("c2d", "DeltaMEY");
//Styles::FormatPad( &c2d, false );
//comp.Draw("DeltaEy", mode);
//Styles::SavePlot("DeltaMEY", outdir.c_str() );

  comp.SetAxis(1, -3.2,3.2);
  TCanvas c3a("c3a", "DeltaPhi");
  Styles::FormatPad( &c3a, false );
  comp.DrawSlice("delta_phi_VS_et_", 3, 1000, mode);
  Styles::SavePlot("deltaPhi", outdir.c_str() );

  TCanvas c3b("c3b", "pf vs Gen");
  Styles::FormatPad( &c3b, false );
  gStyle->SetPalette(1);
  TDirectory* dir = comp.dir0();
  dir->cd();
  TH2F *h2 = (TH2F*) dir->Get("RecEt_VS_TrueEt_");
  h2->Draw("colz");
  Styles::SavePlot("PF_vs_Gen", outdir.c_str() );

  TCanvas c4("c4", "Calo vs Gen");
  Styles::FormatPad( &c4, false );
  gStyle->SetPalette(1);
  TDirectory* dir = comp.dir1();
  dir->cd();
  TH2F *h2 = (TH2F*) dir->Get("RecEt_VS_TrueEt_");
  h2->Draw("colz");
  Styles::SavePlot("Calo_vs_Gen", outdir.c_str() );

  mode = Comparator::GRAPH;
  Style* style1gr = styles.sgr1;
  Style* style2gr = styles.sgr2;
  comp.SetStyles(style1gr, style2gr, "Particle Flow Met", "Calo Met");
  comp.SetAxis(1, 0.0,200.);
  TCanvas c5b("c5b", "MET response");
  Styles::FormatPad( &c5b, false );
  comp.DrawMeanSlice("delta_et_Over_et_VS_et_", 2, mode);
  Styles::SavePlot("MET_Response2", outdir.c_str() );

  comp.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");
  mode = Comparator::SCALE;

  TCanvas c6("c6", "genMET");
  Styles::FormatPad( &c6, false );
  Comparator comp3(file1, dir3.c_str(),
		   file2, dir3.c_str());
  comp3.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");
  comp3.SetAxis(1,0,120);
  comp3.Draw("pt_", mode);
  Styles::SavePlot("genMET", outdir.c_str() );
//  comp.SetAxis(1);

  comp.SetAxis(1, -200,200);
  TCanvas c7("c7", "DeltaMET_20_50");
  Styles::FormatPad( &c7, false );
  comp.DrawSlice("delta_et_VS_et_", 3, 5, mode);
  Styles::SavePlot("deltaMET_20_50", outdir.c_str() );

  comp.SetAxis(1, -1000,1000);
  TCanvas c8("c8", "DeltaMET_50_100");
  Styles::FormatPad( &c8, false );
  comp.DrawSlice("delta_et_VS_et_", 6, 10, mode);
  Styles::SavePlot("deltaMET_50_100", outdir.c_str() );

  comp.SetAxis(1, -200,200);
  TCanvas c9("c9", "DeltaMET_100_200");
  Styles::FormatPad( &c9, false );
  comp.DrawSlice("delta_et_VS_et_", 11, 20, mode);
  Styles::SavePlot("deltaMET_100_200", outdir.c_str() );

  comp.SetAxis(1, -200,200);
  TCanvas c10("c10", "DeltaPhi_20_50");
  Styles::FormatPad( &c10, false );
  comp.DrawSlice("delta_phi_VS_et_", 3, 5, mode);
  Styles::SavePlot("deltaPhi_20_50", outdir.c_str() );

  comp.SetAxis(1, -200,200);
  TCanvas c11("c11", "DeltaPhi_50_100");
  Styles::FormatPad( &c11, false );
  comp.DrawSlice("delta_phi_VS_et_", 6, 10, mode);
  Styles::SavePlot("deltaPhi_50_100", outdir.c_str() );

  comp.SetAxis(1, -200,200);
  TCanvas c12("c12", "DeltaPhi_100_200");
  Styles::FormatPad( &c12, false );
  comp.DrawSlice("delta_phi_VS_et_", 11, 20, mode);
  Styles::SavePlot("deltaPhi_100_200", outdir.c_str() );

  mode = Comparator::GRAPH;
  Style* style1gr = styles.sgr1;
  Style* style2gr = styles.sgr2;
  comp.SetStyles(style1gr, style2gr, "Particle Flow Met", "Calo Met");
  comp.SetAxis(1, 0.0, 200.);

  TCanvas c13("c13", "sigmaMET_MET");
  Styles::FormatPad( &c13, false );
  comp.DrawGaussSigmaOverMeanXSlice("delta_et_VS_et_", 2, 3, 20, false, mode);
  Styles::SavePlot("sigmaMET_MET", outdir.c_str() );

  TCanvas c14b("c14b", "sigmaPhi");
  Styles::FormatPad( &c14b, false );
  comp.DrawGaussSigmaSlice("delta_phi_VS_et_", 3, 3, 20, false, mode);
  Styles::SavePlot("sigmaPhib", outdir.c_str() );

// mode = Comparator::SCALE;

}
