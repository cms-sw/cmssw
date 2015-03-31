{
  gSystem->Load("libFWCoreFWLite.so");
  gSystem->Load("libValidationRecoParticleFlow.so");

//gStyle->SetOptStat(1111);

//  string dir1 = "DQMData/PFTask/Benchmarks/pfMet/Gen";
//  string dir2 = "DQMData/PFTask/Benchmarks/met/Gen";
//string dir1 = "PFTask/Benchmarks/PF";
//string dir2 = "PFTask/Benchmarks/corrCalo";
  const string dir1 = "DQMData/Run\ 1/PFTask/Run\ summary/pfMet";
  const string dir2 = "DQMData/Run\ 1/PFTask/Run\ summary/met";
  const string dir3 = "DQMData/Run\ 1/PFTask/Run\ summary/genMetTrue";
//const char* file1 = "DQM_V0001_R000000001__RelValQCD__FlatPt_15_3000_Fast__ParticleFlow.root";
//const char* file2 = "DQM_V0001_R000000001__RelValQCD__FlatPt_15_3000_Fast__ParticleFlow.root";
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
  comp.DrawSlice("delta_et_VS_et_", 0, 1000, mode);
  Styles::SavePlot("deltaMET", outdir.c_str() );

  comp.SetAxis(1, -500,500);
  TCanvas c1al("c1al", "DeltaMET_log");
  Styles::FormatPad( &c1al, false , false, true  ); 
  comp.DrawSlice("delta_et_VS_et_", 0, 1000, mode);
  Styles::SavePlot("deltaMET_log", outdir.c_str() );

////comp.SetAxis(5);
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
//
//  TCanvas c2b("c2b", "MEY");
//  Styles::FormatPad( &c2b, false );
//  comp.Draw("EyRec", mode);
//  Styles::SavePlot("MEY", outdir.c_str() );

  TCanvas c2c("c2c", "DeltaMEX");
  Styles::FormatPad( &c2c, false );
  comp.SetAxis(1,-100,100);
  comp.Draw("delta_ex_", mode);
  Styles::SavePlot("DeltaMEX", outdir.c_str() );

//  TCanvas c2d("c2d", "DeltaMEY");
//  Styles::FormatPad( &c2d, false );
//  comp.Draw("DeltaEy", mode);
//  Styles::SavePlot("DeltaMEY", outdir.c_str() );

  comp.SetAxis(1, -3.2,3.2);
  TCanvas c3a("c3a", "DeltaPhi");
  Styles::FormatPad( &c3a, false );
  comp.DrawSlice("delta_phi_VS_et_", 2, 1000, mode);
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

  TCanvas c6("c6", "genMET");
  Styles::FormatPad( &c6, false );
  Comparator comp3(file1, dir3.c_str(),
		   file2, dir3.c_str());
  comp3.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");
  comp3.SetAxis(1,0,120);
  comp3.Draw("pt_", mode);
  Styles::SavePlot("genMET", outdir.c_str() );

  comp3.SetAxis(1, 0,3000);
  TCanvas c13t("c13t", "TrueSumEt");
  Styles::FormatPad( &c13t, false );
  comp3.Draw("sumEt_", mode);
  Styles::SavePlot("TrueSumEt", outdir.c_str() );

  comp.SetAxis(1, 0,3000);
  TCanvas c13("c13", "SumEt");
  Styles::FormatPad( &c13, false );
  comp.Draw("sumEt_", mode);
  Styles::SavePlot("SumEt", outdir.c_str() );

  comp.SetAxis(1, -1000,1000);
  TCanvas c13b("c13b", "DeltaSumEt");
  Styles::FormatPad( &c13b, false );
  comp.DrawSlice("delta_set_VS_set_", 0, 3000, mode);
  Styles::SavePlot("DeltaSumEt", outdir.c_str() );

  mode = Comparator::GRAPH;
  Style* style1gr = styles.sgr1;
  Style* style2gr = styles.sgr2;
  comp.SetStyles(style1gr, style2gr, "Particle Flow Met", "Calo Met");
  comp.SetAxis(1, 0.0,3000.);
  TCanvas c14("c14", "SET response");
  Styles::FormatPad( &c14, false );
  comp.DrawMeanSlice("delta_set_Over_set_VS_set_", 30, mode);
  Styles::SavePlot("SET_Response", outdir.c_str() );

  TCanvas c15("c15", "sigmaDeltaMEX");
  Styles::FormatPad( &c15, false );
  comp.DrawGaussSigmaSlice("delta_ex_VS_set_", 20, mode);
  Styles::SavePlot("sigmaDeltaMEX", outdir.c_str() );

  TCanvas c16("c16", "<recoSet/TrueSet>");
  Styles::FormatPad( &c16, false );
  comp.DrawMeanSlice("RecSet_Over_TrueSet_VS_TrueSet_", 20, mode);
  Styles::SavePlot("recSetOverTrueSet", outdir.c_str() );

  TCanvas c17("c17", "sigmaDeltaMEX / <recoSet/TrueSet>");
  Styles::FormatPad( &c17, false );
  comp.DrawGaussSigmaOverMeanSlice("delta_ex_VS_set_", "RecSet_Over_TrueSet_VS_TrueSet_", 20, mode);
  Styles::SavePlot("sigmaDeltaMEX_ratio", outdir.c_str() );

//  comp.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");
//  mode = Comparator::SCALE;
//
////  TCanvas c14("c14", "test");
////  Styles::FormatPad( &c14, false );
////  comp.Draw2D_file1("DeltaEtvsEt", mode);
////  comp.DrawSlice("DeltaEtvsEt", 40., 60., mode);
////  Styles::SavePlot("test", outdir.c_str() );
//
}
