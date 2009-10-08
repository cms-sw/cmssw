{
  gSystem->Load("libFWCoreFWLite.so");
  gSystem->Load("libValidationRecoParticleFlow.so");
  gSystem->Load("libCintex.so");
  ROOT::Cintex::Cintex::Enable();

  gStyle->SetOptStat(1111);

  string dir1 = "DQMData/PFTask/Benchmarks/pfMet/Gen";
  string dir2 = "DQMData/PFTask/Benchmarks/met/Gen";
  const char* file1 = "benchmark.root";
  const char* file2 = "benchmark.root";

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

  comp.SetAxis(5, -200,200);
  TCanvas c1a("c1a", "DeltaMET");
  Styles::FormatPad( &c1a, false); 
  comp.DrawSlice("DeltaEtvsEt", 20., 10000., mode);
//comp.Draw("DeltaEt", mode);
  Styles::SavePlot("deltaMET", outdir.c_str() );

  comp.SetAxis(5, -400,400);
  TCanvas c1al("c1al", "DeltaMET_log");
  Styles::FormatPad( &c1al, false , false, true  ); 
  comp.DrawSlice("DeltaEtvsEt", 20., 10000., mode);
  Styles::SavePlot("deltaMET_log", outdir.c_str() );
  comp.SetAxis(5);

  TCanvas c1b("c1b", "MET");
  Styles::FormatPad( &c1b, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtRec", mode);
  Styles::SavePlot("MET", outdir.c_str() );

  TCanvas c2a("c2a", "MEX");
  Styles::FormatPad( &c2a, false );
  comp.SetAxis(2,-100,100);
  comp.Draw("ExRec", mode);
  Styles::SavePlot("MEX", outdir.c_str() );

  TCanvas c2b("c2b", "MEY");
  Styles::FormatPad( &c2b, false );
  comp.Draw("EyRec", mode);
  Styles::SavePlot("MEY", outdir.c_str() );

  TCanvas c2c("c2c", "DeltaMEX");
  Styles::FormatPad( &c2c, false );
  comp.SetAxis(2,-100,100);
  comp.Draw("DeltaEx", mode);
  Styles::SavePlot("DeltaMEX", outdir.c_str() );

  TCanvas c2d("c2d", "DeltaMEY");
  Styles::FormatPad( &c2d, false );
  comp.Draw("DeltaEy", mode);
  Styles::SavePlot("DeltaMEY", outdir.c_str() );

  comp.SetAxis(10, -3.2,3.2);
  TCanvas c3a("c3a", "DeltaPhi");
  Styles::FormatPad( &c3a, false );
  comp.DrawSlice("DeltaPhivsEt", 20., 10000., mode);
//comp.Draw("DeltaPhi", mode);
  Styles::SavePlot("deltaPhi", outdir.c_str() );

  TCanvas c3b("c3b", "pf vs Gen");
  Styles::FormatPad( &c3b, false );
  gStyle->SetPalette(1);
  TDirectory* dir = comp.dir0();
  dir->cd();
  TH2F *h2 = (TH2F*) dir->Get("EtRecvsEt");
  h2->Draw("colz");
  Styles::SavePlot("PF_vs_Gen", outdir.c_str() );

  TCanvas c4("c4", "Calo vs Gen");
  Styles::FormatPad( &c4, false );
  gStyle->SetPalette(1);
  TDirectory* dir = comp.dir1();
  dir->cd();
  TH2F *h2 = (TH2F*) dir->Get("EtRecvsEt");
  h2->Draw("colz");
  Styles::SavePlot("Calo_vs_Gen", outdir.c_str() );

  mode = Comparator::GRAPH;
  Style* style1gr = styles.sgr1;
  Style* style2gr = styles.sgr2;
  comp.SetStyles(style1gr, style2gr, "Particle Flow Met", "Calo Met");
  comp.SetAxis(1, 0.0,200.);
  TCanvas c5b("c5b", "MET response");
  Styles::FormatPad( &c5b, false );
  comp.DrawMeanSlice("DeltaEtOverEtvsEt", 20, mode);
  Styles::SavePlot("MET_Response2", outdir.c_str() );

  comp.SetStyles(style1, style2, "Particle Flow Met", "Calo Met");
  mode = Comparator::SCALE;
//comp.DrawMeanSlice("DeltaEtOverEtvsEt", 0, 200, 10, -0.4, 0.4,"MET response;trueMET", "cst");

  TCanvas c6("c6", "genMET");
  Styles::FormatPad( &c6, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtGen", mode);
  Styles::SavePlot("genMET", outdir.c_str() );
  comp.SetAxis(1);

  comp.SetAxis(10, -200,200);
  TCanvas c7("c7", "DeltaMET_20_50");
  Styles::FormatPad( &c7, false );
  comp.DrawSlice("DeltaEtvsEt", 20., 50., mode);
  Styles::SavePlot("deltaMET_20_50", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c8("c8", "DeltaMET_50_100");
  Styles::FormatPad( &c8, false );
  comp.DrawSlice("DeltaEtvsEt", 50., 100., mode);
  Styles::SavePlot("deltaMET_50_100", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c9("c9", "DeltaMET_100_200");
  Styles::FormatPad( &c9, false );
  comp.DrawSlice("DeltaEtvsEt", 100., 200., mode);
  Styles::SavePlot("deltaMET_100_200", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c10("c10", "DeltaPhi_20_50");
  Styles::FormatPad( &c10, false );
  comp.DrawSlice("DeltaPhivsEt", 20., 50., mode);
  Styles::SavePlot("deltaPhi_20_50", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c11("c11", "DeltaPhi_50_100");
  Styles::FormatPad( &c11, false );
  comp.DrawSlice("DeltaPhivsEt", 50., 100., mode);
  Styles::SavePlot("deltaPhi_50_100", outdir.c_str() );

  comp.SetAxis(10, -200,200);
  TCanvas c12("c12", "DeltaPhi_100_200");
  Styles::FormatPad( &c12, false );
  comp.DrawSlice("DeltaPhivsEt", 100., 200., mode);
  Styles::SavePlot("deltaPhi_100_200", outdir.c_str() );

  mode = Comparator::GRAPH;
  Style* style1gr = styles.sgr1;
  Style* style2gr = styles.sgr2;
  comp.SetStyles(style1gr, style2gr, "Particle Flow Met", "Calo Met");
  comp.SetAxis(1, 0.0,200.);

//TCanvas c13("c13", "sigmaMET_MET");
//Styles::FormatPad( &c13, false );
//comp.DrawGaussSigmaOverMeanXSlice("DeltaEtvsEt", 20, 200, 9, 0.0, 1.1, "Sigma(DeltaMET)/MET;trueMET", "var", 5, -100.0,100.0,"SigmaOverMeanGaussSlice");
//Styles::SavePlot("sigmaMET_MET", outdir.c_str() );

//TCanvas c14b("c14b", "sigmaPhi");
// Styles::FormatPad( &c14b, false );
//comp.DrawGaussSigmaSlice("DeltaPhivsEt", 20, 200, 5, 0.0, 1.3, "Sigma(DeltaPhi);trueMET", "var", 5, -1.0,1.0,"SigmaGaussSlice",true);
//Styles::SavePlot("sigmaPhib", outdir.c_str() );

//// mode = Comparator::SCALE;

}
