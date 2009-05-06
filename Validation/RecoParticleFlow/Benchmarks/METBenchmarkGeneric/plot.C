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
  TCanvas c1a("c1a", "deltaMET");
  FormatPad( &c1a, false ); 
//comp.DrawSlice("DeltaEtvsEt", ptMin, ptMax, mode);
  comp.Draw("DeltaEt", mode);
  SavePlot("deltaMET", outdir.c_str() );
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
  comp.Draw("DeltaPhi", mode);
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

  
//  TCanvas c7("c7", "eta");
//  FormatPad( &c7, false );
//  comp.Draw("EtaGen", mode);
//  SavePlot("etagen", outdir.c_str() );

//   TCanvas c8("c8", "nGen");
//   FormatPad( &c8, false );
//   comp.Draw("NGen", mode);
//   SavePlot("ngen", outdir.c_str() );

  
}
