{
  
  gROOT->LoadMacro("../Tools/NicePlot.C");
  InitNicePlot();
  gROOT->LoadMacro("../Tools/Comparator.C");
  //  Manager::Init();

  gStyle->SetOptStat(1111);

  // we don't do comparisons between two objects, only PF electrons are shown
  string dir1 = "DQMData/PFTask/Benchmarks/PFlowElectrons/Gen";
  string dir2 = "DQMData/PFTask/Benchmarks/PFlowElectrons/Gen";
  const char* file1 = "benchmark.root";
  const char* file2 = "benchmark.root";


  enum EtaModes {
    BARREL,
    ENDCAP,
    ALL
  };
  
  bool highpt = false;
  float ptMin = 0;
  float ptMax = 9999;
  //Style* style1 = sback;
 
  // Style* style2 = s1;
  Style* style1 = spred;
  Style* style2 = spred;
  Comparator::Mode mode = Comparator::SCALE;

 int etamode = BARREL;
//  int etamode = ENDCAP;

 string outdir = "Plots_BarrelAndEndcap";
 
 // create directory if needed 
 TString release_name = gSystem->Getenv("DBS_RELEASE");
 gSystem->MakeDirectory(release_name);
 dir_name=release_name+"/"+TString(outdir);
 gSystem->MakeDirectory(dir_name); 

//  switch( etamode ) {
//  case BARREL:
//    dir1 = "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen";
//    dir2 = "DQMData/PFTask/Benchmarks/CaloTaus_barrel/Gen";
//    outdir = "Plots_Barrel";
//    break;
//  case ENDCAP:
//    dir1 = "DQMData/PFTask/Benchmarks/PFlowTaus_endcap/Gen";
//    dir2 = "DQMData/PFTask/Benchmarks/CaloTaus_endcap/Gen";
//    outdir = "Plots_Endcap";
//    break;
//  default:
//    break;
//  }   

  Comparator comp(file1,
		  dir1.c_str(),
		  file2,
		  dir2.c_str());
  comp.SetStyles(style1, style2, "Particle Flow Electrons", "");


  TCanvas c0("c0", "legend", 400, 200);
  FormatPad( &c0, false ); 
  comp.Legend().Draw();
  SavePlot("legend", outdir.c_str() );
    
  comp.SetAxis(4, -30,30);
  TCanvas c1a("c1a", "PF Electrons, pT");
  FormatPad( &c1a, false ); 
  comp.DrawSlice("DeltaEtvsEt", ptMin, ptMax, mode);
  SavePlot("deltaEt", outdir.c_str() );


  comp.SetAxis(2, -0.2,0.2);

  TCanvas c2a("c2a", "Eta resolution");
  FormatPad( &c2a, false,false,true );
  comp.DrawSlice("DeltaEtavsEt", ptMin, ptMax, mode);
  SavePlot("deltaEta", outdir.c_str() );

  if(highpt) {
    TCanvas c2b("c2b", "Eta resolution, high pT");
    FormatPad( &c2b, false );
    comp.DrawSlice("DeltaEtavsEt", 400, 600, mode);
    SavePlot("deltaEta_highpT", outdir.c_str() );
  }

  TCanvas c3a("c3a", "Phi resolution");
  FormatPad( &c3a, false,false,true );
  comp.DrawSlice("DeltaPhivsEt", ptMin, ptMax, mode);
  SavePlot("deltaPhi", outdir.c_str() );

  if(highpt) {
    TCanvas c3b("c3b", "Phi resolution, high pT");
    FormatPad( &c3b, false );
    comp.DrawSlice("DeltaPhivsEt", 400, 600, mode);
    SavePlot("deltaPhi_highpT", outdir.c_str() );
  }

  comp.SetAxis(1);


  bool efficiencies = true;

  if (efficiencies ) {

    TCanvas c4("c4", "Efficiency PF vs Pt low");
    FormatPad( &c4, false );
    unsigned rebin = 5;

    Comparator comp2(file1,
		     dir1.c_str(),
		     file1,
		     dir1.c_str() ); 
    comp2.SetStyles(style1, style2,"PFElectrons", "");
    comp2.SetAxis(rebin,0.,40.);
    comp2.Draw("EtGen","EtSeen", Comparator::EFF);
    SavePlot("efficiency_vs_pT_low", outdir.c_str() );

    TCanvas c5("c5", "Efficiency PF vs Pt high");
    FormatPad( &c5, false );

    comp2.SetStyles(style1, style2,"PFElectrons","");
    comp2.SetAxis(rebin,0,100);
    comp2.Draw("EtGen","EtSeen", Comparator::EFF);
    SavePlot("efficiency_vs_pT_high", outdir.c_str() );

    TCanvas c8("c8", "Efficiency PF vs eta");
    FormatPad( &c8, false );
    rebin = 2;

    comp2.SetAxis(rebin,-3,3);
    comp2.Draw("EtaGen","EtaSeen", Comparator::EFF);
    SavePlot("efficiency_vs_Eta", outdir.c_str() );

    TCanvas c9("c9", "Efficiency PF vs phi");
    FormatPad( &c9, false );
    rebin = 4;

    comp2.SetAxis(rebin,-TMath::Pi(),TMath::Pi());
    comp2.Draw("PhiGen","PhiSeen", Comparator::EFF);
    SavePlot("efficiency_vs_Phi", outdir.c_str() );
  }

  TCanvas c6("c6", "pT");
  FormatPad( &c6, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtGen", mode);
  SavePlot("etgen", outdir.c_str() );
  comp.SetAxis(1);

  
  TCanvas c7("c7", "eta");
  FormatPad( &c7, false );
  comp.Draw("EtaGen", mode);
  SavePlot("etagen", outdir.c_str() );

  TCanvas c10("c10", "Delta Et/Et");
  FormatPad( &c10, false );
  comp.DrawSlice("DeltaEtOverEtvsEt", ptMin, ptMax, mode);
  SavePlot("deltaEtoEt", outdir.c_str() );

  gROOT->ProcessLine(".q");
}
