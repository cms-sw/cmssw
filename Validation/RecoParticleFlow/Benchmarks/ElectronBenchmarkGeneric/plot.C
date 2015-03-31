{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");

//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();
//gROOT->LoadMacro("../Tools/Comparator.C");
  //  Manager::Init();

  gStyle->SetOptStat(1111);

  // we don't do comparisons between two objects, only PF electrons are shown
  string dir1 = "DQMData/PFTask/Benchmarks/PFlowElectrons/Gen";
  string dir2 = "DQMData/PFTask/Benchmarks/PFlowElectrons/Gen";
  const char *file1;
  const char *file2;

  bool comparisonMode=false;
  if(TString(gSystem->Getenv("DBS_COMPARE_RELEASE"))!="") comparisonMode=true;

  if(!comparisonMode)
    {
      file1 = "benchmark.root";
      file2 = "benchmark.root";
    }
  else
    {
      file1 = "benchmark_0.root";
      if(TString(gSystem->Getenv("COMPARE_FILE"))=="")
	file2 = "benchmark_1.root";
      else 
	file2 = gSystem->Getenv("COMPARE_FILE");
    }
  cout << " Files " << file1 << " " << file2 << std::endl;
  
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
  Styles styles;
  Style* style1 = styles.spred;
  Style* style2 = styles.spred;
  if(comparisonMode) style2=styles.spblue;
  Comparator::Mode mode = Comparator::SCALE;

 int etamode = BARREL;
//  int etamode = ENDCAP;

 string outdir = "Plots_BarrelAndEndcap";
 if(comparisonMode) 
   outdir="Comp_BarrelAndEndcap";
 
 // create directory if needed 
 gSystem->MakeDirectory(TString(outdir)); 

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

 TString leg1="PF Electrons "+TString(gSystem->Getenv("DBS_COMPARE_RELEASE"));
 TString leg2="";
 if(comparisonMode)
   leg2="PF Electrons "+TString(gSystem->Getenv("DBS_RELEASE"));
 

 Comparator comp(file1,
		 dir1.c_str(),
		 file2,
		 dir2.c_str());
 comp.SetStyles(style1, style2, leg1.Data(), leg2.Data());
 

  TCanvas c0("c0", "legend", 400, 200);
  Styles::FormatPad( &c0, false ); 
  comp.Legend().Draw();
  Styles::SavePlot("legend", outdir.c_str() );
    
  comp.SetAxis(4, -30,30);
  TCanvas c1a("c1a", "PF Electrons, pT");
  Styles::FormatPad( &c1a, false ); 
  comp.DrawSlice("DeltaEtvsEt", ptMin, ptMax, mode);
  Styles::SavePlot("deltaEt", outdir.c_str() );


  comp.SetAxis(2, -0.2,0.2);

  TCanvas c2a("c2a", "Eta resolution");
  Styles::FormatPad( &c2a, false,false,true );
  comp.DrawSlice("DeltaEtavsEt", ptMin, ptMax, mode);
  Styles::SavePlot("deltaEta", outdir.c_str() );

  if(highpt) {
    TCanvas c2b("c2b", "Eta resolution, high pT");
    Styles::FormatPad( &c2b, false );
    comp.DrawSlice("DeltaEtavsEt", 400, 600, mode);
    Styles::SavePlot("deltaEta_highpT", outdir.c_str() );
  }

  TCanvas c3a("c3a", "Phi resolution");
  Styles::FormatPad( &c3a, false,false,true );
  comp.DrawSlice("DeltaPhivsEt", ptMin, ptMax, mode);
  Styles::SavePlot("deltaPhi", outdir.c_str() );

  if(highpt) {
    TCanvas c3b("c3b", "Phi resolution, high pT");
    Styles::FormatPad( &c3b, false );
    comp.DrawSlice("DeltaPhivsEt", 400, 600, mode);
    Styles::SavePlot("deltaPhi_highpT", outdir.c_str() );
  }

  comp.SetAxis(1);


  bool efficiencies = true;

  if (efficiencies ) {

    TCanvas c4("c4", "Efficiency PF vs Pt low");
    Styles::FormatPad( &c4, false );
    unsigned rebin = 5;

    Comparator comp2(file1,
		     dir1.c_str(),
		     file2,
		     dir2.c_str() ); 
    comp2.SetStyles(style1, style2,leg1.Data(),leg2.Data());
    comp2.SetAxis(rebin,0.,40.);
    comp2.Draw("EtSeen","EtGen", Comparator::EFF);
    Styles::SavePlot("efficiency_vs_pT_low", outdir.c_str() );

    TCanvas c5("c5", "Efficiency PF vs Pt high");
    Styles::FormatPad( &c5, false );

    Comparator comp3(file1,
		     dir1.c_str(),
		     file2,
		     dir2.c_str() ); 
    comp3.SetStyles(style1, style2,leg1.Data(),leg2.Data());

    comp3.SetStyles(style1, style2,leg1.Data(),leg2.Data());
    comp3.SetAxis(rebin,0,100.);
    comp3.Draw("EtSeen","EtGen", Comparator::EFF);
    Styles::SavePlot("efficiency_vs_pT_high", outdir.c_str() );

    TCanvas c8("c8", "Efficiency PF vs eta");
    Styles::FormatPad( &c8, false );
    rebin = 2;

    comp2.SetAxis(rebin,-3,3);
    comp2.Draw("EtaSeen","EtaGen", Comparator::EFF);
    Styles::SavePlot("efficiency_vs_Eta", outdir.c_str() );

    TCanvas c9("c9", "Efficiency PF vs phi");
    Styles::FormatPad( &c9, false );
    rebin = 4;

    comp2.SetAxis(rebin,-TMath::Pi(),TMath::Pi());
    comp2.Draw("PhiSeen","PhiGen", Comparator::EFF);
    Styles::SavePlot("efficiency_vs_Phi", outdir.c_str() );
  }

  TCanvas c6("c6", "pT");
  Styles::FormatPad( &c6, false );
  comp.SetAxis(2,0,120);
  comp.Draw("EtGen", mode);
  Styles::SavePlot("etgen", outdir.c_str() );
  comp.SetAxis(1);

  
  TCanvas c7("c7", "eta");
  Styles::FormatPad( &c7, false );
  comp.Draw("EtaGen", mode);
  Styles::SavePlot("etagen", outdir.c_str() );

  TCanvas c10("c10", "Delta Et/Et");
  Styles::FormatPad( &c10, false );
  comp.DrawSlice("DeltaEtOverEtvsEt", ptMin, ptMax, mode);
  Styles::SavePlot("deltaEtoEt", outdir.c_str() );

  gROOT->ProcessLine(".q");
}
