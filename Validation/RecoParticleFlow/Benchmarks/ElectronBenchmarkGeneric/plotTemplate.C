{
gROOT->LoadMacro("../Tools/NicePlot.C");
gROOT->LoadMacro("../Tools/Comparator.C");
  
  InitNicePlot();

  //  Manager::Init();

//  gStyle->SetOptStat(1111);
  gStyle->SetOptStat("mr");

const char* file1 = "FILE1";
const char* file2 = "FILE2";


  enum EtaModes {
    BARREL,
    ENDCAP,
    ALL
  };
  
  bool highpt = false;
  float ptMin = 0;
  float ptMax = 9999;
//Style* style1 = sback;
//Style* style2 = s1;
Style* style1 = spred;
Style* style2 = spblue;
int binning = 2;
Comparator::Mode mode = Comparator::SCALE;

//  int etamode = ALL;
int etamode = BARREL;
//int etamode = ENDCAP;

string dir1 = "DQMData/PFTask/Benchmarks/DIR1/Gen";
string dir2 = "DQMData/PFTask/Benchmarks/DIR2/Gen";
string outdir = "OUTDIR";

  Comparator comp(file1,
		  dir1.c_str(),
		  file2,
		  dir2.c_str());
  comp.SetStyles(style1, style2, "PFlowTaus", "PFlowTaus signal cone 0.5 no material effect");


  TCanvas c0("c0", "legend", 400, 200);
  FormatPad( &c0, false ); 
  comp.Legend().Draw();
  SavePlot("legend", outdir.c_str() );
    
  comp.SetAxis(binning, -30,30);
  TCanvas c1a("c1a", "Tau benchmark, low pT");
  FormatPad( &c1a, false ); 
  comp.DrawSlice("DeltaEtvsEt", ptMin, ptMax, mode);
  SavePlot("tauBenchmarkGeneric_lowpT", outdir.c_str() );
  comp.SetAxis(10);

  TCanvas c1a("c1a", "Tau benchmark relative, low pT");
  FormatPad( &c1a, false ); 
  comp.DrawSlice("DeltaEtOverEtvsEt", ptMin, ptMax, mode);
  SavePlot("tauBenchmarkGenericRelative_lowpT", outdir.c_str() );


  comp.SetAxis(binning, -0.2,0.2);
  TCanvas c2a("c2a", "Eta resolution, low pT");
  FormatPad( &c2a, false );
  comp.DrawSlice("DeltaEtavsEt", ptMin, ptMax, mode);
  SavePlot("deltaEta_lowpT", outdir.c_str() );

  TCanvas c3a("c3a", "Phi resolution, low pT");
  FormatPad( &c3a, false );
  comp.DrawSlice("DeltaPhivsEt", ptMin, ptMax, mode);
  SavePlot("deltaPhi_lowpT", outdir.c_str() );

exit(0);

  bool efficiencies = false;
  
}
