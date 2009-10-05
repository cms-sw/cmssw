{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();
  
gStyle->SetOptStat(1111);

//   string dirpf = "DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen";
//   string dir2 = "DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen";
string dir1 = "DQMData/PFTask/Benchmarks/PFlowTaus/Gen";
//  string dir2 = "DQMData/PFTask/Benchmarks/PFlowTaus/Gen";
string dir2 = "DQMData/PFTask/Benchmarks/CaloTaus/Gen";
//  const char* file1 = "benchmark_oldmethod.root";
const char* file1 = "benchmark.root";
const char* file2 = "benchmark.root";
// const char* file2 = "benchmark_newmethod.root";
// const char* file1 = "ztautau.root";
// const char* file1 = "singleTaus.root";

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
Style* style2 = styles.spblue;
Comparator::Mode mode = Comparator::SCALE;

int etamode = BARREL;
//  int etamode = ENDCAP;

string outdir = "Plots_BarrelAndEndcap";

switch( etamode ) {
 case BARREL:
   dir1 = "DQMData/PFTask/Benchmarks/PFlowTaus_barrel/Gen";
   dir2 = "DQMData/PFTask/Benchmarks/CaloTaus_barrel/Gen";
   outdir = "Plots_Barrel";
   break;
 case ENDCAP:
   dir1 = "DQMData/PFTask/Benchmarks/PFlowTaus_endcap/Gen";
   dir2 = "DQMData/PFTask/Benchmarks/CaloTaus_endcap/Gen";
   outdir = "Plots_Endcap";
   break;
 default:
   break;
}   

Comparator comp(file1,
		dir1.c_str(),
		file2,
		dir2.c_str());
comp.SetStyles(style1, style2, "Particle Flow Taus", "Calo Taus");


TCanvas c0("c0", "legend", 400, 200);
Styles::FormatPad( &c0, false ); 
comp.Legend().Draw();
Styles::SavePlot("legend", outdir.c_str() );
    
comp.SetAxis(4, -30,30);
TCanvas c1a("c1a", "Tau benchmark, low pT");
Styles::FormatPad( &c1a, false ); 
comp.DrawSlice("DeltaEtvsEt", ptMin, ptMax, mode);
Styles::SavePlot("tauBenchmarkGeneric_lowpT", outdir.c_str() );
comp.SetAxis(5);


if(highpt) {
  TCanvas c1b("c1b", "Tau benchmark, high pT");
  Styles::FormatPad( &c1b, false );
  comp.SetAxis(5);
  comp.DrawSlice("DeltaEtOverEtvsEt", 400, 600, mode);
  Styles::SavePlot("tauBenchmarkGeneric_highpT", outdir.c_str() );
}

comp.SetAxis(2, -0.2,0.2);

TCanvas c2a("c2a", "Eta resolution, low pT");
Styles::FormatPad( &c2a, false );
comp.DrawSlice("DeltaEtavsEt", ptMin, ptMax, mode);
Styles::SavePlot("deltaEta_lowpT", outdir.c_str() );

if(highpt) {
  TCanvas c2b("c2b", "Eta resolution, high pT");
  Styles::FormatPad( &c2b, false );
  comp.DrawSlice("DeltaEtavsEt", 400, 600, mode);
  Styles::SavePlot("deltaEta_highpT", outdir.c_str() );
}

TCanvas c3a("c3a", "Phi resolution, low pT");
Styles::FormatPad( &c3a, false );
comp.DrawSlice("DeltaPhivsEt", ptMin, ptMax, mode);
Styles::SavePlot("deltaPhi_lowpT", outdir.c_str() );

if(highpt) {
  TCanvas c3b("c3b", "Phi resolution, high pT");
  Styles::FormatPad( &c3b, false );
  comp.DrawSlice("DeltaPhivsEt", 400, 600, mode);
  Styles::SavePlot("deltaPhi_highpT", outdir.c_str() );
}

comp.SetAxis(1);


bool efficiencies = false;

if (efficiencies ) {

  TCanvas c4("c4", "Efficiency PF");
  Styles::FormatPad( &c4, false );
  unsigned rebin = 10;

  Comparator comp2(file1,
		   dir1.c_str(),
		   file1,
		   dir1.c_str() ); 
  comp2.SetStyles(style1, style2,"num","denom");
  comp2.Draw("EtSeen","EtGen", Comparator::RATIO);
  Styles::SavePlot("efficiency_vs_pT_pf", outdir.c_str() );

  TCanvas c5("c5", "Efficiency Calo");
  Styles::FormatPad( &c5, false );

  Comparator comp3(file1,
		   dir2.c_str(),
		   file1,
		   dir2.c_str() ); 
  comp3.SetStyles(style1, style2,"num","denom");
  comp3.Draw("EtSeen","EtGen", Comparator::RATIO);
  Styles::SavePlot("efficiency_vs_pT_calo", outdir.c_str() );
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

  
}
