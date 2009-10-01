{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Fast_Barrel_330pre5.png","Fast simulation - Barrel resolution");
Resolution(1,"JetBenchmark_Full_330pre5.root","JetBenchmark_Full_Barrel_330pre5.png","Full simulation - Barrel resolution");
Resolution(2,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Fast_Endcap_330pre5.png","Fast simulation - Endcap resolution");
Resolution(2,"JetBenchmark_Full_330pre5.root","JetBenchmark_Full_Endcap_330pre5.png","Full simulation - Endcap resolution");
//Resolution(3,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Fast_Forward_330pre5.png");
//Resolution(3,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Fast_Forward_330pre5.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Full_330pre5.root","BarrelComparison_FastFull_330pre5.png","Barrel Fast-Full comparison");
Compare(2,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Full_330pre5.root","EndcapComparison_FastFull_330pre5.png","Endcap Fast-Full comparison");
Compare(3,"JetBenchmark_Fast_330pre5.root","JetBenchmark_Full_330pre5.root","ForwardComparison_FastFull_330pre5.png","Forward Fast-Full comparison");

gROOT->ProcessLine(".L ResolutionDirection.C");
ResolutionDirection(1,"JetBenchmark_Fast_330pre5.root","AngularResolution_Fast_Barrel_330pre5.png");
ResolutionDirection(0,"JetBenchmark_Fast_330pre5.root","AngularResolution_Fast_Endcap_330pre5.png");
ResolutionDirection(1,"JetBenchmark_Full_330pre5.root","AngularResolution_Full_Barrel_330pre5.png");
ResolutionDirection(0,"JetBenchmark_Full_330pre5.root","AngularResolution_Full_Endcap_330pre5.png");

gROOT->ProcessLine(".L TrackMult.C");
TrackMult("JetBenchmark_Fast_330pre5.root","JetBenchmark_Full_330pre5.root");
}
