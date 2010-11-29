{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Fast_Barrel.png","Fast simulation - Barrel resolution");
 Resolution(1,"JetBenchmark_Full_3100pre6.root","JetBenchmark_Full_Barrel.png","Full simulation - Barrel resolution");
Resolution(2,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Fast_Endcap.png","Fast simulation - Endcap resolution");
 Resolution(2,"JetBenchmark_Full_3100pre6.root","JetBenchmark_Full_Endcap.png","Full simulation - Endcap resolution");
//Resolution(3,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Fast_Forward_3100pre6.png");
//Resolution(3,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Fast_Forward_3100pre6.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Full_3100pre6.root","BarrelComparison_FastFull.png","Barrel Fast-Full comparison");
Compare(2,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Full_3100pre6.root","EndcapComparison_FastFull.png","Endcap Fast-Full comparison");
Compare(3,"JetBenchmark_Fast_3100pre6.root","JetBenchmark_Full_3100pre6.root","ForwardComparison_FastFull.png","Forward Fast-Full comparison");

gROOT->ProcessLine(".L ResolutionDirection.C");
ResolutionDirection(1,"JetBenchmark_Fast_3100pre6.root","AngularResolution_Fast_Barrel.png");
ResolutionDirection(0,"JetBenchmark_Fast_3100pre6.root","AngularResolution_Fast_Endcap.png");
ResolutionDirection(1,"JetBenchmark_Full_3100pre6.root","AngularResolution_Full_Barrel.png");
ResolutionDirection(0,"JetBenchmark_Full_3100pre6.root","AngularResolution_Full_Endcap.png");

gROOT->ProcessLine(".L TrackMult.C");
TrackMult("JetBenchmark_Fast_3100pre6.root","JetBenchmark_Full_3100pre6.root");

gROOT->ProcessLine(".L Fractions.C");
Fractions("JetBenchmark_Fast_3100pre6.root","JetBenchmark_Full_3100pre6.root");
}
