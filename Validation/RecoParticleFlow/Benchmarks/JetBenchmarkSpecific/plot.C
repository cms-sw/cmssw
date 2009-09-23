{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Full_330pre4.root","JetBenchmark_Full_Barrel_330pre4.png","Full simulation - Barrel resolution");
Resolution(1,"JetBenchmark_Fast_330pre4.root","JetBenchmark_Fast_Barrel_330pre4.png","Fast simulation - Barrel resolution");
Resolution(2,"JetBenchmark_Full_330pre4.root","JetBenchmark_Full_Endcap_330pre4.png","Full simulation - Endcap resolution");
Resolution(2,"JetBenchmark_Fast_330pre4.root","JetBenchmark_Fast_Endcap_330pre4.png","Fast simulation - Endcap resolution");
//Resolution(3,"JetBenchmark_Full_330pre4.root","JetBenchmark_Full_Forward_330pre4.png");
//Resolution(3,"JetBenchmark_Fast_330pre4.root","JetBenchmark_Fast_Forward_330pre4.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_330pre4.root","JetBenchmark_Full_330pre4.root","BarrelComparison_FastFull_330pre4.png","Barrel Fast-Full comparison");
Compare(2,"JetBenchmark_Fast_330pre4.root","JetBenchmark_Full_330pre4.root","EndcapComparison_FastFull_330pre4.png","Endcap Fast-Full comparison");
Compare(3,"JetBenchmark_Fast_330pre4.root","JetBenchmark_Full_330pre4.root","ForwardComparison_FastFull_330pre4.png","Forward Fast-Full comparison");

gROOT->ProcessLine(".L ResolutionDirection.C");
ResolutionDirection(1,"JetBenchmark_Fast_330pre4.root","AngularResolution_Fast_Barrel_330pre4.png");
ResolutionDirection(0,"JetBenchmark_Fast_330pre4.root","AngularResolution_Fast_Endcap_330pre4.png");
ResolutionDirection(1,"JetBenchmark_Full_330pre4.root","AngularResolution_Full_Barrel_330pre4.png");
ResolutionDirection(0,"JetBenchmark_Full_330pre4.root","AngularResolution_Full_Endcap_330pre4.png");

}
