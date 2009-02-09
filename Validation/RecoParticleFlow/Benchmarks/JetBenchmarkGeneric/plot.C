{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Full_310pre1.root","JetBenchmark_Full_Barrel_310pre1.png","Full simulation - Barrel resolution");
Resolution(1,"JetBenchmark_Fast_310pre1.root","JetBenchmark_Fast_Barrel_310pre1.png","Fast simulation - Barrel resolution");
Resolution(2,"JetBenchmark_Full_310pre1.root","JetBenchmark_Full_Endcap_310pre1.png","Full simulation - Endcap resolution");
Resolution(2,"JetBenchmark_Fast_310pre1.root","JetBenchmark_Fast_Endcap_310pre1.png","Fast simulation - Endcap resolution");
//Resolution(3,"JetBenchmark_Full_310pre1.root","JetBenchmark_Full_Forward_310pre1.png");
//Resolution(3,"JetBenchmark_Fast_310pre1.root","JetBenchmark_Fast_Forward_310pre1.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_310pre1.root","JetBenchmark_Full_310pre1.root","BarrelComparison_FastFull_310pre1.png","Barrel Fast-Full comparison");
Compare(2,"JetBenchmark_Fast_310pre1.root","JetBenchmark_Full_310pre1.root","EndcapComparison_FastFull_310pre1.png","Endcap Fast-Full comparison");
Compare(3,"JetBenchmark_Fast_310pre1.root","JetBenchmark_Full_310pre1.root","ForwardComparison_FastFull_310pre1.png","Forward Fast-Full comparison");

}
