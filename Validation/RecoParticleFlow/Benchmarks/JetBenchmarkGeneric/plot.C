{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Full_300pre7.root","JetBenchmark_Full_Barrel_300pre7.png","Full simulation - Barrel resolution");
Resolution(1,"JetBenchmark_Fast_300pre7.root","JetBenchmark_Fast_Barrel_300pre7.png","Fast simulation - Barrel resolution");
Resolution(2,"JetBenchmark_Full_300pre7.root","JetBenchmark_Full_Endcap_300pre7.png","Full simulation - Endcap resolution");
Resolution(2,"JetBenchmark_Fast_300pre7.root","JetBenchmark_Fast_Endcap_300pre7.png","Fast simulation - Endcap resolution");
//Resolution(3,"JetBenchmark_Full_300pre7.root","JetBenchmark_Full_Forward_300pre7.png");
//Resolution(3,"JetBenchmark_Fast_300pre7.root","JetBenchmark_Fast_Forward_300pre7.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_300pre7.root","JetBenchmark_Full_300pre7.root","BarrelComparison_FastFull_300pre7.png","Barrel Fast-Full comparison");
Compare(2,"JetBenchmark_Fast_300pre7.root","JetBenchmark_Full_300pre7.root","EndcapComparison_FastFull_300pre7.png","Endcap Fast-Full comparison");
Compare(3,"JetBenchmark_Fast_300pre7.root","JetBenchmark_Full_300pre7.root","ForwardComparison_FastFull_300pre7.png","Forward Fast-Full comparison");

}
