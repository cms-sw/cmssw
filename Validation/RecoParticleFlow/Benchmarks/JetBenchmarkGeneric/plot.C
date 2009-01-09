{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Full_300pre6.root","JetBenchmark_Full_Barrel_300pre6.png");
Resolution(1,"JetBenchmark_Fast_300pre6.root","JetBenchmark_Fast_Barrel_300pre6.png");
Resolution(2,"JetBenchmark_Full_300pre6.root","JetBenchmark_Full_Endcap_300pre6.png");
Resolution(2,"JetBenchmark_Fast_300pre6.root","JetBenchmark_Fast_Endcap_300pre6.png");
Resolution(3,"JetBenchmark_Full_300pre6.root","JetBenchmark_Full_Forward_300pre6.png");
Resolution(3,"JetBenchmark_Fast_300pre6.root","JetBenchmark_Fast_Forward_300pre6.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_300pre6.root","JetBenchmark_Full_300pre6.root","BarrelComparison_FastFull_300pre6.png");
Compare(2,"JetBenchmark_Fast_300pre6.root","JetBenchmark_Full_300pre6.root","EndcapComparison_FastFull_300pre6.png");
Compare(3,"JetBenchmark_Fast_300pre6.root","JetBenchmark_Full_300pre6.root","ForwardComparison_FastFull_300pre6.png");

}
