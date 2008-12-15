{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Full_222.root","JetBenchmark_Full_Barrel_222.png");
Resolution(1,"JetBenchmark_Fast_222.root","JetBenchmark_Fast_Barrel_222.png");
Resolution(0,"JetBenchmark_Full_222.root","JetBenchmark_Full_Endcap_222.png");
Resolution(0,"JetBenchmark_Fast_222.root","JetBenchmark_Fast_Endcap_222.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_222.root","JetBenchmark_Full_222.root","BarrelComparison_FastFull_222.png");
Compare(0,"JetBenchmark_Fast_222.root","JetBenchmark_Full_222.root","EndcapComparison_FastFull_222.png");

}
