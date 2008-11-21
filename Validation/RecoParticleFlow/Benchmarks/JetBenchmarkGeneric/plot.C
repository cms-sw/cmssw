{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gROOT->ProcessLine(".L makeJetResolutionPlot.C");
Resolution(1,"JetBenchmark_Full_220pre1.root","JetBenchmark_Full_Barrel_220pre1.png");
Resolution(1,"JetBenchmark_Fast_220pre1.root","JetBenchmark_Fast_Barrel_220pre1.png");
Resolution(0,"JetBenchmark_Full_220pre1.root","JetBenchmark_Full_Endcap_220pre1.png");
Resolution(0,"JetBenchmark_Fast_220pre1.root","JetBenchmark_Fast_Endcap_220pre1.png");


gROOT->ProcessLine(".L Compare.C");
Compare(1,"JetBenchmark_Fast_220pre1.root","JetBenchmark_Full_220pre1.root","BarrelComparison_FastFull_220pre1.png");
Compare(0,"JetBenchmark_Fast_220pre1.root","JetBenchmark_Full_220pre1.root","EndcapComparison_FastFull_220pre1.png");

}
