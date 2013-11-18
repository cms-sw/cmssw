{
  gROOT->ProcessLine(".x initL1Analysis.C");
  
  std::cout << "Compilation : L1Ntuple" << std::endl;
  gROOT->ProcessLine(".L L1Ntuple.C++");

  std::cout << "Compilation : L1BitCorr" << std::endl;
  gROOT->ProcessLine(".L L1BitCorr.C++");

  std::cout << "Compilation : L1BitCorrLumi" << std::endl;
  gROOT->ProcessLine(".L L1BitCorrLumi.C++");

  std::cout << "Compilation : L1BitCorrV2" << std::endl;
  gROOT->ProcessLine(".L L1BitCorrV2.C++");

  std::cout << "Compilation : L1BitEff" << std::endl;
  gROOT->ProcessLine(".L L1BitEff.C++");
}

