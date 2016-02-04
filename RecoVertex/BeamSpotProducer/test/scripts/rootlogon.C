{
 // gDebug=7;
  gSystem->AddIncludePath(" -I$CMSSW_BASE/src/RecoVertex/BeamSpotProducer/interface");
  gSystem->AddIncludePath(" -I$CMSSW_BASE/src");
  gSystem->AddLinkedLibs(" -L$CMSSW_BASE/lib/$SCRAM_ARCH -lRecoVertexBeamSpotProducer");
  gROOT->ProcessLine(".L NtupleChecker.C+");
  gStyle->SetPalette(1);
}
