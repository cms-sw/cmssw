root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "");
  TString dummy = makeshared.ReplaceAll("-Wshadow ", "");
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libAnalysisDataFormatsSUSYBSMObjects.so");
  gSystem->Load("libDataFormatsVertexReco.so");
  gSystem->Load("libDataFormatsCommon.so");
  gSystem->Load("libDataFormatsHepMCCandidate.so");
  gSystem->Load("libDataFormatsTrackerRecHit2D.so");
  .x DumpInfo.C++("Results/dedxASmi/combined/Eta25/PtMin25/Type0/",14,300);
EOF

