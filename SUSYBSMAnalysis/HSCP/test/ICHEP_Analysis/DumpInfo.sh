root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "-Wno-deprecated-declarations -Wno-deprecated ");
  TString dummy = makeshared.ReplaceAll("-Wshadow ", " -std=c++0x ");
  cout << "Compilling with the following arguments: " << makeshared << endl;
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libAnalysisDataFormatsSUSYBSMObjects.so");
  gSystem->Load("libDataFormatsVertexReco.so");
  gSystem->Load("libDataFormatsCommon.so");
  gSystem->Load("libDataFormatsHepMCCandidate.so");
  gSystem->Load("libDataFormatsTrackerRecHit2D.so");
 .x DumpInfo.C++("Results/Type5/", 45,0);
EOF

