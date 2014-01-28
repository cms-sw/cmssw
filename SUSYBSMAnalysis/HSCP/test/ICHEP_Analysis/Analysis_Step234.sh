root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "");
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libAnalysisDataFormatsSUSYBSMObjects.so");
  gSystem->Load("libDataFormatsVertexReco.so");
  gSystem->Load("libDataFormatsCommon.so");
  gSystem->Load("libDataFormatsHepMCCandidate.so");
  .x Analysis_Step234.C++("ANALYSE", 2, "dedxHarm2", "dedxHarm2", "combined", 40.0, 0.10, -1);
  //.x Analysis_Step234.C++("PLOT"     , 2, 0, "dedxASmi", "dedxHarm2", "combined", -0.6,-0.6,-0.6);
EOF

