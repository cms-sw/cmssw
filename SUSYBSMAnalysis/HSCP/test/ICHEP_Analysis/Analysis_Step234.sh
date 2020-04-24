root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "");
  TString dummy = makeshared.ReplaceAll("-Wshadow ", "");
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libAnalysisDataFormatsSUSYBSMObjects.so");
  gSystem->Load("libDataFormatsVertexReco.so");
  gSystem->Load("libDataFormatsCommon.so");
  gSystem->Load("libDataFormatsHepMCCandidate.so");
  gSystem->Load("libPhysicsToolsUtilities.so");
  .x Analysis_Step234.C++("ANALYSE_DATA", 2, "dedxASmi", "dedxHarm2", "combined", 0, 0, 0, 45.0, 2.1);
  //.x Analysis_Step234.C++("ANALYSE_SIGNAL", 2, "dedxASmi", "dedxHarm2", "combined", 40.0, 0.10, 2.1);
  //.x Analysis_Step234.C++("PLOT"     , 2, 0, "dedxASmi", "dedxHarm2", "combined", -0.6,-0.6,-0.6);
EOF

