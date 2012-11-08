root -l -b << EOF
   TString makeshared(gSystem->GetMakeSharedLib());
   TString dummy = makeshared.ReplaceAll("-W ", "-Wno-deprecated-declarations -Wno-deprecated ");
   TString dummy = makeshared.ReplaceAll("-Wshadow ", " -std=c++0x ");
   cout << "Compilling with the following arguments: " << makeshared << endl;
   gSystem->SetMakeSharedLib(makeshared);
   gSystem->SetIncludePath("-I$ROOFITSYS/include");
   gSystem->Load("libFWCoreFWLite");
   AutoLibraryLoader::enable();
   gSystem->Load("libDataFormatsFWLite.so");
   gSystem->Load("libAnalysisDataFormatsSUSYBSMObjects.so");
   gSystem->Load("libDataFormatsVertexReco.so");
   gSystem->Load("libDataFormatsHepMCCandidate.so");
   gSystem->Load("libPhysicsToolsUtilities.so");
   gSystem->Load("libdcap.so");
   .x TriggerStudy.C++("summary_8TeV_Gluino"     , "Gluino_8TeV_M400_f10" , "Gluino_8TeV_M800_f10" , "Gluino_8TeV_M1200_f10" );
   .x TriggerStudy.C++("summary_8TeV_Gluino_f100", "Gluino_8TeV_M400_f100", "Gluino_8TeV_M800_f100", "Gluino_8TeV_M1200_f100");
   .x TriggerStudy.C++("summary_8TeV_GMStau"     , "GMStau_8TeV_M100"     , "GMStau_8TeV_M308"     , "PPStau_8TeV_M100"      , "PPStau_8TeV_M308");
   .x TriggerStudy.C++("summary_8TeV_DYLQ"       , "DY_8TeV_M100_Q1o3"    , "DY_8TeV_M600_Q1o3"    , "DY_8TeV_M100_Q2o3"     , "DY_8TeV_M600_Q2o3");
   .x TriggerStudy.C++("summary_8TeV_DYHQ"       , "DY_8TeV_M100_Q2"      , "DY_8TeV_M600_Q2"      , "DY_8TeV_M100_Q5"       , "DY_8TeV_M600_Q5");
   .x TriggerStudy.C++("summary_7TeV_Gluino"     , "Gluino_7TeV_M400_f10" , "Gluino_7TeV_M800_f10" , "Gluino_7TeV_M1200_f10" );
   .x TriggerStudy.C++("summary_7TeV_Gluino_f100", "Gluino_7TeV_M400_f100", "Gluino_7TeV_M800_f100", "Gluino_7TeV_M1200_f100");
   .x TriggerStudy.C++("summary_7TeV_GMStau"     , "GMStau_7TeV_M100"     , "GMStau_7TeV_M308"     , "PPStau_7TeV_M100"      , "PPStau_7TeV_M308");
   .x TriggerStudy.C++("summary_7TeV_DYLQ"       , "DY_7TeV_M100_Q1o3"    , "DY_7TeV_M600_Q1o3"    , "DY_7TeV_M100_Q2o3"     , "DY_7TeV_M600_Q2o3");
   .x TriggerStudy.C++("summary_7TeV_DYHQ"       , "DY_7TeV_M100_Q2"      , "DY_7TeV_M600_Q2"      , "DY_7TeV_M100_Q5"       , "DY_7TeV_M600_Q5");
   .q
EOF
