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
   .x /uscms_data/d2/farrell3/WorkArea/14Aug2012/CMSSW_5_3_3/src/SUSYBSMAnalysis/HSCP/test/UsefulScripts/TriggerStudy/TriggerStudy.C++("summary_8TeV_Gluino", "Gluino_8TeV_M800_f10", "Gluino_8TeV_M1200_f10")
   .q
EOF
