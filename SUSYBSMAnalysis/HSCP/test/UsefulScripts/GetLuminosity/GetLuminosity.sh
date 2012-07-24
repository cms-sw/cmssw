#!/bin/bash
root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "-Wno-deprecated-declarations -Wno-deprecated ");
  TString dummy = makeshared.ReplaceAll("-Wshadow ", " -std=c++0x ");
  cout << "Compilling with the following arguments: " << makeshared << endl;
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libDataFormatsCommon.so");
  .x GetLuminosity.C+
EOF
pixelLumiCalc.py --nowarning -c frontier://LumiCalc/CMS_LUMI_PROD -i out.json overview > LUMI_TABLE
#pixelLumiCalc.py --nowarning -c frontier://LumiCalc/CMS_LUMI_PROD -i out_beforeRPCChange.json overview > LUMI_TABLE_BEFORE_RPC_CHANGE

#lumiCalc2.py --nowarning -c frontier://LumiCalc/CMS_LUMI_PROD -i out.json overview -b stable > LUMI_TABLE
#lumiCalc2.py --nowarning -c frontier://LumiCalc/CMS_LUMI_PROD -i out_beforeRPCChange.json overview -b stable > LUMI_TABLE_BEFORE_RPC_CHANGE
