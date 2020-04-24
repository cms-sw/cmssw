#!/bin/bash
root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  TString dummy = makeshared.ReplaceAll("-W ", "");
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();;
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libDataFormatsCommon.so");
  .x GetLuminosity.C+
EOF
lumiCalc2.py --nowarning -c frontier://LumiCalc/CMS_LUMI_PROD -i out.json overview -b stable > LUMI_TABLE
lumiCalc2.py --nowarning -c frontier://LumiCalc/CMS_LUMI_PROD -i out_beforeRPCChange.json overview -b stable > LUMI_TABLE_BEFORE_RPC_CHANGE
