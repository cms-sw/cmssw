#!/bin/bash

PUJSON=pileup_JSON_DCSONLY_190389-208686_corr.txt
MBXSEC=69300
MBXSECLOW=65835; #5% lower
MBXSECHIGH=72765; #5% higher

pileupCalc.py -i out.json --inputLumiJSON $PUJSON --calcMode true --minBiasXsec $MBXSEC --maxPileupBin 60 --numPileupBins 60 PileUp.root
pileupCalc.py -i out.json --inputLumiJSON $PUJSON --calcMode true --minBiasXsec $MBXSECHIGH --maxPileupBin 60 --numPileupBins 60 PileUp_XSecShiftUp.root
pileupCalc.py -i out.json --inputLumiJSON $PUJSON --calcMode true --minBiasXsec $MBXSECLOW --maxPileupBin 60 --numPileupBins 60 PileUp_XSecShiftDown.root

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
  .x GetDataPileUp.C+
EOF
