#!/bin/sh
JSON=Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt
PUJSON=pileup_latest.txt

#https://github.com/vhbb/cmssw/issues/505
#@capalmer85 might want to specify the latest and greatest effective minimum bias cross section, to my knowledge it is 69.2 ± 4.6%
MINBIAS=69200
MINBIASP=72383.2
MINBIASM=66016.8

pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec $MINBIAS --maxPileupBin 50 --numPileupBins 50  outputData.root
pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec $MINBIASP --maxPileupBin 50 --numPileupBins 50  outputDataP.root
pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec $MINBIASM --maxPileupBin 50 --numPileupBins 50  outputDataM.root


