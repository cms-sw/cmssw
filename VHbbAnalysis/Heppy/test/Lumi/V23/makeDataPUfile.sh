#!/bin/sh
JSON=Cert_271036-276384_13TeV_PromptReco_Collisions16_JSON_NoL1T.txt
PUJSON=pileup_latest.txt

MINBIAS=71300
MINBIASP=74865
MINBIASM=67735

pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec $MINBIAS --maxPileupBin 50 --numPileupBins 50  outputData.root
pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec $MINBIASP --maxPileupBin 50 --numPileupBins 50  outputDataP.root
pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec $MINBIASM --maxPileupBin 50 --numPileupBins 50  outputDataM.root


