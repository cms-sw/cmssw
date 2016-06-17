#!/bin/sh
#JSON=newJSON_max196509.txt
JSON=Cert_246908-260627_13TeV_PromptReco_Collisions15_25ns_JSON_v2.txt
#/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/13TeV/Cert_246908-260627_13TeV_PromptReco_Collisions15_25ns_JSON_v2.txt
#PUJSON=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/PileUp/pileup_JSON_DCSONLY_190389-202478_corr.txt
PUJSON=pileup_latest.txt

#INBIAS=75000

#MINBIASP=66475
#MINBIASM=72450
#MINBIAS=71000

pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec 69000 --maxPileupBin 50 --numPileupBins 50  outputData.root
pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec 72450 --maxPileupBin 50 --numPileupBins 50  outputDataP.root
pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON  --calcMode true --minBiasXsec 65715 --maxPileupBin 50 --numPileupBins 50  outputDataM.root

#ileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON --calcMode true --minBiasXsec $MINBIAS --maxPileupBin 52 --numPileupBins 52  outputfile.root
#pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON --calcMode true --minBiasXsec $MINBIASP --maxPileupBin 60 --numPileupBins 60  outputfileP.root
#pileupCalc.py -i ${JSON} --inputLumiJSON $PUJSON --calcMode true --minBiasXsec $MINBIASM --maxPileupBin 60 --numPileupBins 60  outputfileM.root

