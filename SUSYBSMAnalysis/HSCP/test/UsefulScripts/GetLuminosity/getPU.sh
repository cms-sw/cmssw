#/bin/bash

jsons=(out.json) 

#for2011
#PUJSON=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/PileUp/pileup_JSON_2011_4_2_validation.txt 
#MBXSEC = 68000
#for2012
PUJSON=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/PileUp/pileup_latest.txt
MBXSEC=70300

for j in ${jsons[@]}; do 
   pileupCalc.py -i ${j} --inputLumiJSON ${PUJSON} --calcMode true --minBiasXsec ${MBXSEC} --maxPileupBin 65 --numPileupBins 65  ${j}_targetpu.root
done





