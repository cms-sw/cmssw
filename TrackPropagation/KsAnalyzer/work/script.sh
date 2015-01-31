#!/bin/bash

#$1 templatecfg
#$2 job number

afsdir=/afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer

cd $afsdir
eval `scram runtime -sh`
#cd -

basename=KsAnalysis_${2} 
output_dir=/afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/output_KsAnalysis3

file=`sed -n ${2}p fileList.txt`

cat ${afsdir}/${1} | sed -e"s#JOBNUMBER#${2}#g" | sed -e"s#FILE#${file}#g" > ${output_dir}/config/${basename}_cfg.py

export X509_USER_PROXY=/afs/cern.ch/user/l/lviliani/proxy.proxy

cmsRun ${output_dir}/config/${basename}_cfg.py &> ${output_dir}/err/${basename}.log
