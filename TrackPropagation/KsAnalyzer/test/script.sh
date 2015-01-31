#!/bin/bash

#$1 templatecfg
#$2 job number

afsdir=/afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/test

cd $afsdir
eval `scram runtime -sh`
#cd -

basename=KsAnalysis_75
output_dir=/afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/test

export X509_USER_PROXY=/afs/cern.ch/user/l/lviliani/proxy.proxy

cmsRun ${afsdir}/${basename}_cfg.py &> ${afsdir}/${basename}.log
