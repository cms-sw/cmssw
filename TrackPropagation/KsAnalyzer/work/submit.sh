#!/bin/bash

#$1 templatecfg

for i in `seq 1 92`; do
  echo "submitting  job $i"
  bsub -q 2nd -o /afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/output_KsAnalysis3 "script.sh $1 $i"

done  
