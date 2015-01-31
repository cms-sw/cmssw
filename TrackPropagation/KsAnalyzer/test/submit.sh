#!/bin/bash

#$1 templatecfg

  bsub -q 2nd -o /afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/test/out75.log "script.sh $1"
