#!/bin/sh
eval `scramv1 ru -sh`

if [ ! -z "$?DBS_COMPARE_RELEASE" ] ; then
  if [ ! "$DBS_COMPARE_RELEASE" = "" ] ; then
    ../Tools/indexGenCompare.py $DBS_COMPARE_RELEASE/ElectronBenchmarkGeneric_$DBS_SAMPLE$E_SELECTION $DBS_RELEASE/ElectronBenchmarkGeneric_$DBS_SAMPLE$E_SELECTION -m plot.C -f -S
    exit
  fi
fi
