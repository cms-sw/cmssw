#!/bin/sh
eval `scramv1 ru -sh`

Extension=`echo $DBS_SAMPLE$E_SELECTION | tr '_' '-'`
if [ ! -z "$?DBS_COMPARE_RELEASE" ] ; then
  if [ ! "$DBS_COMPARE_RELEASE" = "" ] ; then
    ../Tools/indexGenCompare.py $DBS_COMPARE_RELEASE/ElectronBenchmarkGeneric_$Extension $DBS_RELEASE/ElectronBenchmarkGeneric_$Extension -m plot.C -f -S
    exit
  fi
fi
