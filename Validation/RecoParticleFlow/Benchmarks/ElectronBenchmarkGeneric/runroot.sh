#!/bin/sh
eval `scramv1 ru -sh`
#check if it is single mode or comparison mode
if [ ! -z "$?DBS_COMPARE_RELEASE" ] ; then
  if [ ! "$DBS_COMPARE_RELEASE" = "" ] ; then
    ../Tools/indexGenCompare.py $DBS_COMPARE_RELEASE/ElectronBenchmarkGeneric_$DBS_SAMPLE$E_SELECTION $DBS_RELEASE/ElectronBenchmarkGeneric_$DBS_SAMPLE$E_SELECTION -m plot.C
    echo "Leaving"
    exit
    echo "Did not leave"
  fi
fi

root -b plot.C

#save the plots
mkdir -p $DBS_RELEASE/$DBS_SAMPLE$E_SELECTION
cp Plots_BarrelAndEndcap/* $DBS_RELEASE/$DBS_SAMPLE$E_SELECTION/.
cp benchmark.root $DBS_RELEASE/$DBS_SAMPLE$E_SELECTION/.
