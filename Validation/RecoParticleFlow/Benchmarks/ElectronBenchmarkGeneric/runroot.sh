#!/bin/sh
eval `scramv1 ru -sh`
root -l plot.C

#save the plots
cp -r Plots_BarrelAndEndcap $DBS_RELEASE/$DBS_SAMPLE
cp benchmark.root $DBS_RELEASE/$DBS_SAMPLE/.
