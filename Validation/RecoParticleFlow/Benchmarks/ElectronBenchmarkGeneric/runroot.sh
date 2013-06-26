#!/bin/sh

root -b plot.C

#save the plots
mkdir -p $DBS_RELEASE/$DBS_SAMPLE$E_SELECTION
cp Plots_BarrelAndEndcap/* $DBS_RELEASE/$DBS_SAMPLE$E_SELECTION/.
cp benchmark.root $DBS_RELEASE/$DBS_SAMPLE$E_SELECTION/.

