#!/bin/bash

#run merging config
cmsRun merging_cfg.py

echo "DQM file produced"
file="$(ls | grep -i "DQM")"
echo $file

#clean directory
rm -r pi50_*.py *.log LSFJOB_* pi50_*.root conf.py mc.root
echo "Changing the name of DQM file"
if [ "$#" -ne 1 ]; then
    echo "Give One version name for DQM root file"
    exit
fi

ThisVERS=$1
mv ${file} pi50scan${ThisVERS}_ECALHCAL_CaloTowers.root
echo "New DQM file name:"
echo "pi50scan${ThisVERS}_ECALHCAL_CaloTowers.root"
echo "Moving DQM file into macros/"

mv pi50scan${ThisVERS}_ECALHCAL_CaloTowers.root ../macros/
