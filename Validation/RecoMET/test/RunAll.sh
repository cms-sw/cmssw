#! /bin/bash

current_area=`pwd`
echo $current_area

dirlist="QCDD_80-120 QCD_3500-4000 Wjets_80-120 Wjets_3000-3500 LM1_sfts ttbar"

cd $current_area

eval `scramv1 runtime -sh`

for i in $dirlist; do
echo "Currently running over sample: $i"
cmsRun RunAnalyzers-${i}_cfg.py > LOG-$i 2> Err-$i 
cd $current_area
done
