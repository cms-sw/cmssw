#! /bin/bash

current_area=`pwd`
echo $current_area

dirlist="QCD_Pt_80_120 QCD_Pt_3000_3500 Wjet_Pt_80_120 LM1_sfts TTbar QCD_FlatPt_15_3000"

cd $current_area

eval `scramv1 runtime -sh`

for i in $dirlist; do
echo "Currently running over sample: $i"
#cmsRun RunAnalyzers-$i.cfg > LOG-$i 2> Err-$i 
cmsRun RunAnalyzers-${i}_cfg.py > LOG-$i 2> Err-$i 
cd $current_area
done
