#! /bin/bash

current_area=`pwd`
echo $current_area

dirlist="ZDimu ZprimeDijets QCD_0-15 QCD_15-20 QCD_20-30 QCD_30-50 QCD_50-80 QCD_80-120 QCD_120-170 QCD_170-230 QCD_230-300 QCD_300-380 QCD_380-470 QCD_470-600 QCD_600-800 QCD_800-1000"

cd $current_area

eval `scramv1 runtime -sh`

for i in $dirlist; do
echo "Currently running over sample: $i"
cmsRun RunAnalyzers-$i.cfg > LOG-$i 2> Err-$i 
cd $current_area
done
