#!/bin/bash
if [ $# -lt 1 ];then
  echo " " 
  echo "enter CMSSW version"
  echo " " 
  exit 1
fi

mkdir ~/www/dtrechit_validation/SingleMuPt100/CMSSW_$1
cp *.gif ~/www/dtrechit_validation/SingleMuPt100/CMSSW_$1/
cp DTValidation_RelVal_fromRECO_local_cfg.py ~/www/dtrechit_validation/SingleMuPt100/CMSSW_$1/DTValidation_RelVal.py
cp ~/www/dtrechit_validation/SingleMuPt100/CMSSW_3_8_1/description.txt ~/www/dtrechit_validation/SingleMuPt100/CMSSW_$1
