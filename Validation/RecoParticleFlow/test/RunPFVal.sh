#!/bin/bash

if [[ "$#" == "0" ]]; then
    echo "Please select type of data to analyze and run using command ./RunPFVal.sh QCD (or ZEE or TTbar, command is case sensitive)";
    exit 1;
fi

DATA=$1 

echo "Running on data " $1

eval `scramv1 r -sh`

cat pflowValidationStep1_Template_cfg.py | sed -e "s@DATA@$1@g" > pflowValidationStep1_test_cfg.py 

cmsRun pflowValidationStep1_test_cfg.py

cat pflowValidationStep2_Template_cfg.py | sed -e "s@DATA@$1@g" > pflowValidationStep2_test_cfg.py 

cmsRun pflowValidationStep2_test_cfg.py
