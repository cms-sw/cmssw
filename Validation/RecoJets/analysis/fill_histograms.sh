#!/bin/bash
for input in $*
  do
    if [ -n "$input_file" ] 
    then input_file=$input_file", 'file:"$input"'"
    else input_file="'file:"$input"'"
    fi
done
echo input_file: $input_file

for flavor in midPointCone5CaloJet midPointCone7CaloJet iterativeCone5CaloJet Fastjet10CaloJet Fastjet6CaloJet 
  do
    echo Processing flavor $flavor ...
    cfg_file=$flavor.cfg
    output_file=$flavor.root
    log_file=$flavor.log
    rm -f $cfg_file $log_file
    cat template.cfg | sed s^INPUT_FILES^"$input_file"^g | sed s^OUTPUT_FILE^$output_file^g | sed s^SOURCE^$flavor^g > $cfg_file
      
    eval `scramv1 ru -sh`
    cmsRun $cfg_file > $log_file 2>&1 
done
