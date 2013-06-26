#!/bin/bash

###########################################################################
## This bash script is designed to be called from some sort of batch     ##
## system that does NOT have a local run area. DO NOT USE THIS TO RUN ON ##
## CONDOR.  This does NOT setup a CMSSW release.                         ##
##                                                                       ##
## You call the script:                                                  ##
## runCommand.bash directory  logfile executable [arg1 arg2...]          ##
##                                                                       ##
## cplager  090924                                                       ##
###########################################################################

## Used to run executables.
if [ -z "$3" ]; then
  echo Usage: $0 directory logname executable \[arg1 arg2...\]
  exit;
fi

unset DISPLAY

## setup variables
export DIR=$1
shift
export LOG=$1.log
shift
export EXE=$1
shift

# go to directory (which had better be in a CMSSW release)
cd $DIR

# DO NOT setup CMS environment

echo "hostname:" `hostname` > $LOG
date >> $LOG
pwd >> $LOG
echo $EXE $@ >> $LOG 2>&1
(time $EXE $@)  >> $LOG 2>&1

