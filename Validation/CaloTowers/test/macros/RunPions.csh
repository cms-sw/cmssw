#!/bin/tcsh

#Check to see if the CMS environment is set up
if ($?CMSSW_BASE != 1) then
    echo "CMS environment not set up"
    exit
endif

#Check for correct number of arguments
if ($#argv<2) then
    echo "Script needs at least 2 input variables"
    exit
endif

set NEW_VERS=$1
set OLD_VERS=$2

#If only 2 arguments are given, assume it is FullSim
if ($#argv<3) then
    echo "No 3rd argument given, by default this is FullSim"
    set FASTSIM=false
else
    set FASTSIM=$3
endif

#Go to CaloTowers test directory
cd $CMSSW_BASE/src/Validation/CaloTowers/test/macros

#Check if base directory already exists
if (-d ${NEW_VERS}_vs_${OLD_VERS}_RelVal) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${NEW_VERS}_vs_${OLD_VERS}_SinglePi
cd ${NEW_VERS}_vs_${OLD_VERS}_SinglePi

#Single Pions

if ($FASTSIM != true) then
    cp ../html_indices/SinglePiScan.html  index.html
else
    cp ../html_indices/SinglePiScanFastSim.html  index.html
endif

cd ../

root -b -q 'SinglePi.C("'${OLD_VERS}'","'${NEW_VERS}'",'${FASTSIM}')'
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_SinglePi/

exit
