#!/bin/env tcsh

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

set VAL_VERS=$1
set REF_VERS=$2

if ($#argv>3) then
    set VAL_FILE=$3
    set REF_FILE=$4
else
    set VAL_FILE=${VAL_VERS}.root
    set REF_FILE=${REF_VERS}.root   
endif

#Go to CaloTowers test directory
cd $CMSSW_BASE/src/Validation/CaloTowers/test/macros/

#Check if base directory already exists
if (-d ${VAL_VERS}_vs_${REF_VERS}_RelVal) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${VAL_VERS}_vs_${REF_VERS}_RelVal
cd ${VAL_VERS}_vs_${REF_VERS}_RelVal

cp ../html_indices/TopLevelOneSample.html index.html

#MinBias
mkdir CaloTowers
mkdir RecHits
mkdir RBX

cp ../html_indices/RelVal_RecHits.html    RecHits/index.html
cp ../html_indices/RelVal_CaloTowers.html CaloTowers/index.html
cp ../html_indices/RBX.html               RBX/index.html

cd ../

#Process MC MinBias
root -b -q 'RelValMacro.C("'${REF_VERS}'","'${VAL_VERS}'","'${REF_FILE}'","'${VAL_FILE}'","InputRelVal_Low.txt")'

mv *CaloTowers*.gif ${VAL_VERS}_vs_${REF_VERS}_RelVal/CaloTowers/
mv RBX*gif          ${VAL_VERS}_vs_${REF_VERS}_RelVal/RBX/
mv *gif             ${VAL_VERS}_vs_${REF_VERS}_RelVal/RecHits/

exit
