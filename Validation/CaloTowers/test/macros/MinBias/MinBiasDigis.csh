#!/bin/tcsh

#Check to see if the CMS environment is set up
if ($?CMSSW_BASE != 1) then
    echo "CMS environment not set up"
    exit
endif

#Check for correct number of arguments
if ($#argv<2) then
    echo "Script needs 2 input variable"
    exit
endif

set NEW_VERS=$1
set OLD_VERS=$2

#Go to CaloTowers test directory
cd $CMSSW_BASE/src/Validation/CaloTowers/test/macros/MinBias

#Check if base directory already exists
if (-d ${NEW_VERS}_vs_${OLD_VERS}) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${NEW_VERS}_vs_${OLD_VERS}
cd ${NEW_VERS}_vs_${OLD_VERS}

mkdir Digis

#Create lower directories and distribute html files
mkdir Digis/HB
mkdir Digis/HE
mkdir Digis/HF
mkdir Digis/HO

cp ../../html_indices/Digis_HB.html Digis/HB/index.html
cp ../../html_indices/Digis_HE.html Digis/HE/index.html
cp ../../html_indices/Digis_HF.html Digis/HF/index.html
cp ../../html_indices/Digis_HO.html Digis/HO/index.html

cp ../../html_indices/DigisTopLevel.html index.html

cd ../

#Digis + DB
root -l -q 'CombinedDigis.C("'${OLD_VERS}'","'${NEW_VERS}'")'

mv HcalDigiTask_*HB.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HB/
mv HcalDigiTask_*HE.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HE/
mv HcalDigiTask_*HF.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HF/
mv HcalDigiTask_*HO.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HO/


exit

