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
if (-d ${NEW_VERS}_vs_${OLD_VERS}_RelVal) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${NEW_VERS}_vs_${OLD_VERS}_RelVal
cd ${NEW_VERS}_vs_${OLD_VERS}_RelVal

cp ../../html_indices/MinBiasTopLevel.html index.html

#MinBias
mkdir MinBias
mkdir MinBias/CalTowHB
mkdir MinBias/CalTowHE
mkdir MinBias/CalTowHF
mkdir MinBias/RecHits

cp ../../html_indices/RelVal_RecHits_QCD.html MinBias/RecHits/index.html
cp ../../html_indices/CaloTowers_HB.html      MinBias/CalTowHB/index.html
cp ../../html_indices/CaloTowers_HE.html      MinBias/CalTowHE/index.html
cp ../../html_indices/CaloTowers_HF.html      MinBias/CalTowHF/index.html

cd ../

#Process MC MinBias
root -l -q 'RelValMacro.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_MinBias_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_MinBias_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/MinBias/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/MinBias/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/MinBias/CalTowHF/
rm emean_seq_*.gif
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/MinBias/RecHits/


exit
