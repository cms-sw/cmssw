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
cd $CMSSW_BASE/src/Validation/CaloTowers/test/macros

#Check if base directory already exists
if (-d ${NEW_VERS}_vs_${OLD_VERS}_RelVal) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${NEW_VERS}_vs_${OLD_VERS}_RelVal
cd ${NEW_VERS}_vs_${OLD_VERS}_RelVal

cp ../html_indices/TopLevelRelVal.html index.html

#TTbar
mkdir TTbar
mkdir TTbar/CalTowHB
mkdir TTbar/CalTowHE
mkdir TTbar/CalTowHF
mkdir TTbar/RecHits

cp ../html_indices/RelVal_RecHits_TTBar.html TTbar/RecHits/index.html
cp ../html_indices/CaloTowers_HB.html        TTbar/CalTowHB/index.html
cp ../html_indices/CaloTowers_HE.html        TTbar/CalTowHE/index.html
cp ../html_indices/CaloTowers_HF.html        TTbar/CalTowHF/index.html

#QCD
mkdir QCD
mkdir QCD/CalTowHB
mkdir QCD/CalTowHE
mkdir QCD/CalTowHF
mkdir QCD/RecHits

cp ../html_indices/RelVal_RecHits_QCD.html QCD/RecHits/index.html
cp ../html_indices/CaloTowers_HB.html      QCD/CalTowHB/index.html
cp ../html_indices/CaloTowers_HE.html      QCD/CalTowHE/index.html
cp ../html_indices/CaloTowers_HF.html      QCD/CalTowHF/index.html

cd ../

if (-d emean_seq_${NEW_VERS}_vs_${OLD_VERS}) then
     rm -rf emean_seq_${NEW_VERS}_vs_${OLD_VERS}
endif

mkdir emean_seq_${NEW_VERS}_vs_${OLD_VERS}
mkdir emean_seq_${NEW_VERS}_vs_${OLD_VERS}/TTbar
mkdir emean_seq_${NEW_VERS}_vs_${OLD_VERS}/QCD

#Process TTbar
root -l -q 'RelValMacro.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_TTbar_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_TTbar_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbar/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbar/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbar/CalTowHF/
mv emean_seq_*.gif      emean_seq_${NEW_VERS}_vs_${OLD_VERS}/TTbar
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbar/RecHits/

#Process QCD
root -l -q 'RelValMacro.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_QCD_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_QCD_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCD/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCD/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCD/CalTowHF/
mv emean_seq_*.gif      emean_seq_${NEW_VERS}_vs_${OLD_VERS}/QCD
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCD/RecHits/

exit
