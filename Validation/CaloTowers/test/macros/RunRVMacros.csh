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
mkdir TTbar/RBX

cat ../html_indices/RelVal_RecHits.html | sed -e s/DATA_SAMPLE/TTbar/ > TTbar/RecHits/index.html

cp ../html_indices/CaloTowers_HB.html   TTbar/CalTowHB/index.html
cp ../html_indices/CaloTowers_HE.html   TTbar/CalTowHE/index.html
cp ../html_indices/CaloTowers_HF.html   TTbar/CalTowHF/index.html
cp ../html_indices/RBX.html             TTbar/RBX/index.html

cp -r TTbar TTbarStartup
mv    TTbar TTbarMC

#QCD
mkdir QCD
mkdir QCD/CalTowHB
mkdir QCD/CalTowHE
mkdir QCD/CalTowHF
mkdir QCD/RecHits
mkdir QCD/RBX

cat ../html_indices/RelVal_RecHits.html | sed -e s/DATA_SAMPLE/QCD/ > QCD/RecHits/index.html

cp ../html_indices/CaloTowers_HB.html   QCD/CalTowHB/index.html
cp ../html_indices/CaloTowers_HE.html   QCD/CalTowHE/index.html
cp ../html_indices/CaloTowers_HF.html   QCD/CalTowHF/index.html
cp ../html_indices/RBX.html             QCD/RBX/index.html

cp -r QCD QCDStartup
mv    QCD QCDMC

#Single Pions

mkdir SinglePi50_ECAL+HCAL_Scan

cp ../html_indices/SinglePiScan.html       SinglePi50_ECAL+HCAL_Scan/index.html

cd ../

#Process MC TTbar
root -b -q 'RelValMacro.C("'${OLD_VERS}_MC'","'${NEW_VERS}_MC'","'HcalRecHitValidationRelVal_TTbar_MC_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_TTbar_MC_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarMC/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarMC/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarMC/CalTowHF/

mv RBX*gif              ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarMC/RBX/
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarMC/RecHits/

#Process MC QCD
root -b -q 'RelValMacro.C("'${OLD_VERS}_MC'","'${NEW_VERS}_MC'","'HcalRecHitValidationRelVal_QCD_MC_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_QCD_MC_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDMC/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDMC/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDMC/CalTowHF/

mv RBX*gif              ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDMC/RBX/
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDMC/RecHits/

#Process Startup TTbar
root -b -q 'RelValMacro.C("'${OLD_VERS}_Startup'","'${NEW_VERS}_Startup'","'HcalRecHitValidationRelVal_TTbar_Startup_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_TTbar_Startup_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarStartup/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarStartup/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarStartup/CalTowHF/

mv RBX*gif              ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarStartup/RBX/
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbarStartup/RecHits/

#Process Startup QCD
root -b -q 'RelValMacro.C("'${OLD_VERS}_Startup'","'${NEW_VERS}_Startup'","'HcalRecHitValidationRelVal_QCD_Startup_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_QCD_Startup_${NEW_VERS}.root'")'

mv HB_CaloTowers*HB.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDStartup/CalTowHB/
mv HE_CaloTowers*HE.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDStartup/CalTowHE/
mv HF_CaloTowers*HF.gif ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDStartup/CalTowHF/

mv RBX*gif              ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDStartup/RBX/
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCDStartup/RecHits/

root -b -q 'SinglePi.C("'${OLD_VERS}'","'${NEW_VERS}'")'
mv *gif                 ${NEW_VERS}_vs_${OLD_VERS}_RelVal/SinglePi50_ECAL+HCAL_Scan

exit
