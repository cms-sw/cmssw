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
if (-d ${NEW_VERS}_vs_${OLD_VERS}) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${NEW_VERS}_vs_${OLD_VERS}
cd ${NEW_VERS}_vs_${OLD_VERS}

mkdir CaloTowers
mkdir Digis
mkdir Noise+DB
mkdir RecHits
mkdir SimHits


#Create lower directories and distribute html files
mkdir CaloTowers/HB
mkdir CaloTowers/HE
mkdir CaloTowers/HF

cp ../html_indices/CaloTowers_HB.html CaloTowers/HB/index.html
cp ../html_indices/CaloTowers_HE.html CaloTowers/HE/index.html
cp ../html_indices/CaloTowers_HF.html CaloTowers/HF/index.html

mkdir Digis/HB
mkdir Digis/HE
mkdir Digis/HF
mkdir Digis/HF_gamma
mkdir Digis/HO

cp ../html_indices/Digis_HB.html Digis/HB/index.html
cp ../html_indices/Digis_HE.html Digis/HE/index.html
cp ../html_indices/Digis_HF.html Digis/HF/index.html
cp ../html_indices/Digis_HO.html Digis/HO/index.html
cp ../html_indices/Digis_HF_gamma.html Digis/HF_gamma/index.html

mkdir RecHits/HB
mkdir RecHits/HE
mkdir RecHits/HF
mkdir RecHits/HF_gamma
mkdir RecHits/HO
mkdir RecHits/ALL

cp ../html_indices/RecHits_HB.html RecHits/HB/index.html
cp ../html_indices/RecHits_HE.html RecHits/HE/index.html
cp ../html_indices/RecHits_HF.html RecHits/HF/index.html
cp ../html_indices/RecHits_HO.html RecHits/HO/index.html
cp ../html_indices/RecHits_HF_gamma.html RecHits/HF_gamma/index.html
cp ../html_indices/RecHits_Global.html RecHits/ALL/index.html

mkdir Noise+DB/DB
mkdir Noise+DB/Noise_ZS
mkdir Noise+DB/Noise_NZS

cp ../html_indices/Noise+DB_DB.html Noise+DB/DB/index.html
cp ../html_indices/Noise+DB_ZS.html Noise+DB/Noise_ZS/index.html
cp ../html_indices/Noise+DB_NZS.html Noise+DB/Noise_NZS/index.html

cd ../

#CaloTowers
root -l -q 'CombinedCaloTowers.C("'${OLD_VERS}'","'${NEW_VERS}'")'

mv HB_*gif ${NEW_VERS}_vs_${OLD_VERS}/CaloTowers/HB/
mv HE_*gif ${NEW_VERS}_vs_${OLD_VERS}/CaloTowers/HE/
mv HF_*gif ${NEW_VERS}_vs_${OLD_VERS}/CaloTowers/HF/

#Digis + DB
root -l -q 'CombinedDigis.C("'${OLD_VERS}'","'${NEW_VERS}'")'

mv HcalDigiTask_*HB.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HB/
mv HcalDigiTask_*HE.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HE/
mv HcalDigiTask_*HF.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HF/
mv HcalDigiTask_*HO.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HO/
mv HcalDigiTask_*HF_gamma.gif ${NEW_VERS}_vs_${OLD_VERS}/Digis/HF_gamma/

mv *_gain*gif ${NEW_VERS}_vs_${OLD_VERS}/Noise+DB/DB/
mv *ped_cap*gif ${NEW_VERS}_vs_${OLD_VERS}/Noise+DB/DB/
mv *pedwidth_cap*gif ${NEW_VERS}_vs_${OLD_VERS}/Noise+DB/DB/

#RecHits
root -l -q 'CombinedRecHits.C("'${OLD_VERS}'","'${NEW_VERS}'")'

mv HcalRecHitTask_En_rechits_cone_profile_vs_eta_*_HF.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/HF_gamma/
mv HcalRecHitTask_*_HF_gamma.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/HF_gamma/
mv HcalRecHitTask_En_rechits_cone_profile_vs_eta_depth*gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/ALL/
mv HcalRecHitTask_energy_*_G.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/ALL/

mv HcalRecHitTask_*_HB.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/HB/
mv HcalRecHitTask_*_HE.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/HE/
mv HcalRecHitTask_*_HF.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/HF/
mv HcalRecHitTask_*_HO.gif ${NEW_VERS}_vs_${OLD_VERS}/RecHits/HO/

mv *NZS.gif ${NEW_VERS}_vs_${OLD_VERS}/Noise+DB/Noise_NZS/

mv HcalRecHits_*gif ${NEW_VERS}_vs_${OLD_VERS}/Noise+DB/Noise_ZS/
mv N_*gif ${NEW_VERS}_vs_${OLD_VERS}/Noise+DB/Noise_ZS/

#mv ${NEW_VERS}_vs_${OLD_VERS} /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/HCAL/

exit

