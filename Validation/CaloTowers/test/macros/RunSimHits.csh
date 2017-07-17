#!/bin/tcsh

#Check to see if the CMS environment is set up
if ($?CMSSW_BASE != 1) then
    echo "CMS environment not set up"
#    exit
endif

#Check for correct number of arguments
if ($#argv<2) then
    echo "Script needs 2 input variable"
#    exit
endif

set NEW_VERS=$1
set OLD_VERS=$2

# Two bit value with the first corresponding to whether the validation version is centrally
# harvested (1) or not (0) and the second to whether the reference version is harvested. Thus:
# 00: both are privately produced
# 01: reference version is harvested, validation version is private
# 10: validation version is harvested, reference version is private
# 11: both versions are harvested
# Any other value is the same as 0
set harvest=11

#Check if base directory already exists
#if (-d ${NEW_VERS}_vs_${OLD_VERS}_RelVal) then
#    echo "Directory already exists"
#    exit
#endif

#Create base directory and top directories
mkdir -p ${NEW_VERS}_vs_${OLD_VERS}_RelVal
cd ${NEW_VERS}_vs_${OLD_VERS}_RelVal

cp ../html_indices/TopLevelRelValSimHits.html index.html

#MinBias
mkdir -p MinBias
mkdir -p MinBias/SimHits
cp ../html_indices/RelVal_Simhits.html MinBias/SimHits/index.html

#QCD
mkdir -p QCD
mkdir -p QCD/SimHits
cp ../html_indices/RelVal_Simhits.html QCD/SimHits/index.html

#High Pt QCD
mkdir -p HighPtQCD
mkdir -p HighPtQCD/SimHits
cp ../html_indices/RelVal_Simhits.html HighPtQCD/SimHits/index.html

#TTbar
mkdir -p TTbar
mkdir -p TTbar/SimHits
cp ../html_indices/RelVal_Simhits.html TTbar/SimHits/index.html


cd ../

#Process Startup MinBias
./RelValMacro.exe ${OLD_VERS} ${NEW_VERS} HcalRecHitValidationRelVal_MinBias_${OLD_VERS}.root HcalRecHitValidationRelVal_MinBias_${NEW_VERS}.root InputRelVal_SimHits.txt
#root -b -q 'RelValMacro_SimHitsValidationHcal.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_MinBias_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_MinBias_${NEW_VERS}.root'","InputRelVal_SimHits_Low.txt",'${harvest}')'
#root -b -q 'RelValMacro_HcalSimHitsTask.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_MinBias_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_MinBias_${NEW_VERS}.root'","InputRelVal_SimHits.txt",'${harvest}')'

mv *.gif   ${NEW_VERS}_vs_${OLD_VERS}_RelVal/MinBias/SimHits


#Process Startup QCD
./RelValMacro.exe ${OLD_VERS} ${NEW_VERS} HcalRecHitValidationRelVal_QCD_${OLD_VERS}.root HcalRecHitValidationRelVal_QCD_${NEW_VERS}.root InputRelVal_SimHits.txt
#root -b -q 'RelValMacro_SimHitsValidationHcal.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_QCD_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_QCD_${NEW_VERS}.root'","InputRelVal_SimHits_Low_Free_y.txt",'${harvest}')'
#root -b -q 'RelValMacro_HcalSimHitsTask.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_QCD_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_QCD_${NEW_VERS}.root'","InputRelVal_SimHits.txt",'${harvest}')'


mv *.gif   ${NEW_VERS}_vs_${OLD_VERS}_RelVal/QCD/SimHits


#Process Startup HighPtQCD
./RelValMacro.exe ${OLD_VERS} ${NEW_VERS} HcalRecHitValidationRelVal_HighPtQCD_${OLD_VERS}.root HcalRecHitValidationRelVal_HighPtQCD_${NEW_VERS}.root InputRelVal_SimHits.txt
#root -b -q 'RelValMacro_SimHitsValidationHcal.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_HighPtQCD_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_HighPtQCD_${NEW_VERS}.root'","InputRelVal_SimHits_Low_Free_y.txt",'${harvest}')'
#root -b -q 'RelValMacro_HcalSimHitsTask.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_HighPtQCD_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_HighPtQCD_${NEW_VERS}.root'","InputRelVal_SimHits.txt",'${harvest}')'

mv *.gif   ${NEW_VERS}_vs_${OLD_VERS}_RelVal/HighPtQCD/SimHits


#Process Startup TTbar
./RelValMacro.exe ${OLD_VERS} ${NEW_VERS} HcalRecHitValidationRelVal_TTbar_${OLD_VERS}.root HcalRecHitValidationRelVal_TTbar_${NEW_VERS}.root InputRelVal_SimHits.txt
#root -b -q 'RelValMacro_SimHitsValidationHcal.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_TTbar_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_TTbar_${NEW_VERS}.root'","InputRelVal_SimHits_Low_Free_y.txt",'${harvest}')'
#root -b -q 'RelValMacro_HcalSimHitsTask.C("'${OLD_VERS}'","'${NEW_VERS}'","'HcalRecHitValidationRelVal_TTbar_${OLD_VERS}.root'","'HcalRecHitValidationRelVal_TTbar_${NEW_VERS}.root'","InputRelVal_SimHits.txt",'${harvest}')'

mv *.gif   ${NEW_VERS}_vs_${OLD_VERS}_RelVal/TTbar/SimHits

exit
