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
if (-d ${NEW_VERS}_vs_${OLD_VERS}_RelVal) then
    echo "Directory already exists"
    exit
endif

#Create base directory and top directories
mkdir ${NEW_VERS}_vs_${OLD_VERS}_RelVal
cd ${NEW_VERS}_vs_${OLD_VERS}_RelVal

cp ../html_indices/TopLevelRelVal.html index.html

#MinBias
mkdir MinBias
mkdir MinBias/SimHits
cp ../html_indices/RelVal_Simhits.html MinBias/SimHits/index.html




cd ../

#Process Startup MinBias
root -b -q 'RelValMacro.C("'${OLD_VERS}_Startup'","'${NEW_VERS}_Startup'","'DQM_V0001_R000000002__HcalValidation__Harvesting__${OLD_VERS}.root'","'DQM_V0001_R000000002__HcalValidation__Harvesting__${NEW_VERS}.root'","InputRelVal_Low.txt",'${harvest}')'

mv *.gif   ${NEW_VERS}_vs_${OLD_VERS}_RelVal/MinBias/SimHits
c


exit
