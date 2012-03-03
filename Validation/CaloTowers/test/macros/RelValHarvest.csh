#!/bin/tcsh

# This script retrieves the harvested RelVal files from the web and renames them to the standard HCAL validation format
# It takes two arguments: the file with the list of centrally produced RelVals (sent with every email announcing the 
# samples) and the label for this set of RelVals. Note that you have set up the LCG UI and initialize the voms proxy
# before using this script or access to the webpage will be denied. That is, do something like:
#
# source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh
# voms-proxy-init -voms cms
#
# Caveat: the script relies on the format of both the file names and RelVal sample list file being the same and may
# need modification in the future. In particular, when running with CMSSW versions that are not 39X, replace
# "CMSSW_3_9_x" in the https address with the correct version.

#Check for correct number of arguments
if ($#argv<2) then
    echo "Script needs 2 input variables"
    exit
endif

set filein=$1
set label=$2

cat $filein | grep RelValTTbar            | grep "GEN-SIM-RECO"              >  temp.tmp
cat $filein | grep RelValQCD_Pt_80_120    | grep "GEN-SIM-RECO"              >> temp.tmp
cat $filein | grep RelValQCD_Pt_3000_3500 | grep "GEN-SIM-RECO" | grep MC    >> temp.tmp
cat $filein | grep RelValMinBias          | grep "GEN-SIM-RECO" | grep START >> temp.tmp

cat temp.tmp | sed -e 's%|/%mytempstring%g' | sed -e 's/|//g' | sed -e s/" 9000 "//g | sed -e s/" 0 "//g | sed -e s/True//g | sed -e 's%/%__%g' | sed -e 's%mytempstring%/usr/bin/curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_3_9_x/DQM_V0001_R000000001__%g' | sed -e 's/GEN-SIM-RECO/GEN-SIM-RECO.root/g' > getDQMFiles_${label}.csh

cat temp.tmp | sed -e 's%|/%%g' | sed -e 's/|//g' | sed -e s/" 9000 "//g | sed -e s/" 0 "//g | sed -e s/True//g | sed -e 's%/%__%g' | sed -e 's/GEN-SIM-RECO/GEN-SIM-RECO.root/g' > temp1.tmp

cat temp1.tmp | grep TTbar       | grep MC    | sed -e s/RelVal/"mv DQM_V0001_R000000001__RelVal"/ | sed -e s/root/"root HcalRecHitValidationRelVal_TTbar_MC_${label}.root"/        >  moveDQMFiles_${label}.csh
cat temp1.tmp | grep TTbar       | grep START | sed -e s/RelVal/"mv DQM_V0001_R000000001__RelVal"/ | sed -e s/root/"root HcalRecHitValidationRelVal_TTbar_Startup_${label}.root"/   >> moveDQMFiles_${label}.csh
cat temp1.tmp | grep QCD_Pt_80   | grep MC    | sed -e s/RelVal/"mv DQM_V0001_R000000001__RelVal"/ | sed -e s/root/"root HcalRecHitValidationRelVal_QCD_MC_${label}.root"/          >> moveDQMFiles_${label}.csh
cat temp1.tmp | grep QCD_Pt_80   | grep START | sed -e s/RelVal/"mv DQM_V0001_R000000001__RelVal"/ | sed -e s/root/"root HcalRecHitValidationRelVal_QCD_Startup_${label}.root"/     >> moveDQMFiles_${label}.csh
cat temp1.tmp | grep QCD_Pt_3000 | grep MC    | sed -e s/RelVal/"mv DQM_V0001_R000000001__RelVal"/ | sed -e s/root/"root HcalRecHitValidationRelVal_HighPtQCD_MC_${label}.root"/   >> moveDQMFiles_${label}.csh
cat temp1.tmp | grep MinBias     | grep START | sed -e s/RelVal/"mv DQM_V0001_R000000001__RelVal"/ | sed -e s/root/"root HcalRecHitValidationRelVal_MinBias_Startup_${label}.root"/ >> moveDQMFiles_${label}.csh

chmod +x getDQMFiles_${label}.csh
chmod +x moveDQMFiles_${label}.csh

source getDQMFiles_${label}.csh
source moveDQMFiles_${label}.csh

rm temp.tmp
rm temp1.tmp


