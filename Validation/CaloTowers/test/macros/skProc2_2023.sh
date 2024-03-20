#!/bin/sh
nopu_new="1400pre2_133X_mcRun3_2023_realistic_v3_STD-v1"
nopu_old="1400pre1_133X_mcRun3_2023_realistic_v3-v1"
pu_new="1400pre2_PU_133X_mcRun3_2023_realistic_v3_STD_PU-v3"
pu_old="1400pre1_PU_133X_mcRun3_2023_realistic_v3-v1"

./RunRVMacros_2023.csh $nopu_new  $nopu_old
./RunRVMacros_Pileup2023.csh $pu_new  $pu_old
rsync -av "${nopu_new}_vs_${nopu_old}_RelVal" ykazhyka@lxplus.cern.ch:/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
rsync -av "${pu_new}_vs_${pu_old}_RelVal_PileUp" ykazhyka@lxplus.cern.ch:/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
