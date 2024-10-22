#!/bin/sh
nopu_new="1400_140X_mcRun3_2024_realistic_v3_STD_2024_noPU-v1"
nopu_old="1400pre3_140X_mcRun3_2024_realistic_v1_STD_2024_noPU-v1"
pu_new="1400_140X_mcRun3_2024_realistic_v3_RecoOnly_2024_PU-v1"
pu_old="1400pre3_PU_140X_mcRun3_2024_realistic_v1_STD_2024_PU-v1"

./RunRVMacros_2024.csh  $nopu_new $nopu_old
./RunRVMacros_Pileup2024.csh $pu_new $pu_old
rsync -av "${nopu_new}_vs_${nopu_old}_RelVal" aramayis@lxplus.cern.ch:/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
rsync -av "${pu_new}_vs_${pu_old}_RelVal_PileUp" aramayis@lxplus.cern.ch:/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
