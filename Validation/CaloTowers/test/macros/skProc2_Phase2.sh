#!/bin/sh
nopu_new="1400pre2_133X_mcRun4_realistic_v1_STD_2026D98_noPU-v1"
nopu_old="1400pre1_133X_mcRun4_realistic_v1_2026D98noPU-v1"
pu_new="1400pre2_PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200-v3"
pu_old="1400pre1_PU_133X_mcRun4_realistic_v1_2026D98PU200-v1"

./RunRVMacrosPhase2.csh  $nopu_new $nopu_old 
./RunRVMacros_PileupPhase2.csh $pu_new $pu_old
rsync -av "${nopu_new}_vs_${nopu_old}_RelVal" ykazhyka@lxplus.cern.ch:/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
rsync -av "${pu_new}_vs_${pu_old}_RelVal_PileUp" ykazhyka@lxplus.cern.ch:/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
