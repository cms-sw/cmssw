
RELEASE=CMSSW_17_0_0_pre1

dasgoclient --query="dataset=/RelValQCD_FlatPt_15_3000HS_14/${RELEASE}*mcRun3*noPU*/GEN-SIM-DIGI-RAW"
dasgoclient --query="dataset=/RelValQCD_FlatPt_15_3000HS_14/${RELEASE}*mcRun3*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValZEE_14/${RELEASE}*mcRun3*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValZMM_14/${RELEASE}*mcRun3*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValTenTau_15_500/${RELEASE}*mcRun3*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValNuGun/${RELEASE}*mcRun3*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU

dasgoclient --query="dataset=/RelValQCD_FlatPt_15_3000HS_14/${RELEASE}*mcRun4*noPU*/GEN-SIM-DIGI-RAW"
dasgoclient --query="dataset=/RelValQCD_FlatPt_15_3000HS_14/${RELEASE}*mcRun4*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValZEE_14/${RELEASE}*mcRun4*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValZMM_14/${RELEASE}*mcRun4*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValTenTau_15_500/${RELEASE}*mcRun4*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
dasgoclient --query="dataset=/RelValNuGun/${RELEASE}*mcRun4*PU*/GEN-SIM-DIGI-RAW" | grep -v noPU
