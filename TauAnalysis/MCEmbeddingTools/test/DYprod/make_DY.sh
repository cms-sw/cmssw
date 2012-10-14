#!/bin/bash
CONDITIONS=FrontierConditions_GlobalTag,START53_V7A::All

# mix_E7TeV_Ave25_50ns_PoissonOOTPU_cfi.py -> E7TeV_Ave25_50ns_PoissonOOTPU

#PU=E7TeV_FlatDist10_2011EarlyData_50ns_PoissonOOT  # 42x
#PU=mix_E7TeV_Fall2011ReDigi_prelim_50ns_PoissonOOT_cfi
PU=NoPileUp

cmsDriver.py  TauAnalysis/MCEmbeddingTools/python/DYToMuMu_TMF \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,L1Reco,RECO \
       --no_exec \
       --pileup=$PU \
       --conditions=${CONDITIONS} \
       --fileout=ZmumuTF.root  \
       --python_filename=DYmumuTF.py \
       -n 10 
#       --customise=TauAnalysis/MCEmbeddingTools/customizePU.py 
exit

cmsDriver.py  TauAnalysis/MCEmbeddingTools/python/DYToTauTau_TMF \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,L1Reco,RECO \
       --no_exec \
       --pileup=$PU \
       --conditions=${CONDITIONS} \
       --fileout=ZtautauTF.root  \
       --python_filename=DYtautauTF.py \
       -n 10 
#       --customise=TauAnalysis/MCEmbeddingTools/customizePU.py 



