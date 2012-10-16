#!/bin/bash
CONDITIONS=FrontierConditions_GlobalTag,START42_V13::All
#CONDITIONS=FrontierConditions_GlobalTag,START52_V2A::All

# mix_E7TeV_Ave25_50ns_PoissonOOTPU_cfi.py -> E7TeV_Ave25_50ns_PoissonOOTPU

PU=NoPileUp


cd $CMSSW_BASE/src/

f1=Configuration/GenProduction/python/DYToMuMu_M_20_TuneZ2_7TeV_pythia6_cff.py
if [ ! -f $f1 ]; then
  cvs co -r 1.3 $f1
  scramv1 b
fi

cd -
#cmsDriver.py  Configuration/GenProduction/python/DYToMuMu_M_20_TuneZ2_7TeV_pythia6_cff \
cmsDriver.py  TauAnalysis/MCEmbeddingTools/python/DYToMuMu_TMF \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,RECO \
       --no_exec \
       --pileup=$PU \
       --conditions=${CONDITIONS} \
       --fileout=ZmumuTF_TMVA.root  \
       --python_filename=DYmumuTF_TMVA.py \
       -n 10 \
       --customise=TauAnalysis/MCEmbeddingTools/customize4TMVA.py 


