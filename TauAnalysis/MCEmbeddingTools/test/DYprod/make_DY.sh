#!/bin/bash
CONDITIONS=FrontierConditions_GlobalTag,START42_V13::All

# mix_E7TeV_Ave25_50ns_PoissonOOTPU_cfi.py -> E7TeV_Ave25_50ns_PoissonOOTPU

cd $CMSSW_BASE/src/

f1=Configuration/GenProduction/python/DYToMuMu_M_20_TuneZ2_7TeV_pythia6_cff.py
if [ ! -f $f1 ]; then
  cvs co -r 1.3 $f1
fi

f2=Configuration/GenProduction/python/DYToTauTau_M_20_TuneZ2_7TeV_pythia6_tauola_cff.py
if [ ! -f $f2 ]; then
  cvs co -r 1.1 $f2
fi

cd -

cmsDriver.py  TauAnalysis/MCEmbeddingTools/python/DYToMuMu_TMF \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,RECO \
       --no_exec \
       --pileup=mix_E7TeV_Fall2011ReDigi_prelim_50ns_PoissonOOT_cfi \
       --conditions=${CONDITIONS} \
       --fileout=ZmumuTF.root  \
       --python_filename=DYmumuTF.py \
       --customise=TauAnalysis/MCEmbeddingTools/customizePU.py \
       -n 10


cmsDriver.py  TauAnalysis/MCEmbeddingTools/python/DYToTauTau_TMF \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,RECO \
       --no_exec \
       --pileup=mix_E7TeV_Fall2011ReDigi_prelim_50ns_PoissonOOT_cfi \
       --conditions=${CONDITIONS} \
       --fileout=ZtautauTF.root  \
       --python_filename=DYtautauTF.py \
       --customise=TauAnalysis/MCEmbeddingTools/customizePU.py \
       -n 10



