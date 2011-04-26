#!/bin/bash
#CONDITIONS=FrontierConditions_GlobalTag,MC_3XY_V26::All
#CONDITIONS=FrontierConditions_GlobalTag,START3X_V26::All
#CONDITIONS=FrontierConditions_GlobalTag,GR_R_36X_V12B::All
#CONDITIONS=FrontierConditions_GlobalTag,START36_V10::All
CONDITIONS=FrontierConditions_GlobalTag,START311_V1G1::All

cmsDriver.py TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_Wemb_cff.py \
       -s GEN:ProductionFilterSequence,SIM,DIGI,L1,DIGI2RAW,HLT \
       --no_exec \
       -n -1 \
       --conditions=${CONDITIONS} \
       --fileout=embedded_HLT.root  \
       --python_filename=embed_HLT.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customize_HLT.py


cmsDriver.py \
       -s RAW2DIGI,RECO \
       --no_exec \
       -n -1 \
       --conditions=${CONDITIONS} \
       --fileout=embedded_RECO.root \
       --python_filename=embed_RECO.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customizeSimulation.py

