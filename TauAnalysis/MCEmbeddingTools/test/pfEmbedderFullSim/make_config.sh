#!/bin/bash
#CONDITIONS=FrontierConditions_GlobalTag,MC_3XY_V26::All
CONDITIONS=FrontierConditions_GlobalTag,START3X_V26::All


cmsDriver.py TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff \
       -s GEN:ProductionFilterSequence,SIM,DIGI,L1,DIGI2RAW,HLT \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded_HLT.root  \
       --python_filename=embed_HLT.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customize_HLT.py


cmsDriver.py \
       -s RAW2DIGI,RECO \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded_RECO.root \
       --python_filename=embed_RECO.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customizeSimulation.py

