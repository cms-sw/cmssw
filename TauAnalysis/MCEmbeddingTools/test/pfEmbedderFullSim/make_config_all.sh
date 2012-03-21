#!/bin/bash
#CONDITIONS=FrontierConditions_GlobalTag,MC_3XY_V26::All
#CONDITIONS=FrontierConditions_GlobalTag,START3X_V26::All
#CONDITIONS=FrontierConditions_GlobalTag,START38_V12::All
#CONDITIONS=FrontierConditions_GlobalTag,START311_V2::All
#CONDITIONS=FrontierConditions_GlobalTag,START42_V13::All
CONDITIONS=FrontierConditions_GlobalTag,START52_V5::All

cmsDriver.py TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,RECO \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded.root  \
       --python_filename=embed.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customizeAll.py \
       -n 10



