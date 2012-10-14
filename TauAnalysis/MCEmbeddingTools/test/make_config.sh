#!/bin/bash
CONDITIONS=FrontierConditions_GlobalTag,START53_V7A::All

cmsDriver.py TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff \
       -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded.root  \
       --python_filename=embed.py \
       --customise=TauAnalysis/MCEmbeddingTools/embeddingCustomizeAll \
       -n 10



