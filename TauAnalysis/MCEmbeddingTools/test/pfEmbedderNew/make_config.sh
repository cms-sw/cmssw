#!/bin/bash
#CONDITIONS=FrontierConditions_GlobalTag,MC_3XY_V26::All
CONDITIONS=FrontierConditions_GlobalTag,START3X_V26::All


cmsDriver.py TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff \
       -s GEN:ProductionFilterSequence,FASTSIM \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded.root \
       --python_filename=embed.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customizeSimulation.py

