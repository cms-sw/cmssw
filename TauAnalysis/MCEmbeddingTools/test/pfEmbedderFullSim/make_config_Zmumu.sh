#!/bin/bash
#CONDITIONS=FrontierConditions_GlobalTag,MC_3XY_V26::All
#CONDITIONS=FrontierConditions_GlobalTag,START3X_V26::All
#CONDITIONS=FrontierConditions_GlobalTag,START38_V13::All
CONDITIONS=FrontierConditions_GlobalTag,START311_V1G1::All

cmsDriver.py TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff \
       -s GEN:ProductionFilterSequence,SIM,DIGI,L1,DIGI2RAW,HLT:GRun \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded_HLT.root  \
       --python_filename=embed_HLT.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customize_HLT.py \
       -n -1

cmsDriver.py embedded_HLT \
       --filein="file:embedded_HLT.root" \
       -s RAW2DIGI,RECO \
       --no_exec \
       --conditions=${CONDITIONS} \
       --fileout=embedded_RECO.root \
       --python_filename=embed_RECO.py \
       --customise=TauAnalysis/MCEmbeddingTools/pf_01_customizeSimulation.py \
       -n -1

# preparation for combined grid jobs:
if [ -f myFragmentForCMSSW.py ] 
then
	cat embed_HLT.py myFragmentForCMSSW.py > embed_HLT_Zmumu_singleStep.py
fi

cp embed_RECO.py embed_RECO_Zmumu_singleStep.py
cat >> embed_RECO_Zmumu_singleStep.py << EOF

import os
process.source.fileNames=cms.untracked.vstring()
for f in os.listdir("."):
	if f.startswith("embedded_HLT") and f.endswith(".root"):
		process.source.fileNames.append("file:%s" % (f))
print process.source.fileNames

# do not remove ;)
# __MAX_EVENTS__
# __FILE_NAMES__
# __SKIP_EVENTS__
# __MAX_EVENTS__
# __FILE_NAMES__
# __SKIP_EVENTS__
# __FILE_NAMES2__
# __LUMI_RANGE__
# __MY_JOBID__
EOF

