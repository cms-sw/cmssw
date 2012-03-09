#!/bin/bash
CONDITIONS=FrontierConditions_GlobalTag,START42_V13::All

cmsDriver.py \
	-s NONE \
	 --no_exec \
	--conditions=${CONDITIONS} \
	--fileout=skimmed.root \
	--eventcontent=AODSIM \
	--python_filename=skim.py \
	--customise=TauAnalysis/MCEmbeddingTools/ZmumuStandaloneSelectionAll.py
