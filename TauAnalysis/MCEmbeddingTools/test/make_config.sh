#!/bin/bash
CONDITIONS=FrontierConditions_GlobalTag,MC_31X_V3::All


#
cmsDriver.py TauAnalysis/MCEmbeddingTools/empty_cfi.py \
		-s RAW2DIGI,RECO \
		--fileout=/tmp/output_01.root \
		--filein=file:/storage/6/zeise/temp/Zmumu_Summer09_7TeV_FCF4160D-C5BD-DE11-B41F-003048982851.root \
		--no_exec --python_filename=01_grid_reco_skim_select_zmumu.py -n 100 \
		--conditions ${CONDITIONS} \
		--eventcontent=FEVTDEBUG \
		--customise=TauAnalysis/MCEmbeddingTools/01_customise_select_zmumu_and_replace.py


cmsDriver.py \
        TauAnalysis/MCEmbeddingTools/empty_cfi.py \
        --fileout=/tmp/output_11.root \
        --filein=file:/tmp/output_01.root  \
        --no_exec \
        --beamspot NoSmear \
        --python_filename=11_local_simulate_new_partial_event.py \
        -n 100 \
        --conditions ${CONDITIONS} \
        -s SIM,DIGI,L1,DIGI2RAW,RAW2DIGI \
        --eventcontent=FEVTDEBUG \
        --customise=TauAnalysis/MCEmbeddingTools/11_customise_simulate_new_partial_event.py   

cmsDriver.py \
        TauAnalysis/MCEmbeddingTools/empty_cfi.py \
        --fileout=/tmp/output_12.root \
        --filein=file:/tmp/output_11.root  \
        --no_exec \
        --beamspot NoSmear \
        --python_filename=12_local_overlay_both_events.py \
        -s L1 \
        -n 100 \
        --conditions ${CONDITIONS} \
        --eventcontent=FEVTDEBUG \
        --mc \
        --customise=TauAnalysis/MCEmbeddingTools/12_customise_overlay_both_events.py

cmsDriver.py \
        TauAnalysis/MCEmbeddingTools/empty_cfi.py \
        --fileout=/tmp/output_23.root \
        --filein=file:/tmp/output_12.root  \
        --no_exec \
        --beamspot NoSmear \
        --python_filename=23_local_hlt_overlay_event.py \
        -s HLT \
        -n 100 \
        --conditions ${CONDITIONS} \
        --eventcontent=FEVTDEBUG \
        --customise=TauAnalysis/MCEmbeddingTools/23_customise_hlt_on_overlay_event.py

cmsDriver.py \
        TauAnalysis/MCEmbeddingTools/empty_cfi.py \
        --fileout=/tmp/output_24.root \
        --filein=file:/tmp/output_23.root  \
        --no_exec \
        --beamspot NoSmear \
        --python_filename=24_local_reconstruct_overlay_event.py \
        -s RECO \
        -n 100 \
        --mc \
        --conditions ${CONDITIONS} \
        --eventcontent=FEVTDEBUG \
        --customise=TauAnalysis/MCEmbeddingTools/24_customise_reconstruct_overlay_event.py   
