#!/bin/bash

# parse BASE_DIR and START_ROOT_FILE from command line arguments
set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error when substituting
set -o pipefail # Return the exit status of the last command in the pipeline that failed
while getopts "hb:s:" option; do
	case "${option}" in
	b) BASE_DIR=${OPTARG} ;;
	s) START_ROOT_FILE=${OPTARG} ;;
	h | *)
		echo "Usage: $0 [-b BASE_DIR] [-s START_ROOT_FILE]"
		exit 1
		;;
	esac
done

#set Start root file if not done in getopts
START_ROOT_FILE=${START_ROOT_FILE:-"root://cmsdcache-kit-disk.gridka.de:1094//store/data/Run2024C/Muon0/RAW/v1/000/380/115/00000/00979445-916c-42e2-8038-428d7bd4f176.root"}

: "${BASE_DIR:=$PWD}" # Default to current directory if not set
echo "BASE_DIR: $BASE_DIR"

# Create directories for configs, root files, and logs
[ -d "$BASE_DIR" ] || mkdir -p "$BASE_DIR"

EMB_CONF_DIR=$(realpath "$BASE_DIR/configs")
ROOT_FILES_DIR=$(realpath "$BASE_DIR/root_files")

echo "EMB_CONF_DIR: $EMB_CONF_DIR"
echo "ROOT_FILES_DIR: $ROOT_FILES_DIR"
echo "START_ROOT_FILE: $START_ROOT_FILE"

[ -d "$EMB_CONF_DIR" ] || mkdir "$EMB_CONF_DIR"
[ -d "$ROOT_FILES_DIR" ] || mkdir "$ROOT_FILES_DIR"
[ -d "$LOG_FILES_DIR" ] || mkdir "$LOG_FILES_DIR"

# Selection
echo "################ Selection ################"
cmsDriver.py \
	--step RAW2DIGI,L1Reco,RECO,PAT,FILTER:TauAnalysis/MCEmbeddingTools/Selection_FILTER_cff.makePatMuonsZmumuSelection \
	--processName SELECT \
	--data \
	--scenario pp \
	--conditions auto:run3_data \
	--era Run3_2024 \
	--eventcontent TauEmbeddingSelection \
	--datatier RAWRECO \
	--filein $START_ROOT_FILE \
	--fileout file:$ROOT_FILES_DIR/RAWskimmed.root \
	-n 100 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/selection.py

# LHE production and cleaning
echo "################ LHE production and cleaning ################"
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO \
	--processName LHEembeddingCLEAN \
	--data \
	--scenario pp \
	--conditions auto:run3_data \
	--era Run3_2024 \
	--eventcontent TauEmbeddingCleaning \
	--datatier RAWRECO \
	--procModifiers tau_embedding_cleaning,tau_embedding_mu_to_mu \
	--filein file:$ROOT_FILES_DIR/RAWskimmed.root \
	--fileout file:$ROOT_FILES_DIR/lhe_and_cleaned.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/lheprodandcleaning.py

# Simulation (MC & Detector)
echo "################ Simulation (MC & Detector) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/Simulation_GEN_cfi.py \
	--step GEN,SIM,DIGI,L1,DIGI2RAW \
	--processName SIMembeddingpreHLT \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--era Run3_2024 \
	--conditions auto:phase1_2024_realistic \
	--eventcontent TauEmbeddingSimGen \
	--datatier RAWSIM \
	--procModifiers tau_embedding_sim,tau_embedding_mu_to_mu \
	--filein file:$ROOT_FILES_DIR/lhe_and_cleaned.root \
	--fileout file:$ROOT_FILES_DIR/simulated_and_cleaned_prehlt.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/generator_preHLT.py

# Simulation (Trigger)
echo "################ Simulation (Trigger) ################"
cmsDriver.py \
	--step HLT:Fake2+TauAnalysis/MCEmbeddingTools/Simulation_HLT_customiser_cff.embeddingHLTCustomiser \
	--processName SIMembeddingHLT \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--era Run3_2024 \
	--conditions auto:phase1_2024_realistic \
	--eventcontent TauEmbeddingSimHLT \
	--datatier RAWSIM \
	--filein file:$ROOT_FILES_DIR/simulated_and_cleaned_prehlt.root \
	--fileout file:$ROOT_FILES_DIR/simulated_and_cleaned_hlt.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/generator_HLT.py

# Simulation (Reconstruction)
echo "################ Simulation (Reconstruction) ################"
cmsDriver.py \
	--step RAW2DIGI,L1Reco,RECO,RECOSIM \
	--processName SIMembedding \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--era Run3_2024 \
	--conditions auto:phase1_2024_realistic \
	--eventcontent TauEmbeddingSimReco \
	--datatier RAW-RECO-SIM \
	--procModifiers tau_embedding_sim \
	--filein file:$ROOT_FILES_DIR/simulated_and_cleaned_hlt.root \
	--fileout file:$ROOT_FILES_DIR/simulated_and_cleaned_posthlt.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/generator_postHLT.py

# Merging
echo "################ Merging ################"
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/Merging_USER_cff.merge_step,PAT \
	--processName MERGE \
	--data \
	--scenario pp \
	--conditions auto:run3_data \
	--era Run3_2024 \
	--eventcontent TauEmbeddingMergeMINIAOD \
	--datatier USER \
	--procModifiers tau_embedding_merging \
	--inputCommands 'keep *_*_*_*' \
	--filein file:$ROOT_FILES_DIR/simulated_and_cleaned_posthlt.root \
	--fileout file:$ROOT_FILES_DIR/merged.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/merging.py

# NanoAOD Production
echo "################ NanoAOD Production ################"
cmsDriver.py \
	--step NANO:@TauEmbedding \
	--data \
	--scenario pp \
	--conditions auto:run3_data \
	--era Run3_2024 \
	--eventcontent TauEmbeddingNANOAOD \
	--datatier NANOAODSIM \
	--filein file:$ROOT_FILES_DIR/merged.root \
	--fileout file:$ROOT_FILES_DIR/merged_nano.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/embedding_nanoAOD.py
