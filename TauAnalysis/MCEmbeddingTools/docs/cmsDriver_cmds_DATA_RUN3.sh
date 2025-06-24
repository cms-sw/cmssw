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
	--eventcontent RAWRECO \
	--datatier RAWRECO \
	--outputCommands 'keep *_selectedMuonsForEmbedding_*_SELECT','keep *_patMuonsAfterID_*_SELECT','keep *_slimmedMuons_*_SELECT','keep *_slimmedMuonTrackExtras_*_SELECT','keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT','keep *_firstStepPrimaryVertices_*_SELECT','keep *_ecalDrivenElectronSeeds_*_SELECT' \
	--filein $START_ROOT_FILE \
	--fileout file:$ROOT_FILES_DIR/RAWskimmed.root \
	-n 100 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/selection.py

# LHE production and cleaning
echo "################ LHE production and cleaning ################"
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO:TauAnalysis/MCEmbeddingTools/Cleaning_RECO_cff.reconstruction \
	--processName LHEembeddingCLEAN \
	--data \
	--scenario pp \
	--conditions auto:run3_data \
	--era Run3_2024 \
	--eventcontent RAWRECO \
	--datatier RAWRECO \
	--outputCommands 'drop *_*_*_SELECT','drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*','drop *_ctppsProtons_*_*','drop *_ctppsLocalTrackLiteProducer_*_*','drop *_ctppsDiamondLocalTracks_*_*','drop *_ctppsDiamondRecHits_*_*','drop *_ctppsDiamondRawToDigi_*_*','drop *_ctppsPixelLocalTracks_*_*','drop *_ctppsPixelRecHits_*_*','drop *_ctppsPixelClusters_*_*','drop *_ctppsPixelDigis_*_*','drop *_totemRPLocalTrackFitter_*_*','drop *_totemRPUVPatternFinder_*_*','drop *_totemRPRecHitProducer_*_*','drop *_totemRPClusterProducer_*_*','drop *_totemRPRawToDigi_*_*','drop *_muonSimClassifier_*_*','keep *_patMuonsAfterID_*_SELECT','keep *_slimmedMuons_*_SELECT','keep *_selectedMuonsForEmbedding_*_SELECT','keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT','keep *_firstStepPrimaryVertices_*_SELECT','keep *_offlineBeamSpot_*_SELECT','keep *_l1extraParticles_*_SELECT','keep TrajectorySeeds_*_*_*','keep recoElectronSeeds_*_*_*','keep *_generalTracks_*_LHEembeddingCLEAN','keep *_generalTracks_*_CLEAN','keep *_cosmicsVetoTracksRaw_*_LHEembeddingCLEAN','keep *_cosmicsVetoTracksRaw_*_CLEAN','keep *_electronGsfTracks_*_LHEembeddingCLEAN','keep *_electronGsfTracks_*_CLEAN','keep *_lowPtGsfEleGsfTracks_*_LHEembeddingCLEAN','keep *_lowPtGsfEleGsfTracks_*_CLEAN','keep *_displacedTracks_*_LHEembeddingCLEAN','keep *_displacedTracks_*_CLEAN','keep *_ckfOutInTracksFromConversions_*_LHEembeddingCLEAN','keep *_ckfOutInTracksFromConversions_*_CLEAN','keep *_muons1stStep_*_LHEembeddingCLEAN','keep *_muons1stStep_*_CLEAN','keep *_displacedMuons1stStep_*_LHEembeddingCLEAN','keep *_displacedMuons1stStep_*_CLEAN','keep *_conversions_*_LHEembeddingCLEAN','keep *_conversions_*_CLEAN','keep *_allConversions_*_LHEembeddingCLEAN','keep *_allConversions_*_CLEAN','keep *_particleFlowTmp_*_LHEembeddingCLEAN','keep *_particleFlowTmp_*_CLEAN','keep *_ecalDigis_*_LHEembeddingCLEAN','keep *_ecalDigis_*_CLEAN','keep *_hcalDigis_*_LHEembeddingCLEAN','keep *_hcalDigis_*_CLEAN','keep *_ecalRecHit_*_LHEembeddingCLEAN','keep *_ecalRecHit_*_CLEAN','keep *_ecalPreshowerRecHit_*_LHEembeddingCLEAN','keep *_ecalPreshowerRecHit_*_CLEAN','keep *_hbhereco_*_LHEembeddingCLEAN','keep *_hbhereco_*_CLEAN','keep *_horeco_*_LHEembeddingCLEAN','keep *_horeco_*_CLEAN','keep *_hfreco_*_LHEembeddingCLEAN','keep *_hfreco_*_CLEAN','keep *_standAloneMuons_*_LHEembeddingCLEAN','keep *_glbTrackQual_*_LHEembeddingCLEAN','keep *_externalLHEProducer_*_LHEembedding','keep *_externalLHEProducer_*_LHEembeddingCLEAN' \
	--procModifiers tau_embedding_mu_to_mu \
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
	--eventcontent RAWSIM \
	--datatier RAWSIM \
	--outputCommands 'keep *_*_*_LHEembeddingCLEAN','keep *_*_*_SELECT','drop *_muonReducedTrackExtras_*_*','drop *_*_uncleanedConversions_*','drop *_diamondSampicLocalTracks_*_*','keep *_*_unsmeared_*', \
	--procModifiers tau_embedding_mu_to_mu \
	--filein file:$ROOT_FILES_DIR/lhe_and_cleaned.root \
	--fileout file:$ROOT_FILES_DIR/simulated_and_cleaned_prehlt.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/generator_preHLT.py

# Simulation (Trigger)
echo "################ Simulation (Trigger) ################"
cmsDriver.py \
	--step HLT:TauAnalysis/MCEmbeddingTools/Simulation_HLT_customiser_cff.embeddingHLTCustomiser.Fake2 \
	--processName SIMembeddingHLT \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--era Run3_2024 \
	--conditions auto:phase1_2024_realistic \
	--eventcontent RAWSIM \
	--datatier RAWSIM \
	--outputCommands 'keep *_*_*_SELECT','keep *_*_*_LHEembeddingCLEAN','keep *_*_unsmeared_SIMembeddingpreHLT','keep DcsStatuss_hltScalersRawToDigi_*_*' \
	--filein file:$ROOT_FILES_DIR/simulated_and_cleaned_prehlt.root \
	--fileout file:$ROOT_FILES_DIR/simulated_and_cleaned_hlt.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/generator_HLT.py

# Simulation (Reconstruction)
echo "################ Simulation (Reconstruction) ################"
cmsDriver.py \
	--step RAW2DIGI,L1Reco,RECO:TauAnalysis/MCEmbeddingTools/Simulation_RECO_cff.reconstruction,RECOSIM \
	--processName SIMembedding \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--era Run3_2024 \
	--conditions auto:phase1_2024_realistic \
	--eventcontent RAWRECOSIMHLT \
	--datatier RAW-RECO-SIM \
	--outputCommands 'keep *_*_*_LHEembeddingCLEAN','keep *_*_*_SELECT','keep *_genParticles_*_SIMembedding','keep *_standAloneMuons_*_SIMembedding','keep *_glbTrackQual_*_SIMembedding','keep *_generator_*_SIMembedding','keep *_addPileupInfo_*_SIMembedding','keep *_selectedMuonsForEmbedding_*_*','keep *_slimmedAddPileupInfo_*_*','keep *_embeddingHltPixelVertices_*_*','keep *_*_vertexPosition_*','keep recoMuons_muonsFromCosmics_*_*','keep recoTracks_cosmicMuons1Leg_*_*','keep recoMuons_muonsFromCosmics1Leg_*_*','keep *_muonDTDigis_*_*','keep *_muonCSCDigis_*_*','keep TrajectorySeeds_*_*_*','keep recoElectronSeeds_*_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*','drop *_ctppsProtons_*_*','drop *_ctppsLocalTrackLiteProducer_*_*','drop *_ctppsDiamondLocalTracks_*_*','drop *_ctppsDiamondRecHits_*_*','drop *_ctppsDiamondRawToDigi_*_*','drop *_ctppsPixelLocalTracks_*_*','drop *_ctppsPixelRecHits_*_*','drop *_ctppsPixelClusters_*_*','drop *_ctppsPixelDigis_*_*','drop *_totemRPLocalTrackFitter_*_*','drop *_totemRPUVPatternFinder_*_*','drop *_totemRPRecHitProducer_*_*','drop *_totemRPClusterProducer_*_*','drop *_totemRPRawToDigi_*_*','drop *_muonSimClassifier_*_*','keep *_generalTracks_*_SIMembedding','keep *_cosmicsVetoTracksRaw_*_SIMembedding','keep *_electronGsfTracks_*_SIMembedding','keep *_lowPtGsfEleGsfTracks_*_SIMembedding','keep *_displacedTracks_*_SIMembedding','keep *_ckfOutInTracksFromConversions_*_SIMembedding','keep *_muons1stStep_*_SIMembedding','keep *_displacedMuons1stStep_*_SIMembedding','keep *_conversions_*_SIMembedding','keep *_allConversions_*_SIMembedding','keep *_particleFlowTmp_*_SIMembedding','keep *_ecalDigis_*_SIMembedding','keep *_hcalDigis_*_SIMembedding','keep *_ecalRecHit_*_SIMembedding','keep *_ecalPreshowerRecHit_*_SIMembedding','keep *_hbhereco_*_SIMembedding','keep *_horeco_*_SIMembedding','keep *_hfreco_*_SIMembedding','keep *_*_unsmeared_SIMembeddingpreHLT','keep *_hltScalersRawToDigi_*_SIMembeddingHLT' \
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
	--eventcontent MINIAODSIM \
	--datatier USER \
	--inputCommands 'keep *_*_*_*' \
	--outputCommands 'drop *_*_*_SELECT','keep *_prunedGenParticles_*_MERGE','keep *_generator_*_SIMembeddingpreHLT','keep *_generator_*_SIMembeddingHLT','keep *_generator_*_SIMembedding','keep *_selectedMuonsForEmbedding_*_*','keep *_unpackedPatTrigger_*_*','keep patPackedGenParticles_packedGenParticles_*_*','keep recoGenParticles_prunedGenParticles_*_*','keep *_packedPFCandidateToGenAssociation_*_*','keep *_lostTracksToGenAssociation_*_*','keep LHEEventProduct_*_*_*','keep GenFilterInfo_*_*_*','keep GenLumiInfoHeader_generator_*_*','keep GenLumiInfoProduct_*_*_*','keep GenEventInfoProduct_generator_*_*','keep recoGenParticles_genPUProtons_*_*','keep *_slimmedGenJetsFlavourInfos_*_*','keep *_slimmedGenJets__*','keep *_slimmedGenJetsAK8__*','keep *_slimmedGenJetsAK8SoftDropSubJets__*','keep *_genMetTrue_*_*','keep LHERunInfoProduct_*_*_*','keep GenRunInfoProduct_*_*_*','keep *_genParticles_xyz0_*','keep *_genParticles_t0_*' \
	--filein file:$ROOT_FILES_DIR/simulated_and_cleaned_posthlt.root \
	--fileout file:$ROOT_FILES_DIR/merged.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/merging.py

# NanoAOD Production
echo "################ NanoAOD Production ################"
cmsDriver.py \
	--step NANO:TauAnalysis/MCEmbeddingTools/Nano_cff.embedding_nanoAOD_seq \
	--data \
	--scenario pp \
	--conditions auto:run3_data \
	--era Run3_2024 \
	--eventcontent NANOAODSIM \
	--datatier NANOAODSIM \
	--outputCommands 'keep edmTriggerResults_*_*_SIMembeddingpreHLT','keep edmTriggerResults_*_*_SIMembeddingHLT','keep edmTriggerResults_*_*_SIMembedding','keep edmTriggerResults_*_*_MERGE','keep edmTriggerResults_*_*_NANO' \
	--filein file:$ROOT_FILES_DIR/merged.root \
	--fileout file:$ROOT_FILES_DIR/merged_nano.root \
	-n -1 \
	--nThreads 10 \
	--python_filename $EMB_CONF_DIR/embedding_nanoAOD.py
