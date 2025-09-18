#!/bin/bash
# This script runs all the steps of the embedding workflow
# author: Christian Winter (christian.winter@cern.ch)
# TODO: move the dataset to a more persistent and for cmssw accessible place than a private EOS user folder

# print the exit status before exiting
function die {
    echo $1: status $2
    exit $2
}

## This is a dataset from the CMSSW integration file catalog (IBEos) from the release validation tests (runTheMatrix.py)
dataset="root://eoscms.cern.ch//store/user/cmsbuild/store/data/Run2022C/DoubleMuon/RAW/v1/000/356/381/00000/0b55fb47-4e17-49f2-a753-e625a741e44c.root"

echo "################ Selection ################"
cmsDriver.py \
    --step RAW2DIGI,L1Reco,RECO,PAT,FILTER:TauAnalysis/MCEmbeddingTools/Selection_FILTER_cff.makePatMuonsZmumuSelection \
    --processName SELECT \
    --data \
    --scenario pp \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent TauEmbeddingSelection \
    --datatier RAWRECO \
    --filein $dataset \
    --fileout file:selection.root \
    -n 100 \
    --python_filename selection.py || die 'Failure during selecting step' $?

echo "################ LHE production and cleaning ################"
cmsDriver.py \
    --step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO \
    --processName LHEembeddingCLEAN \
    --data \
    --scenario pp \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent TauEmbeddingCleaning \
    --datatier RAWRECO \
    --procModifiers tau_embedding_cleaning,tau_embedding_mutauh \
    --filein file:selection.root \
    --fileout file:lhe_and_cleaned.root \
    -n -1 \
    --python_filename lheprodandcleaning.py || die 'Failure during LHE and Cleaning step' $?

# Simulation (MC & Detector)
echo "################ Simulation (MC & Detector) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/Simulation_GEN_cfi.py \
    --step GEN,SIM,DIGI,L1,DIGI2RAW \
    --processName SIMembeddingpreHLT \
    --mc \
    --beamspot Realistic25ns13p6TeVEarly2022Collision \
    --geometry DB:Extended \
    --era Run3 \
    --conditions auto:phase1_2022_realistic_postEE \
    --eventcontent TauEmbeddingSimGen \
    --datatier RAWSIM \
    --procModifiers tau_embedding_sim,tau_embedding_mutauh \
    --filein file:lhe_and_cleaned.root \
    --fileout file:simulated_and_cleaned_prehlt.root \
    -n -1 \
    --python_filename generator_preHLT.py || die 'Failure during MC & Detector simulation step' $?

# Simulation (Trigger)
echo "################ Simulation (Trigger) ################"
cmsDriver.py \
    --step HLT:Fake2+TauAnalysis/MCEmbeddingTools/Simulation_HLT_customiser_cff.embeddingHLTCustomiser \
    --processName SIMembeddingHLT \
    --mc \
    --beamspot Realistic25ns13p6TeVEarly2022Collision \
    --geometry DB:Extended \
    --era Run3 \
    --conditions auto:phase1_2022_realistic_postEE \
    --eventcontent TauEmbeddingSimHLT \
    --datatier RAWSIM \
    --filein file:simulated_and_cleaned_prehlt.root \
    --fileout file:simulated_and_cleaned_hlt.root \
    -n -1 \
    --python_filename generator_HLT.py || die 'Failure during Fake Trigger simulation step' $?

# Simulation (Reconstruction)
echo "################ Simulation (Reconstruction) ################"
cmsDriver.py \
    --step RAW2DIGI,L1Reco,RECO,RECOSIM \
    --processName SIMembedding \
    --mc \
    --beamspot Realistic25ns13p6TeVEarly2022Collision \
    --geometry DB:Extended \
    --era Run3 \
    --conditions auto:phase1_2022_realistic_postEE \
    --eventcontent TauEmbeddingSimReco \
    --datatier RAW-RECO-SIM \
    --procModifiers tau_embedding_sim \
    --filein file:simulated_and_cleaned_hlt.root \
    --fileout file:simulated_and_cleaned_posthlt.root \
    -n -1 \
    --python_filename generator_postHLT.py || die 'Failure during reconstruction simulation step' $?

# Merging
echo "################ Merging ################"
cmsDriver.py \
    --step USER:TauAnalysis/MCEmbeddingTools/Merging_USER_cff.merge_step,PAT \
    --data \
    --scenario pp \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent TauEmbeddingMergeMINIAOD \
    --datatier USER \
    --procModifiers tau_embedding_merging \
    --inputCommands 'keep *_*_*_*' \
    --filein file:simulated_and_cleaned_posthlt.root \
    --fileout file:merged.root \
    -n -1 \
    --python_filename merging.py || die 'Failure during the merging step' $?

# NanoAOD Production
echo "################ NanoAOD Production ################"
cmsDriver.py \
    --step NANO:@TauEmbedding \
    --data \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent TauEmbeddingNANOAOD \
    --datatier NANOAODSIM \
    --filein file:merged.root \
    --fileout file:merged_nano.root \
    -n -1 \
    --python_filename embedding_nanoAOD.py || die 'Failure during the nanoAOD step' $?
