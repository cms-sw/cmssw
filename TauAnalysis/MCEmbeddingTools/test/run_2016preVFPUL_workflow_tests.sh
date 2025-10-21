#!/bin/bash
# This script runs all the steps of the embedding workflow
# author: Christian Winter (christian.winter@cern.ch)
# TODO: move the dataset to a more persistent and for cmssw accessible place than a private EOS user folder

# print the exit status before exiting
function die {
    echo $1: status $2
    exit $2
}

## This is a PRE SKIMED dataset
dataset="root://eoscms.cern.ch//store/group/phys_tau/embedding_test_files/2016_C-v2_RAW_preselected.root"

echo "################ Selection ################"
cmsDriver.py RECO \
    --step RAW2DIGI,L1Reco,RECO,PAT \
    --data \
    --scenario pp \
    --conditions auto:run2_data \
    --era Run2_2016_HIPM \
    --eventcontent RAWRECO \
    --datatier RAWRECO \
    --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseSelecting_Reselect \
    --filein $dataset \
    --fileout file:selection.root \
    -n -1 \
    --python_filename selection.py || die 'Failure during selecting step' $?

echo "################ LHE production and cleaning ################"
cmsDriver.py LHEprodandCLEAN \
    --step RAW2DIGI,RECO,PAT \
    --data \
    --scenario pp \
    --conditions auto:run2_data \
    --era Run2_2016_HIPM \
    --eventcontent RAWRECO \
    --datatier RAWRECO \
    --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseLHEandCleaning_Reselect \
    --filein file:selection.root \
    --fileout file:lhe_and_cleaned.root \
    -n -1 \
    --python_filename lheprodandcleaning.py || die 'Failure during LHE and Cleaning step' $?

# Simulation (MC & Detector)
echo "################ Simulation (MC & Detector) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py \
    --step GEN,SIM,DIGI,L1,DIGI2RAW \
    --mc \
    --beamspot Realistic25ns13TeV2016Collision \
    --geometry DB:Extended \
    --era Run2_2016_HIPM \
    --conditions auto:run2_mc_pre_vfp \
    --eventcontent RAWSIM \
    --datatier RAWSIM \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator_preHLT_Reselect \
    --filein file:lhe_and_cleaned.root \
    --fileout file:simulated_and_cleaned_prehlt.root \
    -n -1 \
    --python_filename generator_preHLT.py || die 'Failure during MC & Detector simulation step' $?

# Simulation (Trigger)
echo "################ Simulation (Trigger) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py \
    --step HLT:Fake2 \
    --mc \
    --beamspot Realistic25ns13TeV2016Collision \
    --geometry DB:Extended \
    --era Run2_2016_HIPM \
    --conditions auto:run2_mc_pre_vfp \
    --eventcontent RAWSIM \
    --datatier RAWSIM \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator_HLT_Reselect \
    --filein file:simulated_and_cleaned_prehlt.root \
    --fileout file:simulated_and_cleaned_hlt.root \
    -n -1 \
    --python_filename generator_HLT.py || die 'Failure during Fake Trigger simulation step' $?

# Simulation (Reconstruction)
echo "################ Simulation (Reconstruction) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py \
    --step RAW2DIGI,L1Reco,RECO,RECOSIM \
    --mc \
    --beamspot Realistic25ns13TeV2016Collision \
    --geometry DB:Extended \
    --era Run2_2016_HIPM \
    --conditions auto:run2_mc_pre_vfp \
    --eventcontent RAWRECOSIMHLT \
    --datatier RAW-RECO-SIM \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator_postHLT_Reselect \
    --filein file:simulated_and_cleaned_hlt.root \
    --fileout file:simulated_and_cleaned_posthlt.root \
    -n -1 \
    --python_filename generator_postHLT.py || die 'Failure during reconstruction simulation step' $?

# Merging
echo "################ Merging ################"
cmsDriver.py PAT \
    --step PAT \
    --data \
    --scenario pp \
    --conditions auto:run2_data \
    --era Run2_2016_HIPM \
    --eventcontent MINIAODSIM \
    --datatier USER \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseMerging_Reselect \
    --filein file:simulated_and_cleaned_posthlt.root \
    --fileout file:merged.root \
    -n -1 \
    --python_filename merging.py || die 'Failure during the merging step' $?

# NanoAOD Production
echo "################ NanoAOD Production ################"
cmsDriver.py \
    --step NANO \
    --data \
    --conditions auto:run2_data \
    --era Run2_2016_HIPM,run2_nanoAOD_106Xv2 \
    --eventcontent NANOAODSIM \
    --datatier NANOAODSIM \
    --customise TauAnalysis/MCEmbeddingTools/customisers.customiseNanoAOD \
    --filein file:merged.root \
    --fileout file:merged_nano.root \
    -n -1 \
    --python_filename embedding_nanoAOD.py || die 'Failure during the nanoAOD step' $?
