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
dataset="root://eoscms.cern.ch//store/group/phys_tau/embedding_test_files/2022_G_RAW.root"

echo "################ Selection ################"
cmsDriver.py RECO \
    --step RAW2DIGI,L1Reco,RECO,PAT \
    --data \
    --scenario pp \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent RAWRECO \
    --datatier RAWRECO \
    --customise TauAnalysis/MCEmbeddingTools/customisers.customiseSelecting \
    --filein $dataset \
    --fileout file:selection.root \
    -n -1 \
    --python_filename selection.py || die 'Failure during selecting step' $?

echo "################ LHE production and cleaning ################"
cmsDriver.py LHEprodandCLEAN \
    --step RAW2DIGI,RECO,PAT \
    --data \
    --scenario pp \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent RAWRECO \
    --datatier RAWRECO \
    --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018,TauAnalysis/MCEmbeddingTools/customisers.customiseLHEandCleaning \
    --filein file:selection.root \
    --fileout file:lhe_and_cleaned.root \
    -n -1 \
    --python_filename lheprodandcleaning.py || die 'Failure during LHE and Cleaning step' $?

# Simulation (MC & Detector)
echo "################ Simulation (MC & Detector) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py \
    --step GEN,SIM,DIGI,L1,DIGI2RAW \
    --mc \
    --beamspot Realistic25ns13p6TeVEarly2022Collision \
    --geometry DB:Extended \
    --era Run3 \
    --conditions auto:phase1_2022_realistic_postEE \
    --eventcontent RAWSIM \
    --datatier RAWSIM \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator_preHLT \
    --customise_commands 'process.generator.HepMCFilter.filterParameters.MuMuCut = cms.string("(Mu.Pt > 18 && Had.Pt > 18 && Mu.Eta < 2.2 && Had.Eta < 2.4)");process.generator.HepMCFilter.filterParameters.Final_States = cms.vstring("MuHad");process.generator.nAttempts = cms.uint32(1000);' \
    --filein file:lhe_and_cleaned.root \
    --fileout file:simulated_and_cleaned_prehlt.root \
    -n -1 \
    --python_filename generator_preHLT.py || die 'Failure during MC & Detector simulation step' $?

# Simulation (Trigger)
echo "################ Simulation (Trigger) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py \
    --step HLT:Fake2 \
    --mc \
    --beamspot Realistic25ns13p6TeVEarly2022Collision \
    --geometry DB:Extended \
    --era Run3 \
    --conditions auto:phase1_2022_realistic_postEE \
    --eventcontent RAWSIM \
    --datatier RAWSIM \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator_HLT \
    --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True);' \
    --filein file:simulated_and_cleaned_prehlt.root \
    --fileout file:simulated_and_cleaned_hlt.root \
    -n -1 \
    --python_filename generator_HLT.py || die 'Failure during Fake Trigger simulation step' $?

# Simulation (Reconstruction)
echo "################ Simulation (Reconstruction) ################"
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py \
    --step RAW2DIGI,L1Reco,RECO,RECOSIM \
    --mc \
    --beamspot Realistic25ns13p6TeVEarly2022Collision \
    --geometry DB:Extended \
    --era Run3 \
    --conditions auto:phase1_2022_realistic_postEE \
    --eventcontent RAWRECOSIMHLT \
    --datatier RAW-RECO-SIM \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator_postHLT \
    --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True);' \
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
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent MINIAODSIM \
    --datatier USER \
    --customise \
    TauAnalysis/MCEmbeddingTools/customisers.customiseMerging \
    --filein file:simulated_and_cleaned_posthlt.root \
    --fileout file:merged.root \
    -n -1 \
    --python_filename merging.py || die 'Failure during the merging step' $?

# NanoAOD Production
echo "################ NanoAOD Production ################"
cmsDriver.py \
    --step NANO \
    --data \
    --conditions auto:run3_data \
    --era Run3 \
    --eventcontent NANOAODSIM \
    --datatier NANOAODSIM \
    --customise TauAnalysis/MCEmbeddingTools/customisers.customiseNanoAOD \
    --filein file:merged.root \
    --fileout file:merged_nano.root \
    -n -1 \
    --python_filename embedding_nanoAOD.py || die 'Failure during the nanoAOD step' $?
