#!/bin/bash -ex

##____________________________________________________________________________||
function die { echo $1: status $2 ; exit $2; }

# echo '{
#"274199" : [[1, 180]]
#}' > step1_lumiRanges.log  2>&1
 
# (das_client --limit 0 --query 'file dataset=/DoubleMuon/Run2016B-v2/RAW run=274199') | sort -u > step1_dasquery.log  2>&1
#due to frequent failures of das_client: use file directly

#echo "/store/data/Run2016B/DoubleMuon/RAW/v2/000/274/199/00000/02893BCB-6426-E611-B266-02163E012552.root" > step1_dasquery.log

#cmsDriver.py selecting -s RAW2DIGI,L1Reco,RECO,PAT --data --scenario pp --conditions auto:run2_data --era Run2_2016_HIPM --eventcontent RAWRECO --datatier RAWRECO --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseSelecting --filein filelist:step1_dasquery.log --lumiToProcess step1_lumiRanges.log --fileout step2.root -n 200 --nThreads 32  


## This is a PRE SKIMED dataset, so  
echo "/store/user/swayand/Emmbeddingfiles/Emmbedding_testInput3.root"  > file_list.txt


cmsDriver.py selecting -s RAW2DIGI,L1Reco,RECO,PAT --data --scenario pp --conditions auto:run2_data --era Run2_2016_HIPM --eventcontent RAWRECO --datatier RAWRECO --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseSelecting --filein filelist:file_list.txt --fileout file:selected.root --python_filename selecting.py -n 5 || die 'Failure during selecting step' $?


cmsDriver.py LHEembeddingCLEAN --filein file:selected.root --fileout file:lhe_and_cleaned.root --data --scenario pp --conditions auto:run2_data --era Run2_2016_HIPM  --eventcontent RAWRECO --datatier RAWRECO --step RAW2DIGI,RECO --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseLHEandCleaning -n -1  python_filename lheprodandcleaning.py  || die 'Failure during LHE and Cleaning step' $?

### Do not run HLT in CMSSW_9_0_X so far since the RecoPixelVertexingPixelTrackFittingPlugins. seems not to work
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py --filein file:lhe_and_cleaned.root --fileout file:simulated_and_cleaned.root --conditions auto:run2_mc --era Run2_2016 --eventcontent RAWRECO --step GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,RECO --datatier RAWRECO --customise TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator --beamspot Realistic25ns13TeV2016Collision -n -1 --customise_commands "process.generator.nAttempts = cms.uint32(1000)\n"  --python_filename simulation.py || die 'Failure during Simulation step' $?

cmsDriver.py MERGE -s PAT --filein file:simulated_and_cleaned.root  --fileout file:merged.root --era Run2_2016_HIPM --data --scenario pp --conditions auto:run2_data --eventcontent  MINIAODSIM --datatier USER --customise TauAnalysis/MCEmbeddingTools/customisers.customiseMerging --customise_commands "process.patTrigger.processName = cms.string('SIMembedding')" -n -1  || die 'Failure during merging step' $?
