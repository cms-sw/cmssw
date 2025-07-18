"""
This script runs the SimDoubletsProducer and SimDoubletsAnalyzer for Run3.

To run it, you first have to produce or get the output root file from a step 2 that runs the HLT.
The input file is expected to be named `step2.root` by default, but you can also rename it below.
Then you can simply run the config using:

cmsRun simDoubletsPhase1_TEST.py

It will produce one DQMIO output file named `simDoublets_DQMIO.root`. 
This can be further processed in the harvesting step by running the simDoubletsPhase1_HARVESTING.py script.
"""
inputFile = "step2.root"

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025

process = cms.Process("SIMDOUBLETS",Run3_2025)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:%s' % inputFile),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_hltSiPixelRecHits_*_*'   # we will reproduce the PixelRecHits to have their local position available
    ),
    secondaryFileNames = cms.untracked.vstring()
)

### conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '142X_mcRun3_2025_realistic_v4', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

### load HLTDoLocalPixelSequence
process.load('HLTrigger.Configuration.HLT_GRun_cff')
### load hltTPClusterProducer
process.load("Validation.RecoTrack.associators_cff")
### load the new EDProducer "SimDoubletsProducerPhase1"
process.load("SimTracker.TrackerHitAssociation.simDoubletsProducerPhase1_cfi")
### load the new DQM EDAnalyzer "SimDoubletsAnalyzerPhase1"
process.load("Validation.TrackingMCTruth.simDoubletsAnalyzerPhase1_cfi")

####  set up the path
process.simDoubletPath = cms.Path(
    process.HLTDoLocalPixelSequence *   # reproduce the SiPixelRecHits
    process.hltTPClusterProducer *      # run the cluster to TrackingParticle association
    process.simDoubletsProducerPhase1 * # produce the SimDoublets
    process.simDoubletsAnalyzerPhase1   # analyze the SimDoublets
)

# Output definition
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:simDoublets_DQMIO.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)


process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


process.schedule = cms.Schedule(
      process.simDoubletPath,process.endjob_step,process.DQMoutput_step
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)