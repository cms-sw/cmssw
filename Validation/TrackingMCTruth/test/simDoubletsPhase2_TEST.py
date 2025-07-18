"""
This script runs the SimDoubletsProducer and SimDoubletsAnalyzer for Phase2.

To run it, you first have to produce or get the output root file from a step 2 that runs the HLT.
The input file is expected to be named `step2.root` by default, but you can also rename it below.
Then you can simply run the config using:

cmsRun simDoubletsPhase2_TEST.py

It will produce one DQMIO output file named `simDoublets_DQMIO.root`. 
This can be further processed in the harvesting step by running the simDoubletsPhase2_HARVESTING.py script.
"""
inputFile = "step2.root"

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("SIMDOUBLETS",Phase2C17I13M9)

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

### load hltSiPixelRecHits
process.load("HLTrigger.Configuration.HLT_75e33.modules.hltSiPixelRecHits_cfi")
### load hltTPClusterProducer
process.load("Validation.RecoTrack.associators_cff")
### load the new EDProducer "SimDoubletsProducerPhase2"
process.load("SimTracker.TrackerHitAssociation.simDoubletsProducerPhase2_cfi")
### load the new DQM EDAnalyzer "SimDoubletsAnalyzerPhase2"
process.load("Validation.TrackingMCTruth.simDoubletsAnalyzerPhase2_cfi")

####  set up the paths
process.simDoubletPath = cms.Path(
    process.hltSiPixelRecHits *         # reproduce the SiPixelRecHits
    process.hltTPClusterProducer *      # run the cluster to TrackingParticle association
    process.simDoubletsProducerPhase2 * # produce the SimDoublets
    process.simDoubletsAnalyzerPhase2   # analyze the SimDoublets
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
