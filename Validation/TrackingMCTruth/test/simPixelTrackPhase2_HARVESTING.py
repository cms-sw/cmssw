"""
This script runs the harvesting step on top of the file `simPixelTrack_DQMIO.root` produced when running the
simPixelTrackPhase2_TEST.py script. To harvest simply run:

cmsRun simPixelTrackPhase2_HARVESTING.py

This will produce a DQM file with all SimPixelTrack histograms.
"""

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

process = cms.Process('HARVESTING',Phase2C22I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:simPixelTrack_DQMIO.root')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step4 nevts:100'),
    name = cms.untracked.string('Applications')
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

# Path and EndPath definitions
process.load('Validation.TrackingMCTruth.PostProcessorSimPixelTrack_cff')  # load harvesting config for SimPixelTrack
process.harvesting_step = cms.Path(process.postProcessorSimPixelTrackSequence)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.harvesting_step,process.dqmsave_step)
