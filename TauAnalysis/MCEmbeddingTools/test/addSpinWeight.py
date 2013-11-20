import FWCore.ParameterSet.Config as cms

process = cms.Process('EmbeddedSPIN')

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:embed_AOD.root')
)

# Output definition

process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('spinned.root'),
)

# TauSpinner
process.load('GeneratorInterface.ExternalDecays.TauSpinner_cfi')

# Set up the path
process.step = cms.EndPath(process.TauSpinnerReco*process.output)
