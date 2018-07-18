import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary      = cms.untracked.bool(True)
)

## configure geometry & conditions
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.task = cms.Task()

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.task.add(process.patCandidatesTask)
#Temporary customize to the unit tests that fail due to old input samples
process.patTaus.skipMissingTauID = True
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)

## std sequence for TQAF
process.load("TopQuarkAnalysis.TopEventProducers.tqafSequences_cff")
process.task.add(process.tqafTtSemiLeptonicTask)

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName       = cms.untracked.string('tqaf.root'),
    outputCommands = cms.untracked.vstring('drop *'),
    dropMetaData   = cms.untracked.string("DROPPED")  ## NONE    for none
                                                      ## DROPPED for drop for dropped data
)
process.outpath = cms.EndPath(process.out, process.task)

## PAT content
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out.outputCommands += patEventContentNoCleaning
## TQAF content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import tqafEventContent
process.out.outputCommands += tqafEventContent
