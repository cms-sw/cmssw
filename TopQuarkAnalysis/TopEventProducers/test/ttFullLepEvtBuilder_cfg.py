import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.TtFullLeptonicEvent = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
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

## std sequence for pat
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.task.add(process.patCandidatesTask)
#Temporary customize to the unit tests that fail due to old input samples
process.patTaus.skipMissingTauID = True
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
process.task.add(process.makeGenEvtTask)

## std sequence to produce the ttFullLepEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttFullLepEvtBuilder_cff")
process.task.add(process.makeTtFullLepEventTask)
process.ttFullLepEvent.verbosity = 1

## optional change of settings
#from TopQuarkAnalysis.TopEventProducers.sequences.ttFullLepEvtBuilder_cff import *
#removeTtFullLepHypGenMatch(process)

#setForAllTtFullLepHypotheses(process,"muons","myMuons")
#setForAllTtFullLepHypotheses(process,"jets","myJets")
#setForAllTtFullLepHypotheses(process,"maxNJets",4)
#setForAllTtFullLepHypotheses(process,"jetCorrectionLevel","part")

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName     = cms.untracked.string('ttFullLepEvtBuilder.root'),
    outputCommands = cms.untracked.vstring('drop *'),
    dropMetaData = cms.untracked.string('DROPPED')
)
process.outpath = cms.EndPath(process.out, process.task)

## PAT content
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out.outputCommands += patEventContentNoCleaning
## TQAF content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import tqafEventContent
process.out.outputCommands += tqafEventContent
