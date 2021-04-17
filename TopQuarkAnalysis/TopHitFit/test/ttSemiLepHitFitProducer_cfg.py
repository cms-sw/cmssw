import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.TopHitFit=dict()

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
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)

## std sequence to produce the kinematic fit for semi-leptonic events
process.load("TopQuarkAnalysis.TopHitFit.TtSemiLepHitFitProducer_Muons_cfi")
process.task.add(process.hitFitTtSemiLepEvent)

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ttSemiLepHitFitProducer.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_hitFitTtSemiLepEvent_*_*']

## output path
process.outpath = cms.EndPath(process.out, process.task)
