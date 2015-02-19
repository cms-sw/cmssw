import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
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
    allowUnscheduled = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

## configure geometry & conditions
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

## std sequence for pat
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.load("PhysicsTools.PatAlgos.cleaningLayer1.cleanPatCandidates_cff")
process.cleanPatElectrons.checkOverlaps.muons.requireNoOverlaps = True
process.cleanPatJets.checkOverlaps.muons.requireNoOverlaps     = True
process.cleanPatJets.checkOverlaps.electrons.requireNoOverlaps = True
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## configure mva trainer
process.load("TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainTreeSaver_cff")

## make trainer looper known to the process
from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainTreeSaver_cff import looper
process.looper = looper

## produce pat objects and ttGenEvt and make mva training
process.p = cms.Path(process.saveTrainTree)
