import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
## dump content of TopGenEvent
process.MessageLogger.TopGenEvent=dict()

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

process.task = cms.Task()

## load modules to produce the TtGenEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
process.task.add(process.makeGenEvtTask)

## load analyzer
process.load("TopQuarkAnalysis.Examples.TopGenEventAnalyzer_cfi")

## register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopGenEvent.root')
)

## end path
process.p1 = cms.Path(process.analyzeTopGenEvent, process.task)
