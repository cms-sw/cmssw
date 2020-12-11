
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#

from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(relValTTbar)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary      = cms.untracked.bool(True)
)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("TopQuarkAnalysis.TopEventProducers.producers.pseudoTop_cfi")
process.task = cms.Task(process.pseudoTop)
process.pseudoTop.genParticles = "genParticles"
process.pseudoTop.finalStates = "genParticles"

process.printDecay = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("pseudoTop"),
    maxEventsToPrint = cms.untracked.int32(-1),
#    useMessageLogger = cms.untracked.bool(True)
)

## path
process.p = cms.Path(process.printDecay, process.task)
