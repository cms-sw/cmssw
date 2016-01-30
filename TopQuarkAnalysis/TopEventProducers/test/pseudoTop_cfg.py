
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.categories.append('ParticleListDrawer')

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
    allowUnscheduled = cms.untracked.bool(True),
    wantSummary      = cms.untracked.bool(True)
)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.pseudoTop = cms.EDProducer("PseudoTopProducer",
    genParticles = cms.InputTag("genParticles"),
    finalStates = cms.InputTag("genParticles"),
    leptonMinPt = cms.double(20),
    leptonMaxEta = cms.double(2.4),
    leptonConeSize = cms.double(0.1),
    jetMinPt = cms.double(30),
    jetMaxEta = cms.double(2.4),
    jetConeSize = cms.double(0.4),
    wMass = cms.double(80.4),
    tMass = cms.double(172.5),
)

process.printDecay = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("pseudoTop"),
    maxEventsToPrint = cms.untracked.int32(-1),
#    useMessageLogger = cms.untracked.bool(True)
)

## path
process.p = cms.Path(process.printDecay)
