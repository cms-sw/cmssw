import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.ParticleListDrawer=dict()

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(relValTTbar)
#   fileNames = cms.untracked.vstring(
#    '/store/mc/Fall11/Tbar_TuneZ2_s-channel_7TeV-powheg-tauola/AODSIM/PU_S6_START42_V14B-v1/0000/F4D77C89-79F9-E011-82E8-001A92811746.root'
#    )
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure genEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.stGenEvent_cff")

## produce printout of particle listings (for debugging)
process.load("TopQuarkAnalysis.TopEventProducers.sequences.printGenParticles_cff")

## path1
process.p1 = cms.Path(process.printGenParticles *
                      process.makeGenEvt *
                      process.printDecaySubset)
