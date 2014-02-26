import FWCore.ParameterSet.Config as cms

# Random number generator service for trackingMaterialProducer
from IOMC.RandomEngine.IOMC_cff import *
RandomNumberGeneratorService.trackingMaterialProducer = cms.PSet(
    initialSeed = cms.untracked.uint32(288269),
    engineName = cms.untracked.string('HepJamesRandom')
)

# trackingMaterialProducer
from SimTracker.TrackerMaterialAnalysis.trackingMaterialProducer_cfi import *
