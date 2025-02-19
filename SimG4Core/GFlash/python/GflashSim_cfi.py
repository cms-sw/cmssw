import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.g4SimHits_cfi import *

g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
g4SimHits.Physics.GFlash = cms.PSet(
    GflashHadronPhysics = cms.string('QGSP_FTFP_BERT'),
    GflashEMShowerModel = cms.bool(True),
    energyScaleEB = cms.double(1.032),
    energyScaleEE = cms.double(1.024),
    GflashHadronShowerModel = cms.bool(True),
    GflashHcalOuter = cms.bool(True),
    GflashExportToFastSim = cms.bool(False),
    GflashHistogram = cms.bool(False),
    GflashHistogramName = cms.string('gflash_histogram.root'),
    Verbosity = cms.untracked.int32(0),
    bField = cms.double(3.8),
    watcherOn = cms.bool(False),
    tuning_pList = cms.vdouble()
)
