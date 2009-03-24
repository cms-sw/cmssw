import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.g4SimHits_cfi import *

g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
g4SimHits.Physics.GFlash = cms.PSet(
    GflashHadronPhysics = cms.string('QGSP_BERT'),
    GflashEMShowerModel = cms.bool(True),
    GflashHadronShowerModel = cms.bool(True),
    GflashHistogram = cms.bool(True),
    GflashHistogramName = cms.string('gflash_histogram.root'),
    bField = cms.double(3.8),
    watcherOn = cms.bool(False),
    tuning_pList = cms.vdouble()
)
