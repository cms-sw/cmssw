import FWCore.ParameterSet.Config as cms

from Validation.CheckOverlap.testOverlap_cff import *

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type       = cms.string('CheckOverlap'),
    Resolution = cms.untracked.int32(1000),
    NodeNames  = cms.untracked.vstring('MEP', 'MEN')
))

