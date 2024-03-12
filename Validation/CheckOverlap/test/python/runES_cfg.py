import FWCore.ParameterSet.Config as cms

from Validation.CheckOverlap.testOverlap_cff import *

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type       = cms.string('CheckOverlap'),
    Resolution = cms.untracked.int32(1000),
    NodeNames  = cms.untracked.vstring('SF')
))

# foo bar baz
# qaEK5GwM4wckf
# m6OKsjS02u6Ur
