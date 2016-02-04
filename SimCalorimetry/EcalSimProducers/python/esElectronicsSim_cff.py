import FWCore.ParameterSet.Config as cms

es_electronics_sim = cms.PSet(
    doESNoise  = cms.bool(True),
    numESdetId = cms.int32(137216),
    doFast     = cms.bool(False)
)

