import FWCore.ParameterSet.Config as cms

es_electronics_sim = cms.PSet(
    doFast     = cms.bool(True),
    doESNoise  = cms.bool(True)
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(es_electronics_sim, doESNoise = False)
