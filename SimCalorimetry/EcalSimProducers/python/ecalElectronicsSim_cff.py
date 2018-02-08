import FWCore.ParameterSet.Config as cms

ecal_electronics_sim = cms.PSet(
    doENoise = cms.bool(True),
    ConstantTerm = cms.double(0.003),
    applyConstantTerm = cms.bool(True)
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(ecal_electronics_sim,
    doENoise = False,
    applyConstantTerm = False,
)
