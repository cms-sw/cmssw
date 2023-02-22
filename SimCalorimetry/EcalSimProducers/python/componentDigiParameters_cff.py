import FWCore.ParameterSet.Config as cms

component_digi_parameters = cms.PSet(
    componentDigiTag = cms.string("Component"),
    componentTimeTag = cms.string("Component"),
    componentSeparateDigi = cms.bool(False),
    componentAddToBarrel  = cms.bool(False),
    componentTimePhase  = cms.double(0.),

)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(component_digi_parameters, EcalPreMixStage1 = True)
