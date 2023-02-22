import FWCore.ParameterSet.Config as cms

component_digi_parameters = cms.PSet(
    componentDigiTag = cms.string("Component"),
    componentTimeTag = cms.string("Component"),
    componentSeparateDigi = cms.bool(False),
    componentAddToBarrel  = cms.bool(False),
    componentTimePhase  = cms.double(0.),

)
