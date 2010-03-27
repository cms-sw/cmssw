import FWCore.ParameterSet.Config as cms

apd_sim_parameters = cms.PSet(
    apdAddToBarrel  = cms.bool(False),
    apdSeparateDigi = cms.bool(False),
    apdSimToPELow   = cms.double(4.41e6),
    apdSimToPEHigh  = cms.double(157.5e6),
    apdTimeOffset   = cms.double(-10.0),
    apdDoPEStats    = cms.bool(True),
    apdDigiTag      = cms.string("APD"),
    apdShapeTstart  = cms.double( 74.5 ),
    apdShapeTau     = cms.double( 40.5 )
)

