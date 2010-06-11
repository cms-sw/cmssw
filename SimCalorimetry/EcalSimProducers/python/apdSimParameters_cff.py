import FWCore.ParameterSet.Config as cms

apd_sim_parameters = cms.PSet(
    apdAddToBarrel  = cms.bool(False),
    apdSeparateDigi = cms.bool(True),
    apdSimToPELow   = cms.double(3.5e6),
    apdSimToPEHigh  = cms.double(126.e6),
    apdTimeOffset   = cms.double(-15.0),
    apdDoPEStats    = cms.bool(True),
    apdDigiTag      = cms.string("APD"),
    apdShapeTstart  = cms.double( 74.5 ),
    apdShapeTau     = cms.double( 40.5 )
)

