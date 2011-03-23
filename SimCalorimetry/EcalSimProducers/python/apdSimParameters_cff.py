import FWCore.ParameterSet.Config as cms

apd_sim_parameters = cms.PSet(
    apdAddToBarrel  = cms.bool(False),
    apdSeparateDigi = cms.bool(True),
    apdSimToPELow   = cms.double(2.45e6),
    apdSimToPEHigh  = cms.double(88.2e6),
    apdTimeOffset   = cms.double(-13.5),
    apdTimeOffWidth = cms.double(0.8),
    apdDoPEStats    = cms.bool(True),
    apdDigiTag      = cms.string("APD"),
    apdShapeTstart  = cms.double( 74.5 ),
    apdShapeTau     = cms.double( 40.5 ),
    apdNonlParms    = cms.vdouble( 1.48, -3.75, 1.81, 1.26, 2.0, 45, 1.0 )
)

