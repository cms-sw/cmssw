import FWCore.ParameterSet.Config as cms

trackAssociatorByChi2 = cms.EDProducer("TrackAssociatorByChi2Producer",
    chi2cut = cms.double(25.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    onlyDiagonal = cms.bool(False)
)


