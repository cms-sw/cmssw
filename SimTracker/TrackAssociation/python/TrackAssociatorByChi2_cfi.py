import FWCore.ParameterSet.Config as cms

TrackAssociatorByChi2ESProducer = cms.ESProducer("TrackAssociatorByChi2ESProducer",
    chi2cut = cms.double(25.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    onlyDiagonal = cms.bool(False),
    ComponentName = cms.string('TrackAssociatorByChi2')
)


