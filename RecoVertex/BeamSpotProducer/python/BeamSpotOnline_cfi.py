import FWCore.ParameterSet.Config as cms

onlineBeamSpotProducer = cms.EDProducer('BeamSpotOnlineProducer',
                                label = cms.InputTag('scalers'),
                                changeToCMSCoordinates = cms.bool(True)
)

