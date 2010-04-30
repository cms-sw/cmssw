import FWCore.ParameterSet.Config as cms

onlineBeamSpotProducer = cms.EDProducer('BeamSpotOnlineProducer',
                                        label = cms.InputTag('scalersRawToDigi'),
                                        changeToCMSCoordinates = cms.bool(False),
                                        maxZ = cms.double(40),
                                        maxRadius = cms.double(2)
)

