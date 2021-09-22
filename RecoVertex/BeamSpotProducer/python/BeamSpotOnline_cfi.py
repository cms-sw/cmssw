import FWCore.ParameterSet.Config as cms

onlineBeamSpotProducer = cms.EDProducer('BeamSpotOnlineProducer',
                                        src = cms.InputTag('scalersRawToDigi'),
                                        changeToCMSCoordinates = cms.bool(False),
                                        maxZ = cms.double(40),
                                        maxRadius = cms.double(2),
                                        setSigmaZ = cms.double(-1), #negative value disables it.
                                        gtEvmLabel = cms.InputTag('gtEvmDigis'),
                                        useTransientRecord = cms.bool(False)
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(onlineBeamSpotProducer, useTransientRecord = True)
