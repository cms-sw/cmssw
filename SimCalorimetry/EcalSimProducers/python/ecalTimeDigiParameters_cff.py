import FWCore.ParameterSet.Config as cms

ecal_time_digi_parameters = cms.PSet(
    hitsProducer     = cms.string('g4SimHits'),
    EBtimeDigiCollection = cms.string('EBTimeDigi'),
    EEtimeDigiCollection = cms.string('EETimeDigi'),
    EKtimeDigiCollection = cms.string('EKTimeDigi'),
    timeLayerBarrel = cms.int32(7),
    timeLayerEndcap = cms.int32(3),
    timeLayerShashlik = cms.int32(13)
    )
