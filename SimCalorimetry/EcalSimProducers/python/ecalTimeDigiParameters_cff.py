import FWCore.ParameterSet.Config as cms

ecal_time_digi_parameters = cms.PSet(
    hitsProducerEB     = cms.InputTag('g4SimHits:EcalHitsEB'),
    hitsProducerEE     = cms.InputTag('g4SimHits:EcalHitsEE'),
    EBtimeDigiCollection = cms.string('EBTimeDigi'),
    EEtimeDigiCollection = cms.string('EETimeDigi'),
    timeLayerBarrel = cms.int32(7),
    timeLayerEndcap = cms.int32(3)
    )
