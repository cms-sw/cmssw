import FWCore.ParameterSet.Config as cms

ecal_time_digi_parameters = cms.PSet(
    hitsProducer     = cms.string('g4SimHits'),
    EEtimeDigiCollection = cms.string('EBTimeDigi'),
    EBtimeDigiCollection = cms.string('EETimeDigi'),
    timeLayerBarrel = cms.int32(7),
    timeLayerEndcap = cms.int32(3)
    )
