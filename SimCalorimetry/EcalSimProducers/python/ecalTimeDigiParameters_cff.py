import FWCore.ParameterSet.Config as cms

ecal_time_digi_parameters = cms.PSet(
    hitsProducer     = cms.string('g4SimHits'),
    EBtimeDigiCollection = cms.string('EBTimeDigi'),
    EEtimeDigiCollection = cms.string('EETimeDigi'),
    timeLayerBarrel = cms.int32(7),
    timeLayerEndcap = cms.int32(3)
    )
