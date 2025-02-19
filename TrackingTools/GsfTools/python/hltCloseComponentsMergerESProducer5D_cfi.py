import FWCore.ParameterSet.Config as cms

hltCloseComponentsMergerESProducer5D = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('hltESPCloseComponentsMerger5D'),
    MaxComponents = cms.int32(12),
    DistanceMeasure = cms.string('hltESPKullbackLeiblerDistance5D')
)
