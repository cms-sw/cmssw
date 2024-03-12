import FWCore.ParameterSet.Config as cms

CloseComponentsMerger5D = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('CloseComponentsMerger5D'),
    MaxComponents = cms.int32(12),
    DistanceMeasure = cms.string('KullbackLeiblerDistance5D')
)


# foo bar baz
# q0dYYzLPrMPAX
# 07ZjuQh6aNGL0
