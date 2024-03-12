import FWCore.ParameterSet.Config as cms

KullbackLeiblerDistance5D = cms.ESProducer("DistanceBetweenComponentsESProducer5D",
    ComponentName = cms.string('KullbackLeiblerDistance5D'),
    DistanceMeasure = cms.string('KullbackLeibler')
)


# foo bar baz
# w53zCA3FEbfHy
