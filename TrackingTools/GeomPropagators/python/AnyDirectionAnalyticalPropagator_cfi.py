import FWCore.ParameterSet.Config as cms

AnyDirectionAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnyDirectionAnalyticalPropagator'),
    PropagationDirection = cms.string('anyDirection')
)


# foo bar baz
# 6gc0JKGbUavSg
# fd4ABL68104Rf
