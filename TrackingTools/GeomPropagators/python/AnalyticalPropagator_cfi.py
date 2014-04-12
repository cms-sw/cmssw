import FWCore.ParameterSet.Config as cms

AnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnalyticalPropagator'),
    PropagationDirection = cms.string('alongMomentum')
)


