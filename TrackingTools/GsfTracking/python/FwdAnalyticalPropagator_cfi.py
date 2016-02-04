import FWCore.ParameterSet.Config as cms

fwdAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('fwdAnalyticalPropagator'),
    PropagationDirection = cms.string('alongMomentum')
)


