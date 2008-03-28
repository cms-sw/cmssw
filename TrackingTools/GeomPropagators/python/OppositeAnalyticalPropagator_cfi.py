import FWCore.ParameterSet.Config as cms

OppositeAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnalyticalPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum')
)


