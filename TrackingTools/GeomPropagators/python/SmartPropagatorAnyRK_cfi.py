import FWCore.ParameterSet.Config as cms

SmartPropagatorAnyRK = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAnyRK'),
    TrackerPropagator = cms.string('RKTrackerPropagator'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)


