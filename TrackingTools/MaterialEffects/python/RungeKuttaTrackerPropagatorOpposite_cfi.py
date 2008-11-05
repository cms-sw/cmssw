import FWCore.ParameterSet.Config as cms

RungeKuttaTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('RungeKuttaTrackerPropagatorOpposite'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(True)
)



