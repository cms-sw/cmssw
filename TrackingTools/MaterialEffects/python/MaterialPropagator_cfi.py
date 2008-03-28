import FWCore.ParameterSet.Config as cms

MaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('PropagatorWithMaterial'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False)
)


