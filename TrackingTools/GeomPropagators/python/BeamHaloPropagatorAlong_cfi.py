import FWCore.ParameterSet.Config as cms

BeamHaloPropagatorAlong = cms.ESProducer("BeamHaloPropagatorESProducer",
    ComponentName = cms.string('BeamHaloPropagatorAlong'),
    CrossingTrackerPropagator = cms.string('BeamHaloSHPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    EndCapTrackerPropagator = cms.string('BeamHaloMPropagatorAlong')
)


# foo bar baz
# j5dyahiiXHL38
# btSjsIsJjuIuo
