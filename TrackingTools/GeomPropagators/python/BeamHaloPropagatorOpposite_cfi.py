import FWCore.ParameterSet.Config as cms

BeamHaloPropagatorOpposite = cms.ESProducer("BeamHaloPropagatorESProducer",
    ComponentName = cms.string('BeamHaloPropagatorOpposite'),
    CrossingTrackerPropagator = cms.string('BeamHaloSHPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    EndCapTrackerPropagator = cms.string('BeamHaloMPropagatorOpposite')
)


# foo bar baz
# o5gv1EykSnXIg
# hPCmZo7ggmiUZ
