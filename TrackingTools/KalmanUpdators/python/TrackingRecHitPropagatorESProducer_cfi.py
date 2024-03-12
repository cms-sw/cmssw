import FWCore.ParameterSet.Config as cms

trackingRecHitPropagator = cms.ESProducer("TrackingRecHitPropagatorESProducer",
    ComponentName = cms.string('trackingRecHitPropagator'),
    SimpleMagneticField = cms.string('')
#    SimpleMagneticField = cms.string('ParabolicMf')
)


# foo bar baz
# nEm3mPNnp3y3U
