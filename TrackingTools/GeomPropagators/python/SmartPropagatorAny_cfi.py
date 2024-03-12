import FWCore.ParameterSet.Config as cms

SmartPropagatorAny = cms.ESProducer("SmartPropagatorESProducer",
                                    ComponentName = cms.string('SmartPropagatorAny'),
                                    TrackerPropagator = cms.string('PropagatorWithMaterial'),
                                    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
                                    PropagationDirection = cms.string('alongMomentum'),
                                    Epsilon = cms.double(5.0)
                                    )


# foo bar baz
# 6eQVMvSgpo1DJ
# COGS1I2jAqGlW
