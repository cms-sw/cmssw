import FWCore.ParameterSet.Config as cms

SmartPropagatorAnyOpposite = cms.ESProducer("SmartPropagatorESProducer",
                                            ComponentName = cms.string('SmartPropagatorAnyOpposite'),
                                            TrackerPropagator = cms.string('PropagatorWithMaterialOpposite'),
                                            MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
                                            PropagationDirection = cms.string('oppositeToMomentum'),
                                            Epsilon = cms.double(5.0)
                                            )


