import FWCore.ParameterSet.Config as cms

SmartPropagator = cms.ESProducer("SmartPropagatorESProducer",
                                 ComponentName = cms.string('SmartPropagator'),
                                 TrackerPropagator = cms.string('PropagatorWithMaterial'),
                                 MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
                                 PropagationDirection = cms.string('alongMomentum'),
                                 Epsilon = cms.double(5.0)
                                 )


