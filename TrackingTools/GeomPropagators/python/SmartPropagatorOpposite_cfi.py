import FWCore.ParameterSet.Config as cms

SmartPropagatorOpposite = cms.ESProducer("SmartPropagatorESProducer",
                                         ComponentName = cms.string('SmartPropagatorOpposite'),
                                         TrackerPropagator = cms.string('PropagatorWithMaterialOpposite'),
                                         MuonPropagator = cms.string('SteppingHelixPropagatorOpposite'),
                                         PropagationDirection = cms.string('oppositeToMomentum'),
                                         Epsilon = cms.double(5.0)
                                         )


