import FWCore.ParameterSet.Config as cms

Geant4ePropagator = cms.ESProducer("GeantPropagatorESProducer",
                                   ComponentName = cms.string("Geant4ePropagator"),
                                   PropagationDirection=cms.string("alongMomentum"),
                                   ParticleName=cms.string("mu")
                                   )
