import FWCore.ParameterSet.Config as cms

decaySubset = cms.EDProducer("TopDecaySubset",
    src = cms.InputTag("genParticles")
)


