import FWCore.ParameterSet.Config as cms

initSubset = cms.EDProducer("TopInitSubset",
    src = cms.InputTag("genParticles")
)


