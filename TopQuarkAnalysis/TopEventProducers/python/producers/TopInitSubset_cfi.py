import FWCore.ParameterSet.Config as cms

#
# module to make a persistent copy of all top 
# mother genParticles
#
initSubset = cms.EDProducer("TopInitSubset",
    src = cms.InputTag("genParticles")
)


