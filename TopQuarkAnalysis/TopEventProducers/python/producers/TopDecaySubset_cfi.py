import FWCore.ParameterSet.Config as cms

#
# module to make a persistent copy of all status 2
# (equivalent) genParticles of the top decay 
#
decaySubset = cms.EDProducer("TopDecaySubset",
    src = cms.InputTag("genParticles")
)


