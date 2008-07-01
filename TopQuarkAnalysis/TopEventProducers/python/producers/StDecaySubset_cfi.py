import FWCore.ParameterSet.Config as cms

#
# module to make a persistent copy of all status 2
# (equivalent) genParticles of the top decay 
#
decaySubset_SingleTop = cms.EDProducer("StDecaySubset",
    SwitchChainType = cms.int32(1),         # 1: use daughter-mother-grandmother relationship
                                            # 2: directly look at initial state
    src = cms.InputTag("genParticles")
)


