import FWCore.ParameterSet.Config as cms

decaySubset_SingleTop = cms.EDProducer("StDecaySubset",
    # use daughter-mother-grandmother relationship (SwitchChainType=1) or directly look at initial state (SwitchChainType=2, useful for SingleTop generator)
    SwitchChainType = cms.int32(1),
    src = cms.InputTag("genParticles")
)


