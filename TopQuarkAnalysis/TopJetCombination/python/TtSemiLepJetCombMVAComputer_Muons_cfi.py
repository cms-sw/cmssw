import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiLepJetCombMVA = cms.EDProducer("TtSemiLepJetCombMVAComputer",
    #-------------------------------------------------
    # sources
    #-------------------------------------------------
    jets     = cms.InputTag("selectedLayer1Jets"),
    leptons  = cms.InputTag("selectedLayer1Muons"),
    #-------------------------------------------------
    # number of jets to be considered in combinatorics
    # (has to be >= 4, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    maxNJets = cms.int32(4),
    #-------------------------------------------------
    # maximum number of jet combinations finally
    # written into the event, starting from the "best"
    # (has to be >= 1, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    maxNComb = cms.int32(1)
)
