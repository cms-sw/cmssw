import FWCore.ParameterSet.Config as cms

trainTtSemiJetCombMVA = cms.EDFilter("TtSemiJetCombMVATrainer",
    jets = cms.InputTag("selectedLayer1Jets"),
    # ------------------------------------------------
    # maximum number of jets to be considered
    # (has to be >= 4, can be set to -1 if you
    # want to take all)
    # ------------------------------------------------
    nJetsMax = cms.int32(4),
    # ------------------------------------------------
    # select semileptonic signal channel
    # (all others are taken as background for training)
    # 1: electron, 2: muon, 3: tau
    # ------------------------------------------------
    lepChannel = cms.int32(2),
    matching = cms.InputTag("ttSemiJetPartonMatch"),
    leptons = cms.InputTag("selectedLayer1Muons")
)


