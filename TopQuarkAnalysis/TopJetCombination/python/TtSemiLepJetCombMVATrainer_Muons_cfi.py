import FWCore.ParameterSet.Config as cms

#
# module to make mvaTraining for jet parton associations
#
trainTtSemiLepJetCombMVA = cms.EDAnalyzer("TtSemiLepJetCombMVATrainer",
    leptons  = cms.InputTag("selectedLayer1Muons"),
    jets     = cms.InputTag("selectedLayer1Jets"),
    matching = cms.InputTag("ttSemiLepJetPartonMatch"),                                       

    # ------------------------------------------------
    # select semileptonic signal channel
    # (all others are taken as background for training)
    # 1: electron, 2: muon, 3: tau
    # ------------------------------------------------
    lepChannel = cms.int32(2),

    # ------------------------------------------------
    # maximum number of jets to be considered
    # (has to be >= 4, can be set to -1 if you
    # want to take all)
    # ------------------------------------------------
    maxNJets = cms.int32(4)
)


