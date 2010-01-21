import FWCore.ParameterSet.Config as cms

#
# module to perform MVA training for jet-parton association
#
trainTtSemiLepJetCombMVA = cms.EDAnalyzer("TtSemiLepJetCombMVATrainer",
    #-------------------------------------------------
    # sources (leptons, jets, MET, jet-parton matching)
    #-------------------------------------------------
    leps     = cms.InputTag("selectedLayer1Muons"),
    jets     = cms.InputTag("selectedLayer1Jets"),
    mets     = cms.InputTag("layer1METs"),
    matching = cms.InputTag("ttSemiLepJetPartonMatch"),                                       

    # ------------------------------------------------
    # select semileptonic signal channel
    # (all others are taken as background for training)
    # either "kElec", "kMuon" or "kTau"
    # ------------------------------------------------
    leptonType = cms.string("kMuon"),

    # ------------------------------------------------
    # maximum number of jets to be considered
    # (has to be >= 4, can be set to -1 if you
    # want to take all)
    # ------------------------------------------------
    maxNJets = cms.int32(4)
)


