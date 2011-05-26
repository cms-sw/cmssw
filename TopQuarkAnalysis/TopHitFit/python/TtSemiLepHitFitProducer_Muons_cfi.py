import FWCore.ParameterSet.Config as cms

hitFitTtSemiLepEvent = cms.EDProducer("TtSemiLepHitFitProducerMuon",
    jets = cms.InputTag("selectedPatJets"),
    leps = cms.InputTag("selectedPatMuons"),
    mets = cms.InputTag("patMETs"),

    # ------------------------------------------------
    # maximum number of jets to be considered in the
    # jet combinatorics (has to be >= 4, can be set to
    # -1 if you want to take all)
    # ------------------------------------------------
    maxNJets = cms.int32(4),

    #-------------------------------------------------
    # maximum number of jet combinations finally
    # written into the event, starting from the "best"
    # (has to be >= 1, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    maxNComb = cms.int32(1),
                                      
    # ------------------------------------------------
    # set mass values used in the constraints
    # set mass to 0 for no constrain
    # ------------------------------------------------    
    mW   = cms.double(80.4),
    mTop = cms.double(0.),
)


