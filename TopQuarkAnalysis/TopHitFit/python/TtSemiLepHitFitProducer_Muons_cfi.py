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
    # option to use b-tagging
    # ------------------------------------------------
    bTagAlgo          = cms.string("trackCountingHighEffBJetTags"),
    minBDiscBJets     = cms.double(1.0),
    maxBDiscLightJets = cms.double(3.0),
    useBTagging       = cms.bool(False),
    
    # ------------------------------------------------
    # set mass values used in the constraints
    # set mass to 0 for no constraint
    # ------------------------------------------------    
    mW   = cms.double(80.4),
    mTop = cms.double(0.),
    
    # ------------------------------------------------
    # specify jet correction level as, Uncorrected, L1Offset, L2Relative, L3Absolute, L4Emf,
    # L5Hadron, L6UE, L7Parton, a flavor specification will be added automatically, when
    # chosen
    # ------------------------------------------------
    jetCorrectionLevel = cms.string("L3Absolute"),
    
    # ------------------------------------------------
    # rescale jets
    # ------------------------------------------------
    jes  = cms.double(1.0),
    jesB = cms.double(1.0),
)


