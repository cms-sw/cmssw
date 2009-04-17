import FWCore.ParameterSet.Config as cms

kinFitTtSemiLepEvent = cms.EDProducer("TtSemiLepKinFitProducerMuon",
    jets = cms.InputTag("selectedLayer1Jets"),
    leps = cms.InputTag("selectedLayer1Muons"),
    mets = cms.InputTag("selectedLayer1METs"),

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
    # option to take only a given jet combination
    # instead of going through the full combinatorics
    # ------------------------------------------------
    match = cms.InputTag("findTtSemiLepJetCombMVA"),
    useOnlyMatch = cms.bool(False),

    # ------------------------------------------------
    # settings for the KinFitter
    # ------------------------------------------------    
    maxNrIter = cms.uint32(500),
    maxDeltaS = cms.double(5e-05),
    maxF      = cms.double(0.0001),
    # ------------------------------------------------
    # select parametrisation
    # 0: EMom, 1: EtEtaPhi, 2: EtThetaPhi
    # ------------------------------------------------
    jetParametrisation = cms.uint32(1),
    lepParametrisation = cms.uint32(1),
    metParametrisation = cms.uint32(1),
    # ------------------------------------------------
    # set constraints
    # 1: Whad-mass, 2: Wlep-mass
    # 3: thad-mass, 4: tlep-mass, 5: nu-mass
    # ------------------------------------------------                                   
    constraints = cms.vuint32(1, 2)
)


