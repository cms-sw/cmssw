import FWCore.ParameterSet.Config as cms

kinFitTtFullHadEvent = cms.EDProducer("TtFullHadKinFitProducer",
    jets = cms.InputTag("selectedLayer1Jets"),

    # ------------------------------------------------
    # maximum number of jets to be considered in the
    # jet combinatorics (has to be >= 6, can be set to
    # -1 if you want to take all)
    # ------------------------------------------------
    maxNJets = cms.int32(6),

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
    match = cms.InputTag(""),
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

    # ------------------------------------------------
    # set constraints
    # 1: W1-mass, 2: W2-mass
    # 3: t1-mass, 4: t2-mass
    # 5: equal top-masses
    # ------------------------------------------------                                   
    constraints = cms.vuint32(1, 2, 5),

    # ------------------------------------------------
    # set mass values used in the constraints
    # ------------------------------------------------    
    mW   = cms.double(80.4),
    mTop = cms.double(173.)
)


