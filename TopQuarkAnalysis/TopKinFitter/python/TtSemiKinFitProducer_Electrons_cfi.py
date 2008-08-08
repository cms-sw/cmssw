import FWCore.ParameterSet.Config as cms

kinFitTtSemiEvent = cms.EDProducer("TtSemiKinFitProducerElectron",
    jets = cms.InputTag("selectedLayer1Jets"),
    leps = cms.InputTag("selectedLayer1Electrons"),
    mets = cms.InputTag("selectedLayer1METs"),

    # ------------------------------------------------
    # maximum number of jets to be considered in the
    # jet combinatorics (has to be >= 4)
    # ------------------------------------------------
    maxNJets = cms.uint32(4),

    # ------------------------------------------------
    # option to take only a given jet combination
    # instead of going through the full combinatorics
    # ------------------------------------------------
    match = cms.InputTag("findTtSemiJetCombMVA"),
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
    constraints = cms.vint32(1, 2)
)


