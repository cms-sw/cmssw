import FWCore.ParameterSet.Config as cms

kinFitTtSemiLepEvent = cms.EDProducer("TtSemiLepKinFitProducerElectron",
    jets = cms.InputTag("selectedPatJets"),
    leps = cms.InputTag("selectedPatElectrons"),
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
    # option to take only a given jet combination
    # instead of going through the full combinatorics
    # ------------------------------------------------
    match = cms.InputTag("findTtSemiLepJetCombMVA"),
    useOnlyMatch = cms.bool(False),

    # ------------------------------------------------
    # option to use b-tagging
    # ------------------------------------------------
    bTagAlgo          = cms.string("trackCountingHighEffBJetTags"),
    minBDiscBJets     = cms.double(1.0),
    maxBDiscLightJets = cms.double(3.0),
    useBTagging       = cms.bool(False),

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
    # 1: Whad-mass, 2: Wlep-mass, 3: thad-mass,
    # 4: tlep-mass, 5: nu-mass, 6: equal t-masses
    # 7: sum-pt conservation
    # ------------------------------------------------                                   
    constraints = cms.vuint32(1, 2),

    # ------------------------------------------------
    # set mass values used in the constraints
    # ------------------------------------------------    
    mW   = cms.double(80.4),
    mTop = cms.double(173.),
                                      
    # ------------------------------------------------
    # set correction factor for the jet resolution
    # ------------------------------------------------                                     
    jetEnergyResolutionSmearFactor = cms.double(1.0),
    etaDependentResSmearFactor     = cms.vdouble(1.0),
    etaBinningForSmearFactor       = cms.vdouble(0.0,0.5,1.1,1.7,2.3,-1.0)
)


