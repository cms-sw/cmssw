import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopObjectResolutions.stringResolutions_etEtaPhi_cff import *

kinFitTtFullHadEvent = cms.EDProducer("TtFullHadKinFitProducer",
    jets = cms.InputTag("selectedPatJets"),

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
    # option to use b-tagging
    # ------------------------------------------------
    useBTagging         = cms.bool(True),
    bTagAlgo            = cms.string("trackCountingHighPurBJetTags"),
    minBTagValueBJet    = cms.double(3.0),
    maxBTagValueNonBJet = cms.double(3.0),
    bTags               = cms.uint32(2), # minimal number of b-tagged
                                         # jets, if more are available
                                         # they will be used

    # ------------------------------------------------
    ## specify jet correction level as, Uncorrected,
    ## L1Offset, L2Relative, L3Absolute, L4Emf, 
    ## L5Hadron, L6UE, L7Parton, a flavor specifica-
    ## tion will be added automatically, when chosen     
    # ------------------------------------------------
    jetCorrectionLevel = cms.string("L3Absolute"),
                                      
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
    mTop = cms.double(173.),

    # ------------------------------------------------
    # resolutions used for the kinematic fit
    # ------------------------------------------------
    udscResolutions             = udscResolution.functions,
    bResolutions                = bjetResolution.functions,

    # ------------------------------------------------
    # set correction factor(s) for the jet energy resolution:
    # - (optional) eta dependence assumed to be symmetric
    #   around eta=0, i.e. parametrized in |eta|
    # - any negative value as last bin edge is read as "inf"
    # - make sure that number of entries in vector with
    #   bin edges = number of scale factors + 1
    # ------------------------------------------------
    jetEnergyResolutionScaleFactors = cms.vdouble(1.0),
    jetEnergyResolutionEtaBinning = cms.vdouble(0.0,-1.0)
)


