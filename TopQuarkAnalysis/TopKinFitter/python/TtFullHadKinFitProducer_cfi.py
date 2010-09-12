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
    bTagAlgo            = cms.string("trackCountingHighPurBJetTags"),
    minBTagValueBJet    = cms.double(3.0),
    maxBTagValueNonBJet = cms.double(3.0),
    bTags               = cms.uint32(2), # if set to 1 also tries to take 2 if possible

    # ------------------------------------------------
    ## specify jet correction level as
    ## No Correction : raw                                     
    ## L1Offset      : off
    ## L2Relative    : rel
    ## L3Absolute    : abs
    ## L4Emf         : emf
    ## L5Hadron      : had
    ## L6UE          : ue
    ## L7Parton      : part
    ## a flavor specification will be
    ## added automatically, when chosen
    # ------------------------------------------------
    jetCorrectionLevel = cms.string("abs"),
                                      
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
    udscResolutions = udscResolution.functions,
    bResolutions    = bjetResolution.functions
)


