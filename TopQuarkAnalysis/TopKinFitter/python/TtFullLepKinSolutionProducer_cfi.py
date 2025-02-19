import FWCore.ParameterSet.Config as cms

kinSolutionTtFullLepEvent = cms.EDProducer("TtFullLepKinSolutionProducer",
    jets = cms.InputTag("selectedPatJets"),    
    electrons = cms.InputTag("selectedPatElectrons"),
    muons = cms.InputTag("selectedPatMuons"),       
    mets = cms.InputTag("patMETs"),
    
    # ------------------------------------------------
    # specify jet correction step and flavor correction
    # ------------------------------------------------
    jetCorrectionLevel = cms.string("L3Absolute"),


    # ------------------------------------------------
    # maximum number of jets to be considered in the
    # jet combinatorics (has to be >= 2
    # ------------------------------------------------
    maxNJets = cms.int32(2),
    
    #-------------------------------------------------       
    # maximum number of jet combinations finally
    # written into the event, starting from the "best"
    # (has to be >= 1, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    maxNComb = cms.int32(1),

    # ------------------------------------------------
    # Select if also solutions with wrong charge lepton 
    # combinations are to be looked for, e.g. solutions
    # with 2 equally charged leptons
    # ------------------------------------------------
    searchWrongCharge = cms.bool(False),
    
    # ------------------------------------------------
    # Channel(s) for which solutions are searched
    # ------------------------------------------------
    eeChannel     = cms.bool(False),
    emuChannel    = cms.bool(False),
    mumuChannel   = cms.bool(False),
    etauChannel   = cms.bool(False),
    mutauChannel  = cms.bool(False),
    tautauChannel = cms.bool(False),
    
    # ------------------------------------------------
    # settings for the KinSolver
    # ------------------------------------------------    
    tmassbegin = cms.double(100),
    tmassend   = cms.double(300),
    tmassstep  = cms.double(  1),
    
    # ------------------------------------------------
    # neutrino reference spectrum
    # five parameters:
    #   norm, 
    #   position nu, 
    #   width nu,     
    #   postion nubar, 
    #   width nubar
    # ------------------------------------------------  
    neutrino_parameters = cms.vdouble(30.7137,
                                      56.2880,
				      23.0744,
				      59.1015,
				      24.9145)
      
)


