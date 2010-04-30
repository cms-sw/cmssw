import FWCore.ParameterSet.Config as cms

#
# module to build dileptonic ttbar event solutions
# (one solution for each possible jet combination)
#
solutions = cms.EDProducer("TtDilepEvtSolutionMaker",
    evtSource      = cms.InputTag("genEvt"),
    metSource      = cms.InputTag("patMETs"),
    tauSource      = cms.InputTag("selectedPatTaus"),
    muonSource     = cms.InputTag("selectedPatMuons"),
    electronSource = cms.InputTag("selectedPatElectrons"),
    jetSource      = cms.InputTag("selectedPatJets"),

    ## considered channels                           
    mutauChannel   = cms.bool(True),
    etauChannel    = cms.bool(True),
    emuChannel     = cms.bool(True),
    eeChannel      = cms.bool(True),
    mumuChannel    = cms.bool(True),
    tautauChannel  = cms.bool(True),

    ## choose jet correction scheme
    jetCorrectionScheme = cms.int32(0),

    ## match to gen event?
    matchToGenEvt = cms.bool(True),
                           
    ## configuration of top mass calculation                           
    calcTopMass = cms.bool(True),
    tmassbegin = cms.double(100.0),
    tmassend   = cms.double(300.0),
    tmassstep  = cms.double(1.0),

    nrCombJets = cms.uint32(3),
    bestSolFromMC = cms.bool(False),
     
    neutrino_parameters = cms.vdouble(30.7137,
                                      56.2880,
				      23.0744,
				      59.1015,
				      24.9145)
)
