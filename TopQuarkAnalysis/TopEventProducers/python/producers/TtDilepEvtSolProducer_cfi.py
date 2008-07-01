import FWCore.ParameterSet.Config as cms

#
# module to build dileptonic ttbar event solutions
# (one solution for each possible jet combination)
#
solutions = cms.EDProducer("TtDilepEvtSolutionMaker",
    evtSource      = cms.InputTag("genEvt"),
    metSource      = cms.InputTag("selectedLayer1METs"),
    tauSource      = cms.InputTag("selectedLayer1Taus"),
    muonSource     = cms.InputTag("selectedLayer1Muons"),
    electronSource = cms.InputTag("selectedLayer1Electrons"),
    jetSource      = cms.InputTag("selectedLayer1Jets"),

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
    bestSolFromMC = cms.bool(False)
)
