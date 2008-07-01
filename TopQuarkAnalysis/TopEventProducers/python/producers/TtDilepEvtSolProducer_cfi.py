import FWCore.ParameterSet.Config as cms

solutions = cms.EDProducer("TtDilepEvtSolutionMaker",
    metSource = cms.InputTag("selectedLayer1METs"),
    etauChannel = cms.bool(True),
    tmassend = cms.double(300.0),
    muonSource = cms.InputTag("selectedLayer1Muons"),
    mutauChannel = cms.bool(True),
    jetSource = cms.InputTag("selectedLayer1Jets"),
    evtSource = cms.InputTag("genEvt"),
    eeChannel = cms.bool(True),
    tmassstep = cms.double(1.0),
    mumuChannel = cms.bool(True),
    electronSource = cms.InputTag("selectedLayer1Electrons"),
    tauSource = cms.InputTag("selectedLayer1Taus"),
    emuChannel = cms.bool(True),
    jetCorrectionScheme = cms.int32(0),
    tmassbegin = cms.double(100.0),
    matchToGenEvt = cms.bool(True),
    tautauChannel = cms.bool(True),
    calcTopMass = cms.bool(True),
    nrCombJets = cms.uint32(3),
    bestSolFromMC = cms.bool(False)
)


