import FWCore.ParameterSet.Config as cms

solutions = cms.EDProducer("TtSemiEvtSolutionMaker",
    metSource = cms.InputTag("selectedLayer1METs"),
    maximalDistance = cms.double(0.3),
    addLRSignalSel = cms.bool(True),
    electronSource = cms.InputTag("selectedLayer1Electrons"),
    matchToGenEvt = cms.bool(True),
    maxF = cms.double(0.0001),
    matchingAlgorithm = cms.int32(0),
    jetCorrectionScheme = cms.int32(0),
    leptonFlavour = cms.string('muon'),
    maxNrIter = cms.int32(200),
    lepParametrisation = cms.int32(0),
    nrCombJets = cms.uint32(4),
    lrSignalSelObs = cms.vint32(-1),
    jetSource = cms.InputTag("selectedLayer1Jets"),
    lrSignalSelFile = cms.string('TopQuarkAnalysis/TopEventSelection/data/TtSemiLRSignalSelSelObsAndPurity.root'),
    doKinFit = cms.bool(True),
    useDeltaR = cms.bool(True),
    lrJetCombObs = cms.vint32(-1),
    jetParametrisation = cms.int32(0),
    muonSource = cms.InputTag("selectedLayer1Muons"),
    lrJetCombFile = cms.string('TopQuarkAnalysis/TopJetCombination/data/TtSemiLRJetCombSelObsAndPurity.root'),
    addLRJetComb = cms.bool(True),
    maxDeltaS = cms.double(5e-05),
    useMaximalDistance = cms.bool(True),
    metParametrisation = cms.int32(0),
    constraints = cms.vint32(1)
)


