import FWCore.ParameterSet.Config as cms

#
# module to build semi-leptonic ttbar event solutions
# (one solution for each possible jet combination)
#
solutions = cms.EDProducer("TtSemiEvtSolutionMaker",
    metSource = cms.InputTag("patMETs"),
    muonSource = cms.InputTag("selectedPatMuons"),
    electronSource = cms.InputTag("selectedPatElectrons"),
    jetSource = cms.InputTag("selectedPatJets"),

    ## considered channel
    leptonFlavour = cms.string('muon'),

    ## choose jet correction scheme
    jetCorrectionScheme = cms.int32(0),

    ## match to gen event?
    matchToGenEvt      = cms.bool(True),
    matchingAlgorithm  = cms.int32(0),
    useDeltaR          = cms.bool(True),
    maximalDistance    = cms.double(0.3),
    useMaximalDistance = cms.bool(True),
                           
    ## configure kinematic fit
    doKinFit = cms.bool(True),
    maxNrIter = cms.int32(200),
    maxDeltaS = cms.double(5e-05),
    maxF = cms.double(0.0001),
    constraints = cms.vuint32(1),
    jetParametrisation = cms.int32(0),
    metParametrisation = cms.int32(0),
    lepParametrisation = cms.int32(0),

    ## configuration of private LH ratio method                           
    addLRJetComb = cms.bool(True),
    lrJetCombObs    = cms.vint32(-1),
    lrJetCombFile   = cms.string('TopQuarkAnalysis/TopJetCombination/data/TtSemiLRJetCombSelObsAndPurity.root'),

    addLRSignalSel  = cms.bool(True),
    lrSignalSelObs  = cms.vint32(-1),
    lrSignalSelFile = cms.string('TopQuarkAnalysis/TopEventSelection/data/TtSemiLRSignalSelSelObsAndPurity.root'),

    nrCombJets = cms.uint32(4)
)


