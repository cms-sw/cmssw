import FWCore.ParameterSet.Config as cms

#
# module to build the single top event solutions
# (one solution for each possible jet combination)
#
solutions = cms.EDProducer("StEvtSolutionMaker",
    metSource      = cms.InputTag("patMETs"),
    muonSource     = cms.InputTag("selectedPatMuons"),
    electronSource = cms.InputTag("selectedPatElectrons"),
    jetSource      = cms.InputTag("selectedPatJets"),

    ## lepton flavor
    leptonFlavour = cms.string('muon'),
                           
    ## choose jet correction scheme                       
    jetCorrectionScheme = cms.int32(0),

    ## match to gen event?
    matchToGenEvt = cms.bool(False),

    ## configuration of kinemtaic fit
    doKinFit  = cms.bool(True),
    maxNrIter = cms.int32(200),
    maxDeltaS = cms.double(5e-05),
    maxF      = cms.double(0.0001),
    constraints = cms.vint32(1, 2),
    jetParametrisation = cms.int32(0),
    metParametrisation = cms.int32(0),
    lepParametrisation = cms.int32(0),

    ## configuration of private LH ratio method
    addLRJetComb  = cms.bool(False),
    lrJetCombFile = cms.string('TopQuarkAnalysis/TopJetCombination/data/to_be_added.root')
)


