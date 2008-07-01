import FWCore.ParameterSet.Config as cms

#
# module to build the single top event solutions (one solution for each possible jet combination)
#
solutions = cms.EDProducer("StEvtSolutionMaker",
    metSource = cms.InputTag("selectedLayer1METs"),
    jetParametrisation = cms.int32(0), ## 0: EMom, 1: EtEtaPhi, 2: EtThetaPhi

    muonSource = cms.InputTag("selectedLayer1Muons"),
    jetSource  = cms.InputTag("selectedLayer1Jets"),
    # select the jet energy scale correction scheme to be used
    jetCorrectionScheme = cms.int32(0),
    electronSource = cms.InputTag("selectedLayer1Electrons"),
    doKinFit  = cms.bool(True),
    maxNrIter = cms.int32(200),
    maxDeltaS = cms.double(5e-05),
    lrJetCombFile = cms.string('TopQuarkAnalysis/TopJetCombination/data/to_be_added.root'),
    # 2 = use flavour of role of jet in event
    # other = use standard MCJet calibrations (i.e. no flavour distinction)
    matchToGenEvt = cms.bool(False),
    metParametrisation = cms.int32(0), ## 0: EMom, 1: EtEtaPhi, 2: EtThetaPhi

    lepParametrisation = cms.int32(0), ## 0: EMom, 1: EtEtaPhi, 2: EtThetaPhi

    addLRJetComb = cms.bool(False),
    maxF = cms.double(0.0001),
    leptonFlavour = cms.string('muon'), ##electron or muon

    constraints = cms.vint32(1, 2) ##1: Wlep, 2:tlep, 3:nu-mass

)


