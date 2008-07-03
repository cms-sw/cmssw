import FWCore.ParameterSet.Config as cms

ttSemiEvtKit = cms.EDFilter("TtSemiEvtKit",
    enable = cms.string(''),
    outputTextName = cms.string('TtSemiEvtKit_output.txt'),
    disable = cms.string('all'),
    EvtSolution = cms.InputTag("solutions"),
    ntuplize = cms.string('lrJetCombProb,lrSignalEvtProb,kinFitProbChi2')
)
