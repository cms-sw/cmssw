import FWCore.ParameterSet.Config as cms

SkimSummary = cms.EDAnalyzer(
    'SkimSummary',
    HltLabel = cms.InputTag("TriggerResults","","skim"),
    #maxPaths = cms.untracked.int(25),
    )
