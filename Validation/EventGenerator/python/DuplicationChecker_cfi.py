import FWCore.ParameterSet.Config as cms

duplicationChecker = cms.EDAnalyzer("DuplicationChecker",
    generatedCollection = cms.InputTag("generator",""),
    searchForLHE = cms.bool(False)                                
)
