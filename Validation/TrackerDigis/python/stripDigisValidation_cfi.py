import FWCore.ParameterSet.Config as cms

stripDigisValid = cms.EDAnalyzer("SiStripDigiValid",
    src = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
    runStandalone = cms.bool(False),
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False)
)



