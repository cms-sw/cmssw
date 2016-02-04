import FWCore.ParameterSet.Config as cms

stripDigisValid = cms.EDAnalyzer("SiStripDigiValid",
    src = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False)
)



