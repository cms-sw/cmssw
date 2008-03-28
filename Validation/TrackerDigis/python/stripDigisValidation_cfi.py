import FWCore.ParameterSet.Config as cms

stripDigisValid = cms.EDFilter("SiStripDigiValid",
    src = cms.InputTag("siStripDigis","ZeroSuppressed"),
    outputFile = cms.untracked.string('stripdigihisto.root'),
    verbose = cms.untracked.bool(True)
)


