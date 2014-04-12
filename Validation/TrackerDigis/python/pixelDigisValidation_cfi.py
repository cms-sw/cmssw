import FWCore.ParameterSet.Config as cms

pixelDigisValid = cms.EDAnalyzer("SiPixelDigiValid",
    src = cms.InputTag("simSiPixelDigis"),
    outputFile = cms.untracked.string(''),
    runStandalone = cms.bool(False)
)



