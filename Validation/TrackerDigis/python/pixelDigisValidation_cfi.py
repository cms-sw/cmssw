import FWCore.ParameterSet.Config as cms

pixelDigisValid = cms.EDAnalyzer("SiPixelDigiValid",
    src = cms.InputTag("simSiPixelDigis"),
    outputFile = cms.untracked.string(''),
    runStandalone = cms.bool(False)
)

# This customization will be removed once we have phase2 pixel digis
from Configuration.StandardSequences.Eras import eras
eras.phase2_tracker.toModify(pixelDigisValid, src = cms.InputTag('simSiPixelDigis', "Pixel"))

