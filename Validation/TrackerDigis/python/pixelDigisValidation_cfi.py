import FWCore.ParameterSet.Config as cms

pixelDigisValid = cms.EDFilter("SiPixelDigiValid",
    src = cms.InputTag("siPixelDigis"),
    outputFile = cms.untracked.string('pixeldigihisto.root')
)


