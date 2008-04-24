import FWCore.ParameterSet.Config as cms

pixelDigisValid = cms.EDFilter("SiPixelDigiValid",
    src = cms.InputTag("simSiPixelDigis"),
    outputFile = cms.untracked.string('pixeldigihisto.root')
)



