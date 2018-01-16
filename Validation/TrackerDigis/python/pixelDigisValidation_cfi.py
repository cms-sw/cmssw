import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pixelDigisValid = DQMEDAnalyzer('SiPixelDigiValid',
    src = cms.InputTag("simSiPixelDigis"),
    outputFile = cms.untracked.string(''),
    runStandalone = cms.bool(False)
)

# This customization will be removed once we have phase2 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(pixelDigisValid, src = 'simSiPixelDigis:Pixel')

