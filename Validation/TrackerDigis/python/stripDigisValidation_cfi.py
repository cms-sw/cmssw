import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
stripDigisValid = DQMEDAnalyzer('SiStripDigiValid',
    src = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
    runStandalone = cms.bool(False),
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False)
)



