import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hcalDigiAnalyzer = DQMEDAnalyzer('HcalDigiTester',
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    zside = cms.untracked.string('*'),
    outputFile = cms.untracked.string(''),
    hcalselector = cms.untracked.string('HB')
)



