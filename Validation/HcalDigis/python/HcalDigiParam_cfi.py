import FWCore.ParameterSet.Config as cms

hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigiTester",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    zside = cms.untracked.string('*'),
    outputFile = cms.untracked.string(''),
    hcalselector = cms.untracked.string('HB')
)



