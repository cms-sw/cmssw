import FWCore.ParameterSet.Config as cms

hcalDigiAnalyzer = cms.EDFilter("HcalDigiTester",
    digiLabel = cms.InputTag("hcalUnsuppressedDigis"),
    zside = cms.untracked.string('*'),
    outputFile = cms.untracked.string('HcalDigisValidationHB.root'),
    hcalselector = cms.untracked.string('HB')
)



