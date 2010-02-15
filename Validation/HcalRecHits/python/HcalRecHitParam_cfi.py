import FWCore.ParameterSet.Config as cms

hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
    outputFile = cms.untracked.string(''),
    eventype = cms.untracked.string('single'),
    mc = cms.untracked.string('yes'),
    sign = cms.untracked.string('*'),
    hcalselector = cms.untracked.string('HB'),
    ecalselector = cms.untracked.string('yes')
)



