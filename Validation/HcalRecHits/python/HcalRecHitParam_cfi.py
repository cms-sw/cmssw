import FWCore.ParameterSet.Config as cms

hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),

    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('yes'),

    SimHitCollectionLabel = cms.untracked.InputTag("g4SimHits","HcalHits"),

    TestNumber                = cms.bool(False)
)

hcalNoiseRates = cms.EDAnalyzer('NoiseRates',
    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),
    rbxCollName  = cms.untracked.InputTag('hcalnoise'),
    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5),
    noiselabel   = cms.InputTag('hcalnoise')
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hcalRecoAnalyzer, TestNumber = cms.bool(True) )

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( hcalRecoAnalyzer, SimHitCollectionLabel = cms.untracked.InputTag("famosSimHits","HcalHits") )
