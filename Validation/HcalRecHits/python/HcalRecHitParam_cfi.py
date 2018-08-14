import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hcalRecoAnalyzer = DQMEDAnalyzer('HcalRecHitsValidation',
    TopFolderName             = cms.string('HcalRecHitsV/HcalRecHitTask'),
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
    EBRecHitCollectionLabel   = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
    EERecHitCollectionLabel   = cms.InputTag("ecalRecHit:EcalRecHitsEE"),

    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('yes'),

    SimHitCollectionLabel = cms.untracked.InputTag("g4SimHits","HcalHits"),

    TestNumber                = cms.bool(False)
)

hcalNoiseRates = DQMEDAnalyzer('NoiseRates',
    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),
    rbxCollName  = cms.untracked.InputTag('hcalnoise'),
    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5),
    noiselabel   = cms.InputTag('hcalnoise')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( hcalRecoAnalyzer, SimHitCollectionLabel = cms.untracked.InputTag("fastSimProducer","HcalHits") )

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
(run2_HCAL_2017 & ~fastSim).toModify( hcalRecoAnalyzer, TestNumber = cms.bool(True) )
